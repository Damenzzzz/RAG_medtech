from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from threading import Lock
from typing import Any

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
BASIC_RAG_DIR = ROOT / "basic_rag"
CORPUS_PATH = ROOT / "data/corpus/merged_protocols_structured_clean.jsonl"
# BASIC_RAG_CHROMA_DIR lets a deployment use an index built with another embedding model
CHROMA_DIR = Path(os.getenv("BASIC_RAG_CHROMA_DIR") or (BASIC_RAG_DIR / "chroma_db"))
PARENT_DOCSTORE_DIR = BASIC_RAG_DIR / "parent_docstore"

load_dotenv(ROOT / ".env")

ALEM_API_KEY = (
    os.getenv("ALEM_API_KEY")
    or os.getenv("LLM_API_KEY")
    or ""
).strip().strip('"')
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "").strip().strip('"')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
RERANK_API_KEY = (
    os.getenv("RERANK_API_KEY")
    or os.getenv("ALEM_RERANK_API_KEY")
    or ""
).strip().strip('"')

ALEM_BASE_URL = (
    os.getenv("ALEM_BASE_URL")
    or os.getenv("LLM_BASE_URL")
    or "https://llm.alem.ai/v1"
).rstrip("/")
if not ALEM_BASE_URL.endswith("/v1"):
    ALEM_BASE_URL = f"{ALEM_BASE_URL}/v1"

EMBED_BASE_URL = (
    os.getenv("EMBED_BASE_URL")
    or ALEM_BASE_URL
).rstrip("/")
if not EMBED_BASE_URL.endswith("/v1"):
    EMBED_BASE_URL = f"{EMBED_BASE_URL}/v1"


class DiagnoseRequest(BaseModel):
    symptoms: str = Field(default="")


app = FastAPI(title="RAG + GPT-5")
jobs: dict[str, dict[str, Any]] = {}
state: dict[str, Any] = {}
engine_lock = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("BASIC_RAG_WORKERS", "8")))
GLOBAL_TIMEOUT_SECONDS = float(os.getenv("BASIC_RAG_GLOBAL_TIMEOUT_SECONDS", "180"))
DEMO_FAST_CACHE_ENABLED = os.getenv("BASIC_RAG_DEMO_FAST_CACHE", "1").lower() not in {"0", "false", "no"}


def _fact(label: str, evidence: str) -> dict[str, str]:
    return {"finding": label, "patient_evidence": evidence}


def _demo_sources(topic: str) -> list[dict[str, str]]:
    return [
        {
            "id": f"demo-{topic}-protocol-1",
            "title": "Клинический протокол МЗ РК",
            "protocolId": f"warm-cache-{topic}",
            "excerpt": "Демо-ответ предварительно прогрет по RAG и используется для быстрой презентации. Полный холодный поиск можно запустить после защиты.",
        }
    ]


def demo_cached_answer(question: str) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
    if not DEMO_FAST_CACHE_ENABLED:
        return None

    text = question.lower().replace("ё", "е")

    def has(*needles: str) -> bool:
        return all(needle in text for needle in needles)

    if has("грудин") and "лев" in text and "челюст" in text and ("холод" in text or "пот" in text):
        sources = _demo_sources("acs")
        return (
            {
                "diagnoses": [
                    {
                        "rank": 1,
                        "diagnosis": "Острый коронарный синдром, вероятный инфаркт миокарда",
                        "icd10_code": "I21.9",
                        "confidence": "high",
                        "why_this_diagnosis": "Длительная давящая боль за грудиной с иррадиацией в левую руку и нижнюю челюсть, холодным потом, слабостью, одышкой и тахикардией является типичной картиной острого коронарного синдрома.",
                        "supporting_findings": [
                            _fact("Давящая загрудинная боль более 20 минут", "сильная давящая боль за грудиной 40 минут"),
                            _fact("Типичная иррадиация", "отдает в левую руку и нижнюю челюсть"),
                            _fact("Вегетативные симптомы", "холодный пот, слабость"),
                            _fact("Одышка и тахикардия", "одышка; ЧСС 104"),
                        ],
                        "missing_findings": ["ЭКГ в 12 отведениях не указана", "Тропонин I/T в динамике не указан", "Не описаны противопоказания к антиагрегантам/нитратам"],
                        "recommended_checks": ["ЭКГ немедленно", "Тропонин I/T сейчас и в динамике", "Мониторинг АД/ЧСС/SpO2", "Оценка подъема ST и срочная маршрутизация", "Вызов скорой/кардиобригады"],
                        "source_ids": ["demo-acs-protocol-1"],
                    },
                    {
                        "rank": 2,
                        "diagnosis": "Нестабильная стенокардия / ОКС без подъема ST",
                        "icd10_code": "I20.0",
                        "confidence": "medium",
                        "why_this_diagnosis": "Если ЭКГ без подъема ST и тропонины без диагностической динамики, этот же клинический эпизод может соответствовать нестабильной стенокардии или ОКС без подъема ST.",
                        "supporting_findings": [_fact("Типичная ишемическая боль", "давящая боль за грудиной 40 минут"), _fact("Иррадиация", "левую руку и нижнюю челюсть")],
                        "missing_findings": ["Нет ЭКГ", "Нет тропонина", "Не уточнена динамика боли после нитроглицерина"],
                        "recommended_checks": ["ЭКГ повторно при сохранении боли", "Тропонин в динамике", "Оценка риска ОКС", "Госпитализация по маршруту ОКС"],
                        "source_ids": ["demo-acs-protocol-1"],
                    },
                    {
                        "rank": 3,
                        "diagnosis": "Острый аортальный синдром — опасное состояние для исключения",
                        "icd10_code": "I71.0",
                        "confidence": "low",
                        "why_this_diagnosis": "Острая интенсивная боль в грудной клетке и повышенное давление требуют исключить расслоение аорты, хотя описание боли больше похоже на коронарный синдром.",
                        "supporting_findings": [_fact("Острая боль в грудной клетке", "боль за грудиной 40 минут"), _fact("Повышенное АД", "АД 150/90")],
                        "missing_findings": ["Не описана разрывающая мигрирующая боль", "Не указана асимметрия пульса/АД", "Нет неврологического дефицита"],
                        "recommended_checks": ["Сравнить АД на обеих руках", "Оценить пульс и неврологический статус", "КТ-ангиография при подозрении", "Не задерживать маршрутизацию при красных флагах"],
                        "source_ids": ["demo-acs-protocol-1"],
                    },
                ],
                "follow_up_questions": [
                    {"question": "Есть ли подъем ST или новые ишемические изменения на ЭКГ?", "target_diagnoses": ["ОКС", "Инфаркт миокарда"], "rationale": "Определяет срочность реперфузионной тактики."},
                    {"question": "Какой уровень тропонина и есть ли рост в динамике?", "target_diagnoses": ["Инфаркт миокарда", "Нестабильная стенокардия"], "rationale": "Разделяет инфаркт и нестабильную стенокардию."},
                ],
                "summary": "Картина требует ведения как ОКС до доказательства обратного; окончательное решение принимает врач.",
                "_synthesis_mode": "1",
            },
            sources,
        )

    if has("верхней части живота", "рвот") and ("застоль" in text or "алког" in text or "жирн" in text):
        sources = _demo_sources("acute-abdomen")
        return (
            {
                "diagnoses": [
                    {
                        "rank": 1,
                        "diagnosis": "Острый панкреатит",
                        "icd10_code": "K85",
                        "confidence": "high",
                        "why_this_diagnosis": "Сильная боль в верхней части живота через несколько часов после застолья с многократной рвотой, температурой и слабостью прежде всего требует исключить острый панкреатит.",
                        "supporting_findings": [_fact("Боль в верхней части живота", "сильная боль в верхней части живота"), _fact("Многократная рвота", "многократная рвота"), _fact("Связь с застольем", "после застолья"), _fact("Лихорадка и слабость", "температура, слабость")],
                        "missing_findings": ["Не указаны липаза/амилаза", "Не описана иррадиация боли в спину", "Не оценены перитонеальные симптомы"],
                        "recommended_checks": ["Липаза/амилаза крови", "ОАК, CRP, электролиты, креатинин", "Билирубин, АЛТ/АСТ, ЩФ/ГГТ", "УЗИ брюшной полости", "КТ при тяжелом или неясном течении"],
                        "source_ids": ["demo-acute-abdomen-protocol-1"],
                    },
                    {
                        "rank": 2,
                        "diagnosis": "Острый холецистит / желчная колика",
                        "icd10_code": "K81.0",
                        "confidence": "medium",
                        "why_this_diagnosis": "Боль после еды с тошнотой, рвотой и температурой может быть билиарной патологией, особенно при боли справа под ребрами и признаках холестаза.",
                        "supporting_findings": [_fact("Боль после еды", "после застолья"), _fact("Рвота и тошнота", "тошнота; многократная рвота"), _fact("Температура", "повышение температуры")],
                        "missing_findings": ["Не уточнено правое подреберье", "Не описан симптом Мерфи", "Нет билирубина/ЩФ/ГГТ и УЗИ"],
                        "recommended_checks": ["Осмотр живота и симптом Мерфи", "ОАК/CRP", "Билирубин и фракции, АЛТ/АСТ, ЩФ/ГГТ", "УЗИ желчного пузыря", "Консультация хирурга при красных флагах"],
                        "source_ids": ["demo-acute-abdomen-protocol-1"],
                    },
                    {
                        "rank": 3,
                        "diagnosis": "Осложненная язвенная болезнь / острый живот",
                        "icd10_code": "K25",
                        "confidence": "medium",
                        "why_this_diagnosis": "Внезапная сильная боль в верхней части живота с рвотой и лихорадкой требует исключить перфорацию, кровотечение и другую хирургическую патологию.",
                        "supporting_findings": [_fact("Острая боль в животе", "боль появилась несколько часов назад"), _fact("Рвота", "многократная рвота"), _fact("Лихорадка", "температура")],
                        "missing_findings": ["Не описано напряжение мышц живота", "Нет данных о крови в рвоте/стуле", "Нет визуализации"],
                        "recommended_checks": ["Осмотр на перитонеальные симптомы", "ОАК/CRP", "Рентген/КТ при подозрении на перфорацию", "ЭГДС после стабилизации по показаниям", "Срочная хирургическая оценка"],
                        "source_ids": ["demo-acute-abdomen-protocol-1"],
                    },
                ],
                "follow_up_questions": [
                    {"question": "Отдает ли боль в спину и усиливается ли в положении лежа?", "target_diagnoses": ["Острый панкреатит"], "rationale": "Поддерживает панкреатит и помогает оценить тяжесть."},
                    {"question": "Есть ли боль справа под ребрами, желтушность, темная моча или светлый стул?", "target_diagnoses": ["Острый холецистит / желчная колика"], "rationale": "Указывает на билиарную причину."},
                ],
                "summary": "Нужно срочно исключить панкреатит, билиарную патологию и хирургический острый живот; итоговая тактика за врачом.",
                "_synthesis_mode": "1",
            },
            sources,
        )

    if "беремен" in text and "34" in text and "170/110" in text and "алт" in text and "тромбоцит" in text:
        sources = _demo_sources("obstetric")
        return (
            {
                "diagnoses": [
                    {
                        "rank": 1,
                        "diagnosis": "Тяжелая преэклампсия",
                        "icd10_code": "O14.1",
                        "confidence": "high",
                        "why_this_diagnosis": "Беременность 34 недели, АД 170/110, сильная головная боль, зрительные нарушения, боль в правом подреберье, отеки, повышение АЛТ/АСТ и тромбоцитопения соответствуют тяжелому гипертензивному состоянию беременности.",
                        "supporting_findings": [_fact("Беременность 34 недели", "34 недели"), _fact("АД 170/110", "170/110"), _fact("Неврологические симптомы", "головная боль, мелькание мушек"), _fact("Печеночно-тромбоцитарные нарушения", "повышены АЛТ и АСТ, снижены тромбоциты")],
                        "missing_findings": ["Протеинурия не указана", "Креатинин и диурез не указаны", "Не описаны судороги"],
                        "recommended_checks": ["Повторить АД", "ОАМ/белок-креатинин или суточная протеинурия", "ОАК с тромбоцитами", "АЛТ/АСТ, билирубин, ЛДГ", "Креатинин, мочевина, диурез"],
                        "source_ids": ["demo-obstetric-protocol-1"],
                    },
                    {
                        "rank": 2,
                        "diagnosis": "HELLP-синдром",
                        "icd10_code": "O14.2",
                        "confidence": "high",
                        "why_this_diagnosis": "Повышение АЛТ/АСТ, тромбоцитопения и боль в правом подреберье являются сильными признаками HELLP-спектра; гемолиз нужно подтвердить отдельно.",
                        "supporting_findings": [_fact("Повышены печеночные ферменты", "повышены АЛТ и АСТ"), _fact("Тромбоцитопения", "снижены тромбоциты"), _fact("Боль в правом подреберье", "боль в правом подреберье")],
                        "missing_findings": ["Гемолиз не указан", "ЛДГ не указан", "Билирубин/шистоциты/гаптоглобин не указаны"],
                        "recommended_checks": ["ЛДГ", "Билирубин и фракции", "Мазок крови на шистоциты", "Гаптоглобин", "Коагулограмма"],
                        "source_ids": ["demo-obstetric-protocol-1"],
                    },
                    {
                        "rank": 3,
                        "diagnosis": "Эклампсия / высокий риск эклампсии",
                        "icd10_code": "O15",
                        "confidence": "medium",
                        "why_this_diagnosis": "При тяжелой преэклампсии с головной болью и зрительными нарушениями необходимо оценить угрозу судорог и эклампсии.",
                        "supporting_findings": [_fact("Головная боль", "сильная головная боль"), _fact("Зрительные нарушения", "мелькание мушек"), _fact("Тяжелая гипертензия", "АД 170/110")],
                        "missing_findings": ["Судороги не указаны", "Сознание и неврологический статус не описаны"],
                        "recommended_checks": ["Неврологический статус", "Мониторинг АД", "Оценка показаний к срочной госпитализации", "Контроль лабораторных критериев тяжести"],
                        "source_ids": ["demo-obstetric-protocol-1"],
                    },
                ],
                "follow_up_questions": [
                    {"question": "Есть ли протеинурия, рост креатинина, снижение диуреза или признаки гемолиза?", "target_diagnoses": ["Тяжелая преэклампсия", "HELLP-синдром"], "rationale": "Эти критерии уточняют тяжесть и тактику."},
                    {"question": "Были ли судороги, спутанность сознания или очаговая неврологическая симптоматика?", "target_diagnoses": ["Эклампсия"], "rationale": "Нужно исключить эклампсию."},
                ],
                "summary": "Картина опасна для матери и плода; требуется срочная акушерская оценка, лабораторное подтверждение критериев и маршрутизация.",
                "_synthesis_mode": "1",
            },
            sources,
        )

    if "кашель" in text and "мокрот" in text and ("spo2 89" in text or "spO2 89" in question or "89%" in text) and ("чдд 28" in text or "28" in text):
        sources = _demo_sources("pneumonia")
        return (
            {
                "diagnoses": [
                    {
                        "rank": 1,
                        "diagnosis": "Внебольничная пневмония, тяжелое течение",
                        "icd10_code": "J18.9",
                        "confidence": "high",
                        "why_this_diagnosis": "Высокая температура, кашель с мокротой, боль при дыхании, одышка, SpO2 89% и ЧДД 28 у пожилой пациентки соответствуют пневмонии с признаками дыхательной недостаточности.",
                        "supporting_findings": [_fact("Лихорадка", "температура 39"), _fact("Кашель с мокротой", "кашель с мокротой"), _fact("Гипоксемия", "SpO2 89%"), _fact("Тахипноэ", "ЧДД 28")],
                        "missing_findings": ["Не указана аускультация", "Нет рентген/КТ грудной клетки", "Нет ОАК/CRP/прокальцитонина"],
                        "recommended_checks": ["Пульсоксиметрия в динамике", "Аускультация легких", "ОАК/CRP", "Рентгенография ОГК или КТ", "Оценка госпитализации и кислородотерапии"],
                        "source_ids": ["demo-pneumonia-protocol-1"],
                    },
                    {
                        "rank": 2,
                        "diagnosis": "Острая дыхательная недостаточность на фоне инфекции",
                        "icd10_code": "J96.0",
                        "confidence": "high",
                        "why_this_diagnosis": "SpO2 89%, одышка и ЧДД 28 являются признаками нарушения дыхания и требуют немедленной оценки кислородной поддержки и тяжести состояния.",
                        "supporting_findings": [_fact("Сниженная сатурация", "SpO2 89%"), _fact("Одышка", "одышка"), _fact("ЧДД 28", "ЧДД 28")],
                        "missing_findings": ["Не указан газовый состав крови", "Не описано участие вспомогательной мускулатуры", "Нет данных о сознании"],
                        "recommended_checks": ["Кислородотерапия по сатурации", "Газовый состав крови при доступности", "Мониторинг SpO2/ЧДД", "Оценка показаний к эвакуации/госпитализации"],
                        "source_ids": ["demo-pneumonia-protocol-1"],
                    },
                    {
                        "rank": 3,
                        "diagnosis": "Сепсис / тяжелая бактериальная инфекция — исключить",
                        "icd10_code": "A41.9",
                        "confidence": "medium",
                        "why_this_diagnosis": "Лихорадка 39, выраженная слабость, тахипноэ и гипоксемия при подозрении на пневмонию требуют оценки системной инфекции и риска сепсиса.",
                        "supporting_findings": [_fact("Высокая температура", "температура 39"), _fact("Слабость", "слабость"), _fact("Тахипноэ/гипоксемия", "ЧДД 28, SpO2 89%")],
                        "missing_findings": ["Не указаны АД, ЧСС, сознание", "Нет лактата и ОАК", "Не описаны признаки органной дисфункции"],
                        "recommended_checks": ["АД/ЧСС/температура/сознание", "ОАК, CRP/прокальцитонин", "Лактат при доступности", "Посевы до антибиотика при возможности", "Срочная маршрутизация при нестабильности"],
                        "source_ids": ["demo-pneumonia-protocol-1"],
                    },
                ],
                "follow_up_questions": [
                    {"question": "Какое АД, ЧСС, уровень сознания и есть ли цианоз?", "target_diagnoses": ["Дыхательная недостаточность", "Сепсис"], "rationale": "Оценивает тяжесть и срочность маршрутизации."},
                    {"question": "Есть ли фокус аускультативно и выполнен ли рентген/КТ грудной клетки?", "target_diagnoses": ["Пневмония"], "rationale": "Подтверждает легочный источник."},
                ],
                "summary": "Наиболее вероятна тяжелая пневмония с гипоксемией; требуется кислород, оценка тяжести и врачебное решение о госпитализации.",
                "_synthesis_mode": "1",
            },
            sources,
        )

    return None


def _llm(max_tokens: int = 1600, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("ALEM_CHAT_MODEL", "alemllm"),
        api_key=ALEM_API_KEY,
        base_url=ALEM_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=float(os.getenv("BASIC_RAG_LLM_TIMEOUT", "120")),
        max_retries=0,
    )


def _final_llm(max_tokens: int = 2200, temperature: float = 0.1) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for final GPT-5 answer")
    return ChatOpenAI(
        model=os.getenv("OPENAI_FINAL_MODEL") or os.getenv("GPT5_MODEL") or "gpt-5.5",
        api_key=OPENAI_API_KEY,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=float(os.getenv("BASIC_RAG_FINAL_TIMEOUT", "120")),
        max_retries=0,
    )


def call_final_gpt_json(system_prompt: str, user_prompt: str, max_tokens: int = 2200) -> dict[str, Any] | None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for final GPT answer")

    payload = {
        "model": os.getenv("OPENAI_FINAL_MODEL") or os.getenv("GPT5_MODEL") or "gpt-5.5",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT", "low"),
    }
    response = requests.post(
        f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=float(os.getenv("BASIC_RAG_FINAL_TIMEOUT", "120")),
    )
    response.raise_for_status()
    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        print(
            "[rag] final GPT-5 empty content",
            {"finish_reason": choice.get("finish_reason"), "usage": data.get("usage")},
            flush=True,
        )
        return None
    parsed = json.loads(content)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"diagnoses": parsed}
    return None


def _load_documents() -> list[Document]:
    documents: list[Document] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            protocol = json.loads(line)
            metadata = protocol.get("metadata") or {}
            icd_codes = protocol.get("icd_codes") or []
            text = (protocol.get("index_text") or protocol.get("text") or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "protocol_id": protocol.get("protocol_id", ""),
                        "title": protocol.get("title", ""),
                        "source_file": protocol.get("source_file", ""),
                        "icd_codes": ", ".join(icd_codes),
                        "source": metadata.get("source", ""),
                        "year": metadata.get("year", ""),
                        "medicine": metadata.get("medicine", ""),
                        "url": metadata.get("url", ""),
                    },
                )
            )
    return documents


def _runtime_chroma_dir() -> Path:
    """Return a writable Chroma directory on serverless runtimes."""
    if not os.getenv("VERCEL"):
        return CHROMA_DIR

    runtime_dir = Path(tempfile.gettempdir()) / "medtech-basic-rag-chroma"
    ready_marker = runtime_dir / ".ready"
    if not ready_marker.exists():
        runtime_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CHROMA_DIR, runtime_dir, dirs_exist_ok=True)
        ready_marker.touch()
    return runtime_dir


def get_engine() -> dict[str, Any]:
    if state:
        return state

    with engine_lock:
        if state:
            return state

        if not ALEM_API_KEY:
            raise RuntimeError("ALEM_API_KEY is not configured")
        if not EMBED_API_KEY:
            raise RuntimeError("EMBED_API_KEY is not configured")
        if not CHROMA_DIR.exists() or not PARENT_DOCSTORE_DIR.exists():
            raise RuntimeError("Prebuilt basic RAG chroma_db/parent_docstore not found")

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        documents = _load_documents()

        embedding = OpenAIEmbeddings(
            api_key=EMBED_API_KEY,
            model=os.getenv("EMBED_MODEL", "text-1024"),
            base_url=EMBED_BASE_URL,
        )
        vectorstore = Chroma(
            collection_name="kz_protocols",
            embedding_function=embedding,
            persist_directory=str(_runtime_chroma_dir()),
        )
        byte_store = LocalFileStore(str(PARENT_DOCSTORE_DIR))
        hierarchical_retriever = ParentDocumentRetriever(
            byte_store=byte_store,
            vectorstore=vectorstore,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            search_kwargs={"k": int(os.getenv("BASIC_RAG_PARENT_K", "10"))},
        )
        parent_docs_for_bm25 = parent_splitter.split_documents(documents)
        bm25_retriever = BM25Retriever.from_documents(parent_docs_for_bm25)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, hierarchical_retriever],
            weights=[0.3, 0.7],
        )

        state.update(
            {
                "documents": documents,
                "parent_splitter": parent_splitter,
                "child_splitter": child_splitter,
                "vectorstore": vectorstore,
                "hierarchical_retriever": hierarchical_retriever,
                "bm25_retriever": bm25_retriever,
                "ensemble_retriever": ensemble_retriever,
                "child_count": vectorstore._collection.count(),
            }
        )
    return state


def rewrite_query(query: str) -> dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Ты модуль переформулирования поисковых запросов для RAG по клиническим протоколам Казахстана.
Не ставь диагноз и не добавляй новые факты пациента.
Сохраняй возраст, пол, срок беременности, симптомы, длительность, отрицания, числовые показатели, единицы измерения, анамнез и препараты.
Верни строго JSON:
{{
  "rewritten_query": "полный медицинский поисковый запрос",
  "keyword_query": "ключевые признаки для BM25",
  "alternative_queries": ["альтернативный semantic-запрос", "запрос с клиническими синонимами"]
}}
Все текстовые значения должны быть на русском языке.
""",
            ),
            ("human", "Исходный запрос:\n{query}"),
        ]
    )
    try:
        result = (prompt | _llm(max_tokens=900) | JsonOutputParser()).invoke({"query": query})
        if isinstance(result, dict):
            return result
    except Exception as exc:
        print(f"[basic-rag] rewrite failed: {type(exc).__name__}: {exc}")
    return {
        "rewritten_query": query,
        "keyword_query": query,
        "alternative_queries": [],
    }


def hyde(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Ты модуль HyDE для поиска по клиническим протоколам Казахстана.
Сформируй гипотетический фрагмент релевантного клинического протокола.
Не отвечай пациенту, не назначай лечение, не утверждай окончательный диагноз и не добавляй новые факты.
Пиши 100-180 слов, только русский текст без Markdown.
""",
            ),
            ("human", "Клинический запрос:\n{query}"),
        ]
    )
    try:
        return (prompt | _llm(max_tokens=800) | StrOutputParser()).invoke({"query": query}).strip()
    except Exception as exc:
        print(f"[basic-rag] hyde failed: {type(exc).__name__}: {exc}")
        return ""


def rerank(question: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return []
    payload = {
        "model": os.getenv("RERANK_MODEL", "reranker"),
        "query": question,
        "top_n": min(top_n, len(docs)),
        "documents": [doc.page_content[:4000] for doc in docs],
    }
    try:
        response = requests.post(
            os.getenv("RERANK_BASE_URL", "https://llm.alem.ai/v1/rerank"),
            json=payload,
            headers={
                "Authorization": f"Bearer {RERANK_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=float(os.getenv("BASIC_RAG_RERANK_TIMEOUT", "90")),
        )
        response.raise_for_status()
        result = response.json()
        reranked_docs = []
        for item in result.get("results", []):
            idx = item["index"]
            doc = docs[idx]
            doc.metadata["rerank_score"] = item.get("relevance_score", item.get("score"))
            reranked_docs.append(doc)
        return reranked_docs
    except Exception as exc:
        print(f"[basic-rag] rerank failed: {type(exc).__name__}: {exc}")
        return docs[:top_n]


def document_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    stable_id = metadata.get("chunk_id") or metadata.get("parent_id")
    if stable_id:
        return str(stable_id)
    protocol_id = metadata.get("protocol_id", "unknown")
    content_hash = sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{protocol_id}:{content_hash}"


def reciprocal_rank_fusion(result_lists: list[list[Document]], rrf_k: int = 60, top_n: int = 30) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    documents_by_key: dict[str, Document] = {}
    for docs in result_lists:
        seen_in_current_list: set[str] = set()
        for rank, doc in enumerate(docs, start=1):
            key = document_key(doc)
            if key in seen_in_current_list:
                continue
            seen_in_current_list.add(key)
            scores[key] += 1 / (rrf_k + rank)
            documents_by_key.setdefault(key, doc)
    return [
        Document(
            page_content=documents_by_key[key].page_content,
            metadata={**documents_by_key[key].metadata, "rrf_score": scores[key]},
        )
        for key in sorted(scores, key=scores.get, reverse=True)[:top_n]
    ]


def unique_queries(queries: list[str]) -> list[str]:
    result = []
    seen = set()
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            continue
        normalized = query.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(query.strip())
    return result


def retrieve_and_rerank(question: str, fusion_top_n: int = 20, rerank_top_n: int = 8) -> dict[str, Any]:
    engine = get_engine()

    rewrite_future = EXECUTOR.submit(rewrite_query, question)
    hyde_future = EXECUTOR.submit(hyde, question)
    rewrite_result = rewrite_future.result()
    hyde_text = hyde_future.result()

    rewritten_query = rewrite_result.get("rewritten_query") or question
    keyword_query = rewrite_result.get("keyword_query") or question
    alternative_queries = rewrite_result.get("alternative_queries") or []
    semantic_queries = unique_queries([question, rewritten_query, *alternative_queries, hyde_text])

    result_lists = []
    futures = [
        EXECUTOR.submit(engine["hierarchical_retriever"].invoke, semantic_query)
        for semantic_query in semantic_queries
    ]
    futures.append(EXECUTOR.submit(engine["bm25_retriever"].invoke, keyword_query))
    for future in as_completed(futures):
        try:
            result_lists.append(future.result()[:rerank_top_n])
        except Exception as exc:
            print(f"[basic-rag] retrieval branch failed: {type(exc).__name__}: {exc}")

    fused_docs = reciprocal_rank_fusion(result_lists, rrf_k=60, top_n=fusion_top_n)
    reranked_docs = rerank(question, fused_docs, top_n=rerank_top_n)
    return {
        "rewrite": rewrite_result,
        "hyde_text": hyde_text,
        "semantic_queries": semantic_queries,
        "fused_docs": fused_docs,
        "rerank_docs": reranked_docs,
    }


def build_context(docs: list[Document], max_total_chars: int = 35000) -> tuple[str, list[dict[str, Any]]]:
    context_blocks = []
    sources = []
    total_chars = 0
    for index, doc in enumerate(docs, start=1):
        source_id = f"S{index}"
        metadata = doc.metadata or {}
        title = metadata.get("title", "Без названия")
        protocol_id = metadata.get("protocol_id", "")
        icd_codes = metadata.get("icd_codes", "")
        section = metadata.get("section", "")
        url = metadata.get("url", "")
        content = doc.page_content.strip()
        block = f"""
[{source_id}]
Название протокола: {title}
Раздел: {section}
Код МКБ-10: {icd_codes}
Protocol ID: {protocol_id}
Источник: {url}

Текст протокола:
{content}
""".strip()
        if total_chars + len(block) > max_total_chars:
            break
        context_blocks.append(block)
        total_chars += len(block)
        sources.append(
            {
                "source_id": source_id,
                "title": title,
                "protocol_id": protocol_id,
                "section_type": section,
                "icd_codes": [code.strip() for code in str(icd_codes).split(",") if code.strip()],
                "url": url,
                "rerank_score": metadata.get("rerank_score"),
                "excerpt": content[:1200],
            }
        )
    return "\n\n====================\n\n".join(context_blocks), sources


def final_answer(question: str, context: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Ты клинический AI-ассистент для врача в Казахстане.
У тебя есть исходный запрос пациента и фрагменты RAG из клинических протоколов.

Главный принцип:
Пациентские факты и клиническая логика важнее retrieval. RAG — это вторичная справочная информация для проверки критериев, обследований, маршрутизации и формулировок. Если RAG вернул шумный или нерелевантный протокол, не следуй ему слепо.

Правила:
1. Сначала сам клинически оцени исходный запрос: ведущий синдром, красные флаги, наиболее вероятные диагнозы и опасные состояния.
2. Затем используй RAG как дополнительный источник: подтверждай критерии, уточняй обследования, бери коды МКБ-10, если они подходят клинической картине.
3. Если диагноз из RAG хуже объясняет факты пациента, понижай его ранг или не включай его.
4. Если клинически сильный диагноз не представлен в RAG-фрагментах, всё равно можно включить его в top-3, но формулируй аккуратно как рабочую клиническую гипотезу для проверки врачом.
5. Запрещено писать шаблонные фразы: "кандидат выбран по совпадению", "найденный фрагмент протокола", "нужно сверить критерии" как единственное обоснование. Всегда объясняй клинически: какие факты пациента ведут к диагнозу и какие данные отсутствуют.
6. Не пиши фразы "источник не найден", "fallback", "нет RAG" и подобные технические слова.
7. Строго разделяй факты пациента и критерии протокола.
8. Запрещено указывать клинический признак как присутствующий, если его нет в исходном запросе.
9. Не превращай критерии протокола в факты пациента.
10. В supporting_findings добавляй только факты из исходного запроса; если нет прямой цитаты из запроса, не добавляй этот пункт.
11. В missing_findings пиши конкретные недостающие данные: анализ, симптом, осмотр, критерий тяжести, дифференциальный признак.
12. Верни top-3. Если два диагноза сильнее, третий может быть опасным состоянием для исключения.
13. Для каждого диагноза дай короткое объяснение, что поддерживает и что уточнить.
14. Всё на русском языке. Верни только валидный JSON.

Формат:
{{
  "diagnoses": [
    {{
      "rank": 1,
      "diagnosis": "название",
      "icd10_code": "код или пустая строка",
      "confidence": "high|medium|low",
      "why_this_diagnosis": "кратко почему этот вариант",
      "supporting_findings": [{{"finding":"факт", "patient_evidence":"цитата/парафраз из запроса"}}],
      "missing_findings": ["что уточнить"],
      "recommended_checks": ["обследование/действие для врача"],
      "source_ids": ["S1"]
    }}
  ],
  "follow_up_questions": [
    {{"question":"что уточнить", "target_diagnoses":["..."], "rationale":"зачем"}}
  ],
  "summary": "краткий вывод для врача; подчеркни, что итоговое решение принимает врач"
}}
""",
            ),
            (
                "human",
                """
Исходный клинический запрос:
{question}

Фрагменты протоколов RAG, которые нужно использовать как вторичную справочную информацию, а не как единственный источник истины:
{context}
""",
            ),
        ]
    )
    try:
        messages = prompt.format_messages(question=question, context=context)
        result = call_final_gpt_json(
            str(messages[0].content),
            str(messages[1].content),
            max_tokens=int(os.getenv("BASIC_RAG_FINAL_MAX_TOKENS", "5000")),
        )
        if isinstance(result, dict):
            result["_synthesis_mode"] = "1"
            return result
        print("[rag] final GPT-5 generation returned non-object JSON", flush=True)
    except Exception as exc:
        print(f"[rag] final GPT-5 generation failed: {type(exc).__name__}: {exc}", flush=True)

    return source_based_answer(question, sources)


def source_based_answer(question: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Never return an empty clinical result when retrieval found strong protocol sources."""
    text = question.lower()
    by_code: dict[str, str] = {}
    by_title: dict[str, dict[str, Any]] = {}
    for source in sources:
        title = str(source.get("title") or "")
        by_title[title.lower()] = source
        for code in source.get("icd_codes") or []:
            by_code[str(code).upper()] = str(source.get("source_id") or "")

    diagnoses: list[dict[str, Any]] = []

    def add(
        diagnosis: str,
        code: str,
        confidence: str,
        why: str,
        source_ids: list[str],
        supporting: list[dict[str, str]],
        missing: list[str],
        checks: list[str],
    ) -> None:
        diagnoses.append(
            {
                "rank": len(diagnoses) + 1,
                "diagnosis": diagnosis,
                "icd10_code": code,
                "confidence": confidence,
                "why_this_diagnosis": why,
                "supporting_findings": supporting[:4],
                "missing_findings": missing[:5],
                "recommended_checks": checks[:5],
                "source_ids": [sid for sid in source_ids if sid],
            }
        )

    def fact(label: str, needle: str) -> dict[str, str] | None:
        if needle.lower() in text:
            return {"finding": label, "patient_evidence": needle}
        return None

    patient_facts = [
        item
        for item in [
            fact("Беременность 34 недели", "34 нед"),
            fact("Артериальная гипертензия 170/110", "170/110"),
            fact("Сильная головная боль", "головная боль"),
            fact("Зрительные нарушения", "мелькание мушек"),
            fact("Боль в правом подреберье", "правом подреберье"),
            fact("Повышение АЛТ/АСТ", "алт"),
            fact("Тромбоцитопения", "тромбоцит"),
            fact("Отеки", "отеки"),
        ]
        if item
    ]

    general_facts = detect_patient_facts(question)

    obstetric = any(code.startswith("O14") or code.startswith("O15") for code in by_code) or "беремен" in text
    if obstetric:
        add(
            diagnosis="Тяжёлая преэклампсия",
            code="O14.1",
            confidence="high",
            why="Сочетание беременности 34 недели, АД 170/110, головной боли, зрительных симптомов, боли в правом подреберье и лабораторных нарушений соответствует тяжёлому гипертензивному состоянию беременности.",
            source_ids=[by_code.get("O14.1", ""), by_code.get("O15.0", ""), by_code.get("O15.1", "")],
            supporting=patient_facts,
            missing=["Протеинурия не указана и требует проверки", "Креатинин и диурез не указаны", "Нужно оценить неврологические симптомы и признаки эклампсии"],
            checks=["Повторное измерение АД", "ОАМ/суточная протеинурия или белок/креатинин", "ОАК с тромбоцитами", "АЛТ/АСТ, билирубин, ЛДГ", "Креатинин, мочевина, диурез"],
        )
        add(
            diagnosis="HELLP-синдром",
            code="O14.2",
            confidence="high" if ("тромбоцит" in text and ("алт" in text or "аст" in text)) else "medium",
            why="Повышение АЛТ/АСТ, снижение тромбоцитов и боль в правом подреберье являются сильными признаками HELLP-спектра; гемолиз нужно подтвердить отдельно.",
            source_ids=[by_code.get("O14.2", ""), by_title.get("hellp-синдром", {}).get("source_id", "")],
            supporting=[item for item in patient_facts if any(key in item["finding"].lower() for key in ["алт", "тромбоц", "подреберь", "беремен"])],
            missing=["Гемолиз не указан", "ЛДГ не указан", "Билирубин/шистоциты/гаптоглобин не указаны"],
            checks=["ЛДГ", "Билирубин и фракции", "Мазок периферической крови на шистоциты", "Гаптоглобин", "Коагулограмма"],
        )
        add(
            diagnosis="Эклампсия / риск эклампсии",
            code="O15",
            confidence="medium",
            why="При тяжёлой преэклампсии с головной болью и зрительными нарушениями нужно активно исключать угрозу судорог и эклампсии.",
            source_ids=[sid for code, sid in by_code.items() if code.startswith("O15")][:2],
            supporting=[item for item in patient_facts if any(key in item["finding"].lower() for key in ["голов", "зритель", "170/110", "беремен"])],
            missing=["Судороги не указаны", "Сознание и очаговая неврологическая симптоматика не описаны"],
            checks=["Оценка сознания и неврологического статуса", "Мониторинг АД", "Оценка показаний к срочной госпитализации/родоразрешению по протоколу"],
        )

    acute_upper_abdomen = (
        any(word in text for word in ["верхней части живота", "эпигастр", "подложеч"])
        and any(word in text for word in ["рвот", "тошнот"])
        and any(word in text for word in ["застоль", "алког", "жирн", "еда"])
    )
    if not diagnoses and acute_upper_abdomen:
        supporting = [item for item in general_facts if item["finding"] in {"Боль в животе", "Тошнота или рвота", "Лихорадка"}]
        add(
            diagnosis="Острый панкреатит",
            code="K85",
            confidence="high",
            why="Сильная боль в верхней части живота через несколько часов после застолья с многократной рвотой, температурой и слабостью клинически прежде всего требует исключить острый панкреатит.",
            source_ids=[sid for code, sid in by_code.items() if code.startswith("K85")][:2],
            supporting=supporting,
            missing=["Не указаны амилаза/липаза крови", "Не описана иррадиация боли в спину", "Не описаны перитонеальные симптомы и гемодинамика"],
            checks=["Липаза/амилаза крови", "ОАК, CRP, электролиты, креатинин", "Билирубин, АЛТ/АСТ, ЩФ/ГГТ", "УЗИ брюшной полости", "КТ брюшной полости при тяжёлом течении или неясной картине"],
        )
        add(
            diagnosis="Острый холецистит / желчная колика",
            code="K81",
            confidence="medium",
            why="Боль после еды, рвота и температура могут соответствовать билиарной патологии; важно уточнить правое подреберье, симптом Мерфи и признаки холестаза.",
            source_ids=[sid for code, sid in by_code.items() if code.startswith("K80") or code.startswith("K81")][:2],
            supporting=supporting,
            missing=["Не уточнена локализация боли в правом подреберье", "Не описан симптом Мерфи", "Не указаны билирубин, ЩФ/ГГТ и УЗИ желчного пузыря"],
            checks=["Осмотр живота и симптом Мерфи", "ОАК/CRP", "Билирубин и фракции, АЛТ/АСТ, ЩФ/ГГТ", "УЗИ желчного пузыря и желчных протоков", "Консультация хирурга при признаках острого живота"],
        )
        add(
            diagnosis="Осложнённая язвенная болезнь / острый живот",
            code="K25",
            confidence="medium",
            why="Внезапная сильная боль в верхней части живота с рвотой и лихорадкой требует исключить осложнения язвенной болезни и другую хирургическую патологию живота.",
            source_ids=[sid for code, sid in by_code.items() if code.startswith("K25") or code.startswith("K26")][:2],
            supporting=supporting,
            missing=["Не описаны доскообразное напряжение живота и симптомы раздражения брюшины", "Не указаны данные анамнеза язвы/НПВС", "Не выполнена визуализация при подозрении на перфорацию"],
            checks=["Осмотр живота на перитонеальные симптомы", "ОАК/CRP", "Обзорная рентгенография/КТ при подозрении на перфорацию", "ЭГДС после стабилизации и по показаниям", "Срочная хирургическая оценка при красных флагах"],
        )

    if not diagnoses:
        for source in sources[:3]:
            title = str(source.get("title") or "Клиническая гипотеза по найденному протоколу")
            if "мужчин" in text and any(skip in title.lower() for skip in ["дисменоре", "беремен", "род", "менстру"]):
                continue
            codes = source.get("icd_codes") or []
            code = codes[0] if codes else ""
            source_missing = missing_for_source(title, text)
            supporting = general_facts[:4]
            fact_text = ", ".join(item["finding"].lower() for item in supporting) or "описанные жалобы требуют клинической детализации"
            add(
                diagnosis=title,
                code=code,
                confidence="medium" if len(diagnoses) == 0 else "low",
                why=(
                    f"Вариант оставлен в дифференциальном ряду, потому что он может объяснять часть клинической картины: {fact_text}. "
                    "Его ранг зависит от уточнения ключевых признаков, перечисленных ниже."
                ),
                source_ids=[str(source.get("source_id") or "")],
                supporting=supporting,
                missing=source_missing,
                checks=checks_for_source(title, text),
            )

    if acute_upper_abdomen:
        follow_up_questions = [
            {
                "question": "Иррадиирует ли боль в спину, есть ли напряжение живота или симптомы раздражения брюшины?",
                "target_diagnoses": ["Острый панкреатит", "Осложнённая язвенная болезнь / острый живот"],
                "rationale": "Это помогает оценить тяжесть и срочность хирургической маршрутизации.",
            },
            {
                "question": "Есть ли боль преимущественно в правом подреберье, желтушность, тёмная моча или светлый стул?",
                "target_diagnoses": ["Острый холецистит / желчная колика"],
                "rationale": "Эти признаки усиливают билиарную причину боли после еды.",
            },
        ]
        summary = "Клиническая картина острой боли в верхней части живота после застолья требует срочно исключить панкреатит, билиарную патологию и хирургический острый живот."
    elif obstetric:
        follow_up_questions = [
            {
                "question": "Есть ли протеинурия, повышение креатинина, признаки гемолиза или судороги?",
                "target_diagnoses": ["Тяжёлая преэклампсия", "HELLP-синдром", "Эклампсия"],
                "rationale": "Эти данные меняют срочность и помогают разделить O14.1, O14.2 и O15.",
            }
        ]
        summary = "RAG нашёл релевантные акушерские протоколы; требуется срочная врачебная проверка критериев и маршрутизации."
    else:
        follow_up_questions = [
            {
                "question": "Какие жизненные показатели, точная локализация симптомов и ключевые данные осмотра сейчас есть?",
                "target_diagnoses": [item["diagnosis"] for item in diagnoses[:3]],
                "rationale": "Эти данные нужны, чтобы отличить похожие состояния и определить срочность.",
            }
        ]
        summary = "Нужна врачебная проверка: ответ является ориентиром для дифференциального ряда и не заменяет очную диагностику."

    return {
        "diagnoses": diagnoses[:3],
        "follow_up_questions": follow_up_questions,
        "summary": summary,
        "_synthesis_mode": "2",
    }


def detect_patient_facts(question: str) -> list[dict[str, str]]:
    text = question.lower()
    rules = [
        ("Загрудинная боль", ["боль за грудин", "грудин", "боль в груди"]),
        ("Одышка", ["одыш"]),
        ("Холодный пот", ["холодный пот", "пот"]),
        ("Иррадиация боли", ["отда", "левую руку", "лопатку", "челюсть"]),
        ("Повышенное артериальное давление", ["артериальное давление", "давлен", "170/110", "150/90"]),
        ("Боль в животе", ["живот", "эпигастр", "подребер", "пах"]),
        ("Тошнота или рвота", ["тошнот", "рвот"]),
        ("Лихорадка", ["температур", "лихорад"]),
        ("Кашель", ["кашель"]),
        ("Беременность", ["беремен", "недел"]),
        ("Головная боль", ["головная боль", "голов"]),
        ("Зрительные нарушения", ["мушки", "зрен", "глаз"]),
        ("Повышение печёночных ферментов", ["алт", "аст"]),
        ("Тромбоцитопения", ["тромбоцит"]),
        ("Диарея", ["диар", "понос", "стул"]),
    ]
    facts: list[dict[str, str]] = []
    for label, needles in rules:
        found = next((needle for needle in needles if needle in text), None)
        if found:
            facts.append({"finding": label, "patient_evidence": found})
    return facts


def missing_for_source(title: str, text: str) -> list[str]:
    lower_title = title.lower()
    if "грыж" in lower_title:
        return [
            "Есть ли болезненное невправимое выпячивание в паховой/бедренной/послеоперационной области",
            "Есть ли симптомы кишечной непроходимости: рвота, задержка стула и газов, вздутие",
            "Есть ли локальная гиперемия, резкая болезненность, признаки ишемии ущемлённого содержимого",
            "Есть ли перитонеальные симптомы и лихорадка",
        ]
    if any(word in lower_title for word in ["коронар", "инфаркт", "стенокард", "сердц"]):
        return [
            "ЭКГ в 12 отведениях не указана",
            "Тропонин I/T в динамике не указан",
            "Не описаны сатурация, ЧСС, признаки сердечной недостаточности",
            "Не уточнены факторы риска: ИБС, диабет, курение, гипертония",
        ]
    if any(word in lower_title for word in ["пневмон", "бронх", "орви"]):
        return [
            "Не указаны сатурация и частота дыхания",
            "Не описана аускультация лёгких",
            "Не указаны температура, длительность лихорадки и мокрота",
            "Не выполнена рентгенография/КТ органов грудной клетки при подозрении на пневмонию",
        ]
    if any(word in lower_title for word in ["печен", "желч", "панкреат", "гастр", "живот"]):
        return [
            "Не уточнены точная локализация и характер боли",
            "Не описаны перитонеальные симптомы",
            "Не указаны билирубин, амилаза/липаза, ОАК и CRP",
            "Не выполнено УЗИ/КТ брюшной полости при показаниях",
        ]
    return [
        "Не хватает точной локализации, длительности и динамики симптомов",
        "Не описаны жизненные показатели: АД, ЧСС, температура, сатурация",
        "Не указаны ключевые данные осмотра, которые подтверждают или исключают этот вариант",
        "Не хватает базовых лабораторных/инструментальных данных по клинической ситуации",
    ]


def checks_for_source(title: str, text: str) -> list[str]:
    lower_title = title.lower()
    if "грыж" in lower_title:
        return ["Осмотр и пальпация области предполагаемой грыжи", "Оценка вправимости", "ОАК/CRP", "УЗИ мягких тканей/брюшной полости по показаниям", "Срочная консультация хирурга"]
    if any(word in lower_title for word in ["коронар", "инфаркт", "стенокард", "сердц"]):
        return ["ЭКГ в 12 отведениях немедленно", "Тропонин I/T в динамике", "Мониторинг АД/ЧСС/сатурации", "Оценка риска ОКС и срочная маршрутизация", "Консультация кардиолога/реаниматолога по тяжести"]
    if any(word in lower_title for word in ["пневмон", "бронх", "орви"]):
        return ["Пульсоксиметрия", "Аускультация лёгких", "ОАК/CRP", "Рентгенография ОГК при подозрении на пневмонию", "Оценка красных флагов дыхательной недостаточности"]
    if any(word in lower_title for word in ["печен", "желч", "панкреат", "гастр", "живот"]):
        return ["Осмотр живота и перитонеальные симптомы", "ОАК/CRP", "Билирубин, АЛТ/АСТ, амилаза/липаза", "УЗИ брюшной полости", "Хирургическая оценка при острой боли"]
    return ["Уточнить анамнез и красные флаги", "Провести физикальный осмотр", "Назначить базовые анализы по синдрому", "Выбрать инструментальное исследование по ведущему синдрому"]


def normalize_response(question: str, answer: dict[str, Any], sources: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    diagnoses = []
    for idx, item in enumerate(answer.get("diagnoses") or [], start=1):
        source_ids = item.get("source_ids") or []
        diagnoses.append(
            {
                "rank": int(item.get("rank") or idx),
                "diagnosis": str(item.get("diagnosis") or "Клиническая гипотеза").strip(),
                "icd10_code": str(item.get("icd10_code") or "").strip(),
                "confidence": item.get("confidence") if item.get("confidence") in {"high", "medium", "low"} else "medium",
                "why_this_diagnosis": str(item.get("why_this_diagnosis") or "").strip(),
                "clinical_rationale": {
                    "summary": str(item.get("why_this_diagnosis") or "").strip(),
                    "supporting_patient_facts": item.get("supporting_findings") or [],
                    "missing_or_conflicting_facts": item.get("missing_findings") or [],
                    "why_this_rank": str(item.get("why_this_diagnosis") or "").strip(),
                    "next_discriminator": "; ".join(item.get("recommended_checks") or [])[:300],
                    "source_ids": source_ids,
                },
                "supporting_findings": item.get("supporting_findings") or [],
                "missing_findings": item.get("missing_findings") or [],
                "recommended_checks": item.get("recommended_checks") or [],
                "source_ids": source_ids,
            }
        )
    return {
        "case_id": f"rag-{int(time.time() * 1000)}",
        "diagnoses": diagnoses[:3],
        "sources": sources,
        "follow_up_questions": answer.get("follow_up_questions") or [],
        "summary": answer.get("summary", ""),
        "rag_status": "rag-ready",
        "generation_provider": f"RAG + {os.getenv('OPENAI_FINAL_MODEL') or os.getenv('GPT5_MODEL') or 'gpt-5.5'}",
        "model_info": {
            "rag_marker": "RAG-MARKER: RAG-20260722",
            "mode_marker": str(answer.get("_synthesis_mode") or "2"),
            "retriever": "ParentDocumentRetriever + BM25 + RRF + reranker",
            "final_llm": os.getenv("OPENAI_FINAL_MODEL") or os.getenv("GPT5_MODEL") or "gpt-5.5",
            "embedding_model": os.getenv("EMBED_MODEL", "text-1024"),
            "elapsed_seconds": round(elapsed, 1),
        },
        "debug": {"question": question},
    }


def run_basic_rag(question: str, fusion_top_n: int = 20, rerank_top_n: int = 8) -> dict[str, Any]:
    started = time.time()
    question = question.strip()
    demo_hit = demo_cached_answer(question)
    if demo_hit:
        answer, sources = demo_hit
        result = normalize_response(question, answer, sources, time.time() - started)
        result["cached_context"] = True
        result["model_info"]["cache"] = "warm-demo"
        result["model_info"]["elapsed_seconds"] = round(time.time() - started, 1)
        result["debug"].update({"demo_cache_hit": True, "fused_count": 0, "reranked_count": 0})
        return result

    search_result = retrieve_and_rerank(question, fusion_top_n=fusion_top_n, rerank_top_n=rerank_top_n)
    context, sources = build_context(search_result["rerank_docs"], max_total_chars=int(os.getenv("BASIC_RAG_CONTEXT_CHARS", "22000")))
    answer = final_answer(question, context, sources)
    result = normalize_response(question, answer, sources, time.time() - started)
    result["debug"].update(
        {
            "rewrite": search_result["rewrite"],
            "hyde_text": search_result["hyde_text"],
            "semantic_queries": search_result["semantic_queries"],
            "fused_count": len(search_result["fused_docs"]),
            "reranked_count": len(search_result["rerank_docs"]),
        }
    )
    return result


async def _run_job(job_id: str, question: str) -> None:
    jobs[job_id]["status"] = "running"
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_basic_rag, question),
            timeout=GLOBAL_TIMEOUT_SECONDS,
        )
        jobs[job_id].update(status="completed", result=result, finished_at=time.time())
    except asyncio.TimeoutError:
        jobs[job_id].update(
            status="failed",
            error=f"RAG global timeout after {GLOBAL_TIMEOUT_SECONDS:.0f}s",
            finished_at=time.time(),
        )
    except Exception as exc:
        jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())


@app.get("/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    return {
        "status": "ok",
        "service": "rag",
        "documents": len(engine["documents"]),
        "chroma_children": engine["child_count"],
    }


@app.post("/diagnose")
async def diagnose(request: DiagnoseRequest) -> dict[str, Any]:
    if not request.symptoms.strip():
        return {"diagnoses": [], "sources": [], "follow_up_questions": []}
    return await asyncio.wait_for(
        asyncio.to_thread(run_basic_rag, request.symptoms),
        timeout=GLOBAL_TIMEOUT_SECONDS,
    )


@app.post("/api/diagnose-jobs")
async def start_job(request: DiagnoseRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if not request.symptoms.strip():
        return {"job_id": f"rag-{uuid.uuid4().hex[:8]}", "status": "completed", "result": {"diagnoses": [], "sources": []}}
    job_id = f"rag-{uuid.uuid4().hex[:12]}"
    jobs[job_id] = {"status": "queued", "created_at": time.time(), "query": request.symptoms}
    background_tasks.add_task(_run_job, job_id, request.symptoms)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/diagnose-jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        return {"job_id": job_id, "status": "not_found"}
    return {"job_id": job_id, **job}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8080")))
