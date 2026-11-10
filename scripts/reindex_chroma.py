"""Re-embed the existing Chroma collection with another embedding model.

Keeps every child chunk's id, text and metadata (incl. doc_id -> parent_docstore),
so ParentDocumentRetriever keeps working; only the vectors change. Resumable.

Usage (inside the basic-rag container or a venv with requirements.txt):
  python scripts/reindex_chroma.py --src basic_rag/chroma_db --dst basic_rag/chroma_db_gemini \
      --base-url http://gemini-proxy:8000/v1 --model gemini-embedding-001:document
Then run the server with BASIC_RAG_CHROMA_DIR=<dst> and EMBED_MODEL=<model>:query.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import chromadb
import requests

COLLECTION = "kz_protocols"


def embed(base_url: str, model: str, api_key: str, texts: list[str]) -> list[list[float]]:
    for attempt in range(6):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/embeddings",
                              json={"model": model, "input": texts},
                              headers={"Authorization": f"Bearer {api_key}"}, timeout=300)
            if r.status_code == 200:
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            print(f"  HTTP {r.status_code}: {r.text[:200]}", flush=True)
        except requests.RequestException as exc:
            print(f"  {type(exc).__name__}: {exc}", flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError("embedding failed after retries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="basic_rag/chroma_db")
    ap.add_argument("--dst", default="basic_rag/chroma_db_gemini")
    ap.add_argument("--base-url", default=os.getenv("EMBED_BASE_URL", "http://gemini-proxy:8000/v1"))
    ap.add_argument("--model", default="gemini-embedding-001:document")
    ap.add_argument("--api-key", default=os.getenv("EMBED_API_KEY", "local"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    src = chromadb.PersistentClient(path=args.src).get_collection(COLLECTION)
    dst_client = chromadb.PersistentClient(path=args.dst)
    dst = dst_client.get_or_create_collection(COLLECTION, configuration={"hnsw": {"space": "l2"}})

    total = src.count()
    done_ids = set(dst.get(include=[])["ids"]) if dst.count() else set()
    print(f"source {total} chunks, already in destination {len(done_ids)}", flush=True)

    pending = []
    page = 2000
    for offset in range(0, total, page):
        got = src.get(limit=page, offset=offset, include=["documents", "metadatas"])
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]):
            if i not in done_ids:
                pending.append((i, d or "", m))
    batches = [pending[i:i + args.batch] for i in range(0, len(pending), args.batch)]
    print(f"to embed: {len(pending)} chunks in {len(batches)} batches", flush=True)

    def work(batch):
        return batch, embed(args.base_url, args.model, args.api_key, [d for _, d, _ in batch])

    t0, n = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        for batch, vectors in ex.map(work, batches):
            dst.add(ids=[i for i, _, _ in batch], documents=[d for _, d, _ in batch],
                    metadatas=[m for _, _, m in batch], embeddings=vectors)
            n += len(batch)
            if n % (args.batch * 20) < args.batch or n == len(pending):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"  {n}/{len(pending)}  {rate:.0f}/s  eta {(len(pending) - n) / max(rate, 1e-6) / 60:.1f} min", flush=True)
    print(f"done: destination has {dst.count()} chunks", flush=True)


if __name__ == "__main__":
    main()
