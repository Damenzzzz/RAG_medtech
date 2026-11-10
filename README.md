# RAG_medtech — Basic RAG for KazMedSim

FastAPI RAG service over 1207 Kazakhstan clinical protocols (Chroma vector index + parent
document store, Alem `text-1024` embeddings + Alem reranker, OpenAI GPT as the final model).
It is the diagnosis backend for the [MedTech / KazMedSim](https://github.com/Damenzzzz/MedTech)
frontend (`RAG_SERVICE_URL`). Deployed on Vercel as project `medtech-basic-rag`.

Recovered on 2026-11-10 from the Vercel CLI deployment — it was never in git before.

## Layout
- `basic_rag/basic_rag_server.py` — FastAPI app (`/health`, `/diagnose`, `/api/diagnose-jobs`, `/api/diagnose-jobs/{id}`)
- `basic_rag/parent_docstore/` — parent documents (in git)
- `basic_rag/chroma_db/` — Chroma index, `data/corpus/…clean.jsonl` — corpus: **not in git**, see below
- `app.py`, `vercel.json`, `scripts/vercel_build.py` — Vercel entrypoint and build-time asset restore

## Large assets
Published as GitHub Release `assets-v1` (Chroma sqlite split in two parts, HNSW index, corpus).

```bash
./scripts/fetch_assets.sh   # downloads + verifies SHA256 + unpacks
```

## Run
```bash
cp .env.example .env        # ALEM_API_KEY, EMBED_API_KEY, RERANK_API_KEY, OPENAI_API_KEY
./run_local.sh              # http://127.0.0.1:8080/health
# or
docker build -t medtech-basic-rag . && docker run --env-file .env -p 8080:8080 medtech-basic-rag
```

The full local stack (frontend + both RAG services + Supabase) lives in
`MedTech/docker/` and expects this repo cloned next to `MedTech/`.

## Other embedding models
`BASIC_RAG_CHROMA_DIR` selects the Chroma index. `scripts/reindex_chroma.py` re-embeds the
existing 22,100 chunks (same ids/metadata, so the parent docstore still matches) with any
OpenAI-compatible embeddings endpoint, e.g. Gemini via MedTech's `docker/gemini-proxy`:
```bash
python scripts/reindex_chroma.py --dst basic_rag/chroma_db_gemini \
  --base-url http://gemini-proxy:8000/v1 --model gemini-embedding-001:document
# run with BASIC_RAG_CHROMA_DIR=basic_rag/chroma_db_gemini EMBED_MODEL=gemini-embedding-001:query
```
