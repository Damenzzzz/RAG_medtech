# Basic RAG (Chroma + parent docstore, 1207 KZ clinical protocols).
# Run scripts/fetch_assets.sh first so the Chroma index and corpus are present.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN test -f basic_rag/chroma_db/chroma.sqlite3 && test -f data/corpus/merged_protocols_structured_clean.jsonl \
    || (echo "Missing RAG assets: run scripts/fetch_assets.sh" && exit 1)
EXPOSE 8080
CMD ["uvicorn", "basic_rag.basic_rag_server:app", "--host", "0.0.0.0", "--port", "8080"]
