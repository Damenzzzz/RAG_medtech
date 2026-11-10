#!/usr/bin/env bash
# Download the large prebuilt RAG assets from the GitHub Release and unpack them:
#   basic_rag/chroma_db/                               (Chroma: sqlite + HNSW index)
#   data/corpus/merged_protocols_structured_clean.jsonl (1207 protocols)
# Needs the GitHub CLI (`gh auth login`) because the repo may be private.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${ASSETS_TAG:-assets-v1}"
REPO="${ASSETS_REPO:-Damenzzzz/RAG_medtech}"

if [[ -f basic_rag/chroma_db/chroma.sqlite3 && -f data/corpus/merged_protocols_structured_clean.jsonl ]]; then
  echo "Assets already present."; exit 0
fi

mkdir -p .vercel-build-assets
gh release download "$TAG" --repo "$REPO" --dir .vercel-build-assets --clobber
(cd .vercel-build-assets && shasum -a 256 -c SHA256SUMS)

mkdir -p basic_rag/chroma_db data/corpus
cat .vercel-build-assets/chroma.sqlite3.tar.gz.part-* | tar xzf - -C basic_rag/chroma_db
tar xzf .vercel-build-assets/chroma_hnsw.tar.gz -C basic_rag/chroma_db
tar xzf .vercel-build-assets/corpus.tar.gz -C data/corpus
echo "Assets restored."
