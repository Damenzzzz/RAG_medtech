#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill API keys, then run again."
  exit 1
fi

uvicorn basic_rag.basic_rag_server:app --host 127.0.0.1 --port "${PORT:-8080}"
