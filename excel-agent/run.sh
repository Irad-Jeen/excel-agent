#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if [ -f .env ]; then
  set -a; source .env; set +a
fi

uvicorn app.main:app --reload --port 8000
