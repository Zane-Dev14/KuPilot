#!/bin/bash
set -e

echo "==> Starting K8s Failure Intelligence Copilot"

# Auto-ingest sample data on startup
echo "==> Ingesting sample data..."
python scripts/ingest.py || echo "Warning: Ingestion failed, continuing anyway..."

echo "==> Starting FastAPI server..."
exec uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
