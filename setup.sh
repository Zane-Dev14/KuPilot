#!/bin/zsh
# All-in-one setup for K8s Failure Intelligence Copilot (macOS / zsh)
# Usage: ./setup.sh
set -e
cd "$(dirname "$0")"

G='\033[0;32m' Y='\033[1;33m' R='\033[0;31m' B='\033[0;34m' N='\033[0m'
step() { echo "\n${Y}── $1${N}" }
ok()   { echo "${G}✓ $1${N}" }

echo "${B}═══════════════════════════════════════════════════════${N}"
echo "${B}  K8s Failure Intelligence Copilot — Setup${N}"
echo "${B}═══════════════════════════════════════════════════════${N}"

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
step "Checking prerequisites"
command -v python3 &>/dev/null || { echo "${R}❌ python3 not found${N}"; exit 1; }
ok "Python 3 ($(python3 --version 2>&1))"

# ── 2. Python env ────────────────────────────────────────────────────────────
step "Python environment"
[[ -d venv ]] || python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt
ok "Dependencies installed"

# ── 3. Check .env ────────────────────────────────────────────────────────────
step "Checking configuration"
if [[ ! -f .env ]]; then
    echo "${Y}⚠  No .env file found — copying from .env.example${N}"
    cp .env.example .env
    echo "${Y}   Edit .env and set your GOOGLE_API_KEY before running the app${N}"
fi
ok "Configuration checked"

# ── 4. Tests ─────────────────────────────────────────────────────────────────
step "Running offline tests"
pytest tests/test_basic.py -v --tb=short || { echo "${R}❌ Tests failed${N}"; exit 1; }
ok "All tests passed"

# ── 5. Pre-download embeddings ───────────────────────────────────────────────
step "Pre-downloading embedding model"
python3 -c "
from langchain_huggingface import HuggingFaceEmbeddings
HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True})
print('  Embeddings model cached')
"
ok "Embeddings ready"

# ── 6. Ingest sample data ───────────────────────────────────────────────────
step "Ingesting sample data into Chroma"
python3 scripts/ingest.py
ok "Data ingested"

# ── Done ──────────────────────────────────────────────────────────────────────
echo "\n${B}═══════════════════════════════════════════════════════${N}"
echo "${G}✅ Setup complete!${N}"
echo "${B}═══════════════════════════════════════════════════════${N}"
echo "
Next steps:
  1. Set your API key:  ${Y}export GOOGLE_API_KEY=your-key${N}  (or edit .env)
  2. CLI chat:          ${Y}python scripts/chat.py${N}
  3. Web UI:            ${Y}python -m uvicorn src.api:app --reload${N}
     Open ${Y}http://localhost:8000${N} in your browser
  4. Docker:            ${Y}docker compose up --build${N}
"
