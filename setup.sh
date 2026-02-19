#!/bin/bash
# Complete setup & test script for K8s Failure Intelligence Copilot
# Run this after cloning to get everything working

set -e

PROJECT_ROOT="/Users/eric/IBM/Projects/courses/Deliverables/Week-2"
cd "$PROJECT_ROOT"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}K8s Failure Intelligence Copilot — Complete Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Check prerequisites
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}STEP 1: Checking prerequisites...${NC}"

# Check Python
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}❌ Python 3.11 not found. Please install it.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3.11${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install it.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose${NC}"

# Check Ollama is installed (not necessarily running)
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama not found in PATH. Please install it separately.${NC}"
    echo "    Get it from: https://ollama.ai"
    echo ""
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Fix Milvus (clean rebuild)
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}STEP 2: Fixing Milvus (clean rebuild)...${NC}"

echo "  • Stopping containers..."
docker compose down 2>/dev/null || true

echo "  • Removing old volumes..."
docker volume rm $(docker volume ls -q | grep -E 'milvus|etcd|minio' 2>/dev/null || echo "") 2>/dev/null || true

sleep 2

echo "  • Starting fresh Milvus stack..."
docker compose up -d

echo "  • Waiting for services to be healthy (30-60 seconds)..."
max_attempts=60
attempt=0

milvus_ready=false
while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    
    if curl -s http://localhost:19530 > /dev/null 2>&1 && \
       docker ps | grep milvus-standalone | grep -q healthy; then
        milvus_ready=true
        break
    fi
    
    if [ $((attempt % 10)) -eq 0 ]; then
        echo -e "    Attempt ${attempt}/${max_attempts}..."
    fi
    sleep 1
done

if [ "$milvus_ready" = true ]; then
    echo -e "${GREEN}✓ Milvus ready${NC}"
else
    echo -e "${YELLOW}⚠️  Milvus may not be fully ready, continuing anyway...${NC}"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Setup Python environment
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}STEP 3: Setting up Python environment...${NC}"

if [ ! -d "venv" ]; then
    echo "  • Creating virtual environment..."
    python3.11 -m venv venv
fi

echo "  • Activating virtual environment..."
source venv/bin/activate

echo "  • Installing dependencies..."
pip install -q -r requirements.txt

echo -e "${GREEN}✓ Python environment ready${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Run offline tests
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}STEP 4: Running offline unit tests...${NC}"
echo "  (These don't need Milvus or Ollama)"
echo ""

if pytest tests/test_basic.py -v --tb=short; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: System diagnostics
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}STEP 5: System diagnostics...${NC}"

python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/eric/IBM/Projects/courses/Deliverables/Week-2')

from src.config import get_settings
from src.vectorstore import MilvusStore

s = get_settings()
print(f"  • Embedding model: {s.embedding_model} ({s.embedding_dimension}-dim)")
print(f"  • Simple LLM: {s.simple_model}")
print(f"  • Complex LLM: {s.complex_model}")
print(f"  • Milvus URI: {s.milvus_uri}")
print(f"  • Complexity threshold: {s.query_complexity_threshold}")

m = MilvusStore()
if m.health_check():
    print(f"  • Milvus connection: ✓")
else:
    print(f"  • Milvus connection: ❌")
    sys.exit(1)
EOF

echo -e "${GREEN}✓ All systems ready${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Ingest sample data
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}STEP 6: Ingesting sample data...${NC}"

python scripts/ingest.py

echo -e "${GREEN}✓ Sample data ingested${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Start Ollama in a separate terminal:"
echo "   ${YELLOW}ollama serve${NC}"
echo ""
echo "2️⃣  Pull the required models (if not already cached):"
echo "   ${YELLOW}ollama pull llama3.1:8b-instruct-q8_0${NC}"
echo "   ${YELLOW}ollama pull deepseek-r1:32b  # Optional, for complex reasoning${NC}"
echo ""
echo "3️⃣  Run the interactive chatbot:"
echo "   ${YELLOW}python scripts/chat.py${NC}"
echo ""
echo "4️⃣  Or start the FastAPI server:"
echo "   ${YELLOW}python -m uvicorn src.api:app --reload${NC}"
echo ""
echo "Test questions to try:"
echo "  • ${YELLOW}What is CrashLoopBackOff?${NC}"
echo "  • ${YELLOW}Why would a pod have OOMKilled status?${NC}"
echo "  • ${YELLOW}My pod keeps crashing. Why and how do I fix it?${NC}"
echo ""
