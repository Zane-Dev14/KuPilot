# Kubernetes Failure Intelligence Copilot

A production-grade Retrieval-Augmented Generation (RAG) system for intelligent diagnosis of Kubernetes failures, combining semantic embeddings, smart retrieval with reranking, and adaptive multi-model LLM selection.

Diagnose pod crashes, resource exhaustion, scheduling failures, networking issues, and configuration problems with AI-powered root cause analysis, explanations, and remediation steps.

## Features

- **Intelligent Document Ingestion**: Parse K8s manifests (YAML), structured/unstructured logs, Events, Helm charts, and markdown runbooks with domain-aware chunking
- **Multilingual Embeddings**: bge-m3 (384-dim) for efficient semantic representation
- **Smart Retrieval**: Query analyzer extracts K8s metadata, decomposes multi-part questions, metadata-filtered vector search
- **Learning-to-Rank**: bge-reranker-v2-m3 cross-encoder reranking for top-K precision
- **Adaptive LLM**: Fast llama3.1 for simple queries; deepseek-r1:32b for reasoning-heavy diagnoses (with exposed reasoning chain)
- **Structured Output**: Root cause, explanation, recommended fix, confidence score, sources
- **Multi-turn Conversations**: Session-based memory via LangGraph's InMemorySaver
- **LangGraph-Ready**: Architecture designed for easy integration of multi-step agent workflows and tool execution (Phase 2)
- **Production-Grade**: Type hints, structured logging, health checks, error handling, comprehensive testing hooks

## Prerequisites

- Python 3.11 or later
- Docker (or Rancher Desktop on macOS)
- Ollama (for local LLM inference)
- 4GB RAM minimum (8GB+ recommended for comfortable use)

### Optional: Apple Silicon (M1/M2/M3) Support

Set `EMBEDDING_DEVICE=mps` in `.env` for GPU-accelerated embeddings on Apple Silicon.

## Quick Start

### 1. Clone & Setup Python Environment

```bash
cd /Users/eric/IBM/Projects/courses/Deliverables/Week-2

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Milvus Vector Database

```bash
# Start Milvus (etcd + MinIO + Milvus Standalone)
docker compose up -d

# Wait for health checks to pass (20-30 seconds)
docker compose logs -f milvus-standalone | grep "Healthy"

# Verify on web UI (optional)
open http://localhost:9091/webui
```

### 3. Ensure Ollama is Running

```bash
# Start Ollama server (macOS)
ollama serve

# In another terminal, pull required models
ollama pull llama3.1        # For simple queries
ollama pull deepseek-r1:32b # For reasoning-heavy queries (optional)
```

### 4. Configure Application

```bash
# Copy example environment config
cp .env.example .env

# Edit if needed (values in .env.example are sensible defaults for localhost)
# Note: For Apple Silicon, change EMBEDDING_DEVICE=mps in .env
```

### 5. Verify Setup

```bash
# Check each component
python -c "from src.config import get_settings; print(get_settings())"
python -c "from src.embeddings.embedding_service import get_embeddings; e = get_embeddings(); print(f'Embeddings: {len(e.embed_query(\"test\"))}-dim')"
python -c "from src.vectorstore.milvus_client import MilvusVectorStore; m = MilvusVectorStore(); print('Milvus: OK' if m.health_check() else 'FAILED')"
```

### 6. Ingest Sample Data

```bash
python scripts/ingest.py --path data/sample/
```

Expected output:
```
Documents loaded: 12
Chunks created: 12
Chunks stored: 12
```

### 7. Run Interactive Chat

```bash
python scripts/chat.py

# Try: "Why is my pod crashing with ImagePullBackOff?"
# Or: "Explain CrashLoopBackOff and how to fix it"
# Or: "My pod is in Pending state in namespace prod, why?"
```

### 8. Start FastAPI Server

```bash
python -m uvicorn src.api.server:app --reload

# In another terminal, test:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is my pod crashing?"}'
```

## Project Structure

```
k8s-failure-intelligence-copilot/
├── IMPLEMENTATION_PLAN.md       # Detailed technical plan
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── docker-compose.yml           # Milvus stack (etcd + minio + milvus)
│
├── src/                         # Main application code
│   ├── config.py                # Pydantic settings from .env
│   ├── models/
│   │   └── schemas.py           # Pydantic data models
│   ├── embeddings/
│   │   └── embedding_service.py # bge-m3 embeddings (singleton)
│   ├── vectorstore/
│   │   └── milvus_client.py     # Milvus connection & collection mgmt
│   ├── ingestion/
│   │   ├── base_loader.py       # Abstract base loader
│   │   ├── yaml_loader.py       # K8s manifests & Helm
│   │   ├── log_loader.py        # K8s pod logs
│   │   ├── events_loader.py     # K8s Events
│   │   ├── markdown_loader.py   # Runbooks & docs
│   │   └── pipeline.py          # Orchestrator
│   ├── retrieval/
│   │   ├── query_analyzer.py    # Extract K8s metadata, decompose queries
│   │   ├── reranker.py          # bge-reranker-v2-m3 service
│   │   └── retriever.py         # LCEL retrieval chain
│   ├── chains/
│   │   ├── model_selector.py    # Query complexity estimation
│   │   └── rag_chain.py         # RAG diagnosis chain
│   ├── memory/
│   │   └── chat_memory.py       # Session memory (future)
│   ├── tools/
│   │   └── k8s_tools.py         # LangGraph-ready tool stubs
│   └── api/
│       └── server.py            # FastAPI endpoints
│
├── scripts/
│   ├── ingest.py                # CLI: ingest documents
│   └── chat.py                  # CLI: interactive diagnosis
│
├── data/sample/
│   ├── manifests/               # Sample K8s YAML files
│   ├── logs/                    # Sample pod logs
│   ├── events/                  # Sample K8s events (JSON)
│   └── docs/                    # Sample markdown runbooks
│
└── tests/
    ├── test_ingestion.py
    ├── test_embeddings.py
    ├── test_milvus.py
    ├── test_query_analyzer.py
    ├── test_reranker.py
    └── test_rag_chain.py
```

## Configuration Reference

All configuration is loaded from environment variables (see `.env.example`).

### Critical Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_URI` | `http://localhost:19530` | Milvus gRPC endpoint |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace model (384-dim) |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `mps` (Apple Silicon) |
| `SIMPLE_MODEL` | `llama3.1` | Fast LLM for simple queries |
| `COMPLEX_MODEL` | `deepseek-r1:32b` | Reasoning LLM (4-8x slower) |
| `QUERY_COMPLEXITY_THRESHOLD` | `0.7` | Score (0-1) to trigger deepseek |

### Optional Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `HNSW_M` | `16` | HNSW graph degree (larger = more accurate, slower) |
| `HNSW_EF_CONSTRUCTION` | `256` | HNSW index construction effort |
| `SEARCH_EF` | `128` | HNSW search effort |
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap for context continuity |
| `RETRIEVAL_TOP_K` | `4` | Final retrieval count (after reranking) |
| `RETRIEVAL_RERANK_K` | `10` | Retrieve before reranking |

## API Endpoints

### `/health` (GET)

Health check.

**Response:**
```json
{"status": "ok"}
```

### `/diagnose` (POST)

Diagnose a Kubernetes failure.

**Request:**
```json
{
  "question": "Why is my pod crashing with ImagePullBackOff?",
  "namespace": "prod",
  "force_model": "llama3.1",
  "session_id": "session_123"
}
```

**Response:**
```json
{
  "diagnosis": {
    "root_cause": "Container image not found in registry",
    "explanation": "The image gcr.io/myapp:invalid-tag does not exist in Google Container Registry",
    "recommended_fix": "1. Verify image name and tag\n2. Check registry credentials\n3. Re-push image if needed",
    "confidence": 0.95,
    "sources": ["data/sample/manifests/deployment-crashing.yaml"],
    "reasoning_model_used": "llama3.1",
    "thinking_chain": null
  },
  "session_id": "session_123"
}
```

### `/query-analysis` (POST)

Debug endpoint: show query analysis (metadata extraction, decomposition).

**Request:**
```json
{
  "question": "Why is pod web-app crashing in namespace prod and how do I fix it?"
}
```

**Response:**
```json
{
  "metadata": {
    "namespace": "prod",
    "pod": "web-app",
    "container": null,
    "node": null,
    "error_type": "crashed",
    "labels_dict": {}
  },
  "sub_queries": [
    "Why is pod web-app crashing in namespace prod",
    "how do I fix it"
  ]
}
```

### `/ingest` (POST)

Ingest documents into the knowledge base.

**Request:**
```json
{
  "path": "data/sample/",
  "doc_type": "yaml"
}
```

**Response:**
```json
{
  "documents_loaded": 12,
  "chunks_created": 12,
  "chunks_stored": 12,
  "errors": []
}
```

## Model Selection Strategy

The system automatically chooses between two LLMs based on query complexity:

1. **Simple Queries** (score < 0.7): `llama3.1`
   - Fast inference (~500ms)
   - Good for factual questions: "What is CrashLoopBackOff?"
   - Pattern matching: "My pod has status X, what does it mean?"

2. **Complex Queries** (score >= 0.7): `deepseek-r1:32b`
   - Slower inference (~3-5s) but reasoning-capable
   - Root cause analysis: "Why is this happening despite..."
   - Multi-step diagnosis: "Pod crashed AND service is unreachable, what's wrong?"
   - Ambiguous situations: "My pod might be crashing because..."

**Complexity Scoring:**
- Reasoning keywords ("why", "explain", "diagnose"): +0.3
- Multiple questions: +0.2 per additional question
- Long query (>200 chars): +0.1
- Uncertainty words ("maybe", "could", "possibly"): +0.2

You can override model selection via `force_model` parameter.

## Adding Custom Documents

### From CLI

```bash
# Ingest a single manifest
python scripts/ingest.py --path my-deployment.yaml --type yaml

# Ingest all logs in a directory
python scripts/ingest.py --path ./k8s-logs/ --type log

# Ingest Kubernetes events (exported as JSON)
python scripts/ingest.py --path events.json --type event

# Auto-detect type (recommended)
python scripts/ingest.py --path data/
```

### From API

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "data/sample/", "doc_type": null}'
```

### Custom Document Format

Each loader accepts raw files and outputs LangChain `Document` objects with structured metadata for filtering:

- **YAML**: kind, name, namespace, api_version, labels
- **Logs**: timestamp, pod_name, container, log_level
- **Events**: reason, event_type, involved_object_kind, involved_object_name, namespace, count
- **Markdown**: doc_type, source, header_hierarchy

Metadata enables filtered retrieval: "Show me logs from namespace prod with error_type=crashed"

## Troubleshooting

### Milvus Not Connecting

```bash
# Check docker containers
docker compose ps
# Expected: 3 running containers (milvus-standalone, milvus-etcd, milvus-minio)

# Check health
docker compose logs milvus-standalone | tail -20

# Rebuild if needed
docker compose down -v
docker compose up -d
```

### Ollama Models Not Found

```bash
# List available models
ollama list

# Pull required models
ollama pull llama3.1
ollama pull deepseek-r1:32b
```

### Embeddings Dimension Mismatch

```bash
# Verify actual embedding dimension
python -c "from src.embeddings.embedding_service import get_embeddings; e = get_embeddings(); print(len(e.embed_query('test')))"

# Should output 384. If different, update EMBEDDING_DIMENSION in .env
```

### No Documents Retrieved

1. Check ingestion completed: `docker compose exec milvus-standalone curl -s localhost:9091/webui` (Web UI shows collections)
2. Verify query analysis: `curl -X POST http://localhost:8000/query-analysis -H "Content-Type: application/json" -d '{"question": "test"}'`
3. Check vector dimension match: embeddings must be 384-dim, Milvus collection must expect 384-dim

### Slow Inference

- Check Ollama model is cached: `ollama pull llama3.1`
- For Apple Silicon: EMBEDDING_DEVICE=mps speeds up embeddings 3-5x
- deepseek-r1:32b is inherently slow (3-5s); use for genuinely complex queries only

## Development & Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_rag_chain.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Add Logging

All modules use `logging` module. Control verbosity via `LOG_LEVEL` in `.env`:

```bash
LOG_LEVEL=DEBUG  # Most verbose
LOG_LEVEL=INFO   # Standard
LOG_LEVEL=WARNING # Error/warning only
```

### Profile Performance

```python
import time
start = time.time()
diagnosis = rag_chain.diagnose("Test query")
print(f"Inference time: {time.time() - start:.2f}s")
```

### Inspect Milvus Data

```bash
# Web UI
open http://localhost:9091/webui
# Username: root, Password: empty

# Or via Python SDK
from pymilvus import Collection
coll = Collection("k8s_failures")
print(coll.num_entities)
```

## Architecture Decisions

### Why bge-m3?

- 384-dim (vs 1024-dim of bge-large-en-v1.5): 3x smaller storage, 3x faster search
- Multilingual: Supports K8s content in multiple languages
- Reranking recovers precision lost from smaller dimensionality

### Why reranking?

- Bge-reranker-v2-m3 is cross-encoder-based (query+document pairs scored together)
- NDCG gains typically 5-10% for top-K precision
- Latency impact negligible (~100-200ms per query) vs accuracy gain

### Why deepseek-r1:32b as fallback?

- Reasoning models significantly improve root cause analysis for ambiguous failures
- But 4-8x slower than base models; reserve for genuinely complex cases
- Exposed reasoning chain aids transparency and debugging

### Why LangChain LCEL?

- Composable, type-safe chains with LangChain v1
- Direct compatibility with LangGraph for Phase 2 (agents, multi-step workflows)
- Minimal migration effort vs custom orchestration code

## Next Steps

### Phase 2: LangGraph Agents & Tools

- Multi-step diagnostic workflows (e.g., "diagnose pod crash" → "check recent events" → "suggest fix" → "offer kubectl command")
- Tool execution: kubectl commands, shell scripts, log tail, metric queries
- Context passing and stateful diagnosis

### Phase 3: Production Deployment

- PostgreSQL checkpointer for distributed session memory
- LangSmith tracing for observability
- Prometheus metrics & Grafana dashboards
- Multi-GPU inference orchestration
- Kubernetes-native deployment (helm charts)

## Contributing

PRs welcome. Follow:

- Type hints on all functions
- Docstrings on classes & public methods
- Unit tests for new features
- Logging at key decision points

## License

[Your License Here]

## Support

Questions? Open an issue or contact the team.

---

**Status**: Phase 1 (core RAG pipeline) ready for implementation. Phase 2 (agents, tools, scaling) planned.
