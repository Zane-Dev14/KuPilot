# Kubernetes Failure Intelligence Copilot — Phase 1 Final Delivery

**Date:** February 18, 2026  
**Version:** 1.0 — Planning & Scaffolding Complete  
**Status:** Ready for Implementation

---

## Executive Summary

**Objective:** Build a production-grade RAG system for intelligent diagnosis of Kubernetes failures.

**Scope:** Phase 1 covers core infrastructure, intelligent retrieval, document ingestion, and structured diagnosis generation. Phase 2 (agents, tools, scaling) deferred.

**Completion:** All planning, design, scaffolding, and documentation delivered. Code skeletons in `IMPLEMENTATION_PLAN.md` ready for implementation.

**Effort:** 13-20 hours to code Phase 1 (2500 LOC across 22 modules).

---

## What You're Getting

### 1. Complete Technical Specification

**[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** (5500+ lines)

- **22 detailed implementation steps** with full code skeletons
- Step-by-step module breakdown (config → embeddings → vectorstore → ingestion → retrieval → chains → API)
- Every class, method, and important logic explained
- Configuration reference and tuning guidance
- Verification checklist (10+ milestones)
- Architecture decisions with rationale

Copy-paste ready code blocks for all critical paths.

### 2. Quick-Start & API Reference

**[README.md](README.md)** (600+ lines)

- **5-minute quick start** (setup, ingest, query)
- API endpoint documentation with curl examples
- Configuration reference (all env vars explained)
- Model selection strategy
- Troubleshooting guide
- Development setup for testing

### 3. Infrastructure as Code

**[docker-compose.yml](docker-compose.yml)**
- Milvus standalone stack (etcd + minio + milvus-standalone)
- Health checks, volumes, networking
- One command: `docker compose up -d`

**[requirements.txt](requirements.txt)**
- 19 pinned dependencies, LangChain v1 + Ollama + Milvus stack
- Ready for `pip install -r requirements.txt`

**[.env.example](.env.example)**
- All configuration variables with sensible defaults
- Comments for each setting
- Copy to `.env` and customize

**[.gitignore](.gitignore)**
- Python standard ignores + project-specific patterns

### 4. Project Scaffolding

**11 Python packages** (src/ + tests/)
- Config layer
- Embeddings service
- Vector store client
- Ingestion pipeline (5 loaders)
- Retrieval tier (3 modules)
- Chains & LLM selection
- Memory management
- Tools (LangGraph-ready)
- FastAPI integration
- Test framework

All with `__init__.py` files and proper package structure.

### 5. Execution Summary

**[EXECUTION_SUMMARY.md](EXECUTION_SUMMARY.md)**
- Deliverables checklist
- Implementation roadmap (7 phases, parallelizable)
- Success criteria
- File inventory & LOC estimates
- Technology stack ratified
- Critical decisions locked

---

## Architecture at a Glance

```
User Query
    ↓
┌─────────────────────────────────┐
│  Query Analyzer                 │  Extract K8s metadata (namespace, pod, error_type, labels)
│  - extract_k8s_metadata()       │  Decompose multi-part questions
│  - decompose_query()            │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Vector Search (Milvus)         │  bge-m3 embeddings (384-dim)
│  - retrieval_rerank_k docs      │  HNSW index, cosine similarity
│  - Metadata filters             │  Namespace/pod filtering
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Reranking                      │  bge-reranker-v2-m3
│  - Cross-encoder scoring        │  Top-K precision +5-10% NDCG
│  - Top-K reordering             │  ~100-200ms latency
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Model Selection (Complexity)   │  Simple (llama3.1) vs Complex (deepseek-r1:32b)
│  - estimate_query_complexity()  │  Heuristic scoring: +0.3 for "why", +0.2 for multi-part, etc.
│  - select_model()               │  Activate deepseek only if needed
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  RAG Diagnosis Chain            │  LangChain LCEL pattern
│  - Retrieve context             │  Structured output: FailureDiagnosis
│  - Generate diagnosis           │  root_cause, explanation, fix, confidence, sources
│  - Expose reasoning (if deepseek)
└─────────────────────────────────┘
    ↓
Structured Output (JSON)
{
  "root_cause": "Image pull failed",
  "explanation": "Registry timeout",
  "recommended_fix": "Check registry credentials",
  "confidence": 0.92,
  "sources": ["manifest.yaml", "pod.log"],
  "reasoning_model_used": "llama3.1",
  "thinking_chain": null
}
```

---

## Technology Stack (Final)

| Layer | Component | Technology | Choice Rationale |
|-------|-----------|-----------|-----------------|
| **Embeddings** | Text → Vector | bge-m3 (384-dim) | Multilingual, 3x smaller than v1.5, reranking recovers recall |
| **Vector DB** | Similarity Search | Milvus (HNSW) | Battle-tested, cosine similarity, HNSW index optimal for text |
| **Reranking** | Relevance Scoring | bge-reranker-v2-m3 | CrossEncoder, consistent with embedding family, +5-10% NDCG gain |
| **Retrieval** | Query Intent | Custom Query Analyzer | K8s-specific metadata extraction, fast (<10ms), deterministic |
| **LLM (Simple)** | Fast Inference | llama3.1 via Ollama | ~500ms, instruction-following, suitable for 90% of queries |
| **LLM (Complex)** | Reasoning | deepseek-r1:32b via Ollama | ~3-5s, only for complex diagnoses, exposed reasoning chain |
| **API** | Server | FastAPI | Type-safe, async, auto-docs, Pydantic integration |
| **Memory** | Conversation | LangGraph InMemorySaver | Modern pattern, checkpointer-based, Phase 2 scales to PostgreSQL |

---

## File Map

```
/Users/eric/IBM/Projects/courses/Deliverables/Week-2/

Documentation:
├── IMPLEMENTATION_PLAN.md       ← Step-by-step code + specs (START HERE)
├── EXECUTION_SUMMARY.md         ← Status + roadmap
├── README.md                    ← Quick start + troubleshooting
└── PLAN_FINAL.md                ← This file

Configuration:
├── requirements.txt             ← pip install target
├── .env.example                 ← Environment variables template
├── docker-compose.yml           ← Milvus stack (one command)
└── .gitignore                   ← Git ignore rules

Source Code (Ready for Implementation):
└── src/
    ├── __init__.py
    ├── config.py                [TO IMPLEMENT] ~200 lines
    ├── models/
    │   ├── __init__.py
    │   └── schemas.py           [TO IMPLEMENT] ~100 lines
    ├── embeddings/
    │   ├── __init__.py
    │   └── embedding_service.py [TO IMPLEMENT] ~100 lines
    ├── vectorstore/
    │   ├── __init__.py
    │   └── milvus_client.py     [TO IMPLEMENT] ~150 lines
    ├── ingestion/
    │   ├── __init__.py
    │   ├── base_loader.py       [TO IMPLEMENT] ~80 lines
    │   ├── yaml_loader.py       [TO IMPLEMENT] ~150 lines
    │   ├── log_loader.py        [TO IMPLEMENT] ~150 lines
    │   ├── events_loader.py     [TO IMPLEMENT] ~120 lines
    │   ├── markdown_loader.py   [TO IMPLEMENT] ~100 lines
    │   └── pipeline.py          [TO IMPLEMENT] ~150 lines
    ├── retrieval/
    │   ├── __init__.py
    │   ├── query_analyzer.py    [TO IMPLEMENT] ~200 lines
    │   ├── reranker.py          [TO IMPLEMENT] ~80 lines
    │   └── retriever.py         [TO IMPLEMENT] ~150 lines
    ├── chains/
    │   ├── __init__.py
    │   ├── model_selector.py    [TO IMPLEMENT] ~120 lines
    │   └── rag_chain.py         [TO IMPLEMENT] ~150 lines
    ├── memory/
    │   └── __init__.py          [Future: Session memory]
    ├── tools/
    │   └── __init__.py          [Future: LangGraph tools]
    └── api/
        ├── __init__.py
        └── server.py            [TO IMPLEMENT] ~150 lines

Scripts:
├── scripts/ingest.py            [TO IMPLEMENT] ~50 lines (CLI ingestion)
└── scripts/chat.py              [TO IMPLEMENT] ~80 lines (Interactive diagnosis)

Test Suite (Ready for Implementation):
└── tests/
    ├── __init__.py
    ├── test_ingestion.py        [SKELETON]
    ├── test_embeddings.py       [SKELETON]
    ├── test_milvus.py           [SKELETON]
    ├── test_query_analyzer.py   [SKELETON]
    ├── test_reranker.py         [SKELETON]
    └── test_rag_chain.py        [SKELETON]

Sample Data (Ready):
└── data/sample/
    ├── manifests/               [TO CREATE: 2-3 YAML files]
    ├── logs/                    [TO CREATE: 2-3 log files]
    ├── events/                  [TO CREATE: events.json]
    └── docs/                    [TO CREATE: markdown guide]
```

---

## Implementation Path

### Phase 1: Core Infrastructure (13-20 hours)

**Order** (can parallelize after Step 4):

1. **Config & Setup** (1-2 hrs) 
   - src/config.py
   - Test: Settings load from .env

2. **Embeddings** (1-2 hrs)
   - src/embeddings/embedding_service.py
   - Test: bge-m3 loads, outputs 384-dim

3. **Vector Store** (1-1.5 hrs)
   - src/vectorstore/milvus_client.py
   - Test: Health check passes, collection created

4. **Data Models** (0.5 hrs)
   - src/models/schemas.py
   - Test: Pydantic validation

5. **Ingestion Loaders** (4-6 hrs, parallelizable)
   - src/ingestion/base_loader.py
   - src/ingestion/yaml_loader.py
   - src/ingestion/log_loader.py
   - src/ingestion/events_loader.py
   - src/ingestion/markdown_loader.py
   - src/ingestion/pipeline.py
   - Test: Sample data loads, 20+ chunks stored

6. **Retrieval Tier** (2-3 hrs, parallelizable)
   - src/retrieval/query_analyzer.py
   - src/retrieval/reranker.py
   - src/retrieval/retriever.py
   - Test: Metadata extraction, reranking works

7. **Chains & LLM** (2-3 hrs)
   - src/chains/model_selector.py
   - src/chains/rag_chain.py
   - Test: Model selection, structured output

8. **API & CLI** (1-2 hrs)
   - src/api/server.py
   - scripts/ingest.py
   - scripts/chat.py
   - Test: /diagnose endpoint, interactive chat

9. **Integration & Testing** (2-3 hrs)
   - End-to-end ingest → query → diagnose
   - Sample data test
   - CLI and API smoke tests

10. **Documentation** (1 hr)
    - Code docstrings
    - Type hints
    - Example queries

---

## What to Do Next

### Step 1: Review Documentation

1. Read **IMPLEMENTATION_PLAN.md** (5500 lines, detailed code skeletons)
2. Skim **README.md** (quick start, API examples)
3. Review **EXECUTION_SUMMARY.md** (overview, roadmap)

### Step 2: Set Up Environment

```bash
cd /Users/eric/IBM/Projects/courses/Deliverables/Week-2

# Python environment
python3.11 -m venv venv
source venv/bin/activate

# Dependencies (not needed yet, but verify structure)
# pip install -r requirements.txt

# Milvus stack (prepare)
docker compose config  # Verify syntax

# Ollama (check running on localhost:11434)
ollama list
```

### Step 3: Start Implementation

**First Module: src/config.py**

- Simplest, zero dependencies on other modules
- ~200 lines, straightforward Pydantic BaseSettings
- Complete code skeleton in IMPLEMENTATION_PLAN.md Step 3
- Verify with: `python -c "from src.config import get_settings; print(get_settings())"`

**Second Module: src/embeddings/embedding_service.py**

- Depends only on config
- ~100 lines, wraps HuggingFaceEmbeddings
- Test: `python -c "from src.embeddings.embedding_service import get_embeddings; e = get_embeddings(); print(len(e.embed_query('test')))"`
- Expected: `384`

**Third Module: src/vectorstore/milvus_client.py**

- Depends on embeddings + config
- Requires docker compose up -d
- ~150 lines
- Test: `python -c "from src.vectorstore.milvus_client import MilvusVectorStore; m = MilvusVectorStore(); print('OK' if m.health_check() else 'FAILED')"`
- Expected: `OK`

**Then:** Parallelizable loaders, retrieval, chains.

---

## Success Metrics (Phase 1)

### Infrastructure Checkpoints
- [ ] Python 3.11+ virtual environment created
- [ ] Dependencies install without errors
- [ ] Milvus stack runs (3 containers healthy)
- [ ] Ollama serving models (llama3.1 available)

### Module Checkpoints
- [ ] config.py loads settings from .env
- [ ] embedding_service.py loads bge-m3, outputs 384-dim vectors
- [ ] milvus_client.py connects to Milvus, creates collection
- [ ] All loaders ingest sample data correctly
- [ ] Query analyzer extracts metadata (namespace, pod, error_type)
- [ ] Reranker reorders documents by relevance
- [ ] Model selector activates deepseek for complex queries
- [ ] RAG chain returns FailureDiagnosis with all fields populated

### System Checkpoints
- [ ] End-to-end: ingest sample data → query → diagnose response
- [ ] FastAPI server runs, /health endpoint responds
- [ ] CLI ingest script works: `python scripts/ingest.py --path data/sample/`
- [ ] CLI chat script works interactively
- [ ] Type hints on all functions & methods
- [ ] Logging at key decision points

### Quality Checkpoints
- [ ] All unit tests pass
- [ ] Code follows PEP 8 style
- [ ] Docstrings on classes & public methods
- [ ] Error handling for network failures, file I/O, LLM timeouts
- [ ] No hardcoded values (all via .env)

---

## Key Decisions (Locked)

### 1. Embeddings: bge-m3 (384-dim) + Reranking

**Why not bge-large-en-v1.5 (1024-dim)?**
- bge-m3: 3x smaller storage, 3x faster search, multilingual
- Reranker (bge-reranker-v2-m3) recovers 5-10% NDCG recall loss
- Net: Faster, smaller, same accuracy, multilingual

### 2. Model Selection: Adaptive (llama3.1 + deepseek-r1:32b)

**Why not always deepseek?**
- Reasoning models are 4-8x slower (~3-5s vs ~500ms)
- 90% of queries are factual or pattern-matching (don't need reasoning)
- Use deepseek only for genuinely complex diagnoses (decomposition, ambiguity, multi-step logic)

### 3. Query Analyzer: Custom (Regex + Heuristics)

**Why not LLM-based?**
- Deterministic, fast (<10ms), no LLM cost
- K8s-specific patterns (namespace, pod, error_type)
- No need for learned patterns in Phase 1

### 4. Retrieval: Vector Search + Reranking + Filtering

**Why three-tier?**
- Vector search: Semantic relevance (fast)
- Metadata filtering: Precise scoping (namespace, pod)
- Reranking: Top-K precision (crossencoder re-scores)
- Optimal recall + latency tradeoff

### 5. Ingestion: 5 Custom Loaders (YAML, Logs, Events, Markdown, Helm)

**Why not generic TextLoader?**
- K8s-specific structure extraction (kind, namespace, reason)
- Domain-aware chunking (by resource, time window, header)
- Rich metadata enables intelligent filtering

### 6. Memory: LangGraph InMemorySaver (dev) → PostgresSaver (prod)

**Why not traditional memory classes?**
- Modern LangChain v1 pattern
- Checkpointer-based (multi-session, stateless)
- Zero refactor needed for Phase 2 agent integration

### 7. LangChain LCEL (not full LangGraph yet)

**Why not LangGraph agents from the start?**
- Simpler Phase 1 scope (focus on RAG, not orchestration)
- Clear migration path (LCEL chains → agent nodes)
- Proven patterns, battle-tested

---

## Known Limitations & Future Work

### Phase 1 Limitations
- No multi-turn personality (each query is independent)
- Query analyzer uses regex (not ML-based)
- No context window management (long query handling)
- Single user (no multi-tenant isolation)
- In-memory model (no model caching across requests)

### Phase 2 Enhancements
- LangGraph workflows for multi-step diagnosis
- Tool execution (kubectl, shell commands)
- Stateful conversation (chat history in PostgreSQL)
- Streaming output for long diagnoses
- LangSmith tracing & observability
- Prometheus metrics & dashboards
- Evaluation framework (RAGAS scoring)

### Phase 3+ Scaling
- Cluster deployment (Kubernetes native)
- Multi-GPU inference (vLLM, TGI)
- Federated learning for domain adaptation
- Real-time metric queries (Prometheus API)
- Incident database integration

---

## Getting Help

### Documentation
1. **IMPLEMENTATION_PLAN.md** — Detailed specs + code skeletons
2. **README.md** — Quick start, API examples, troubleshooting
3. **EXECUTION_SUMMARY.md** — Roadmap, decisions, effort estimates

### Debugging
- Enable `LOG_LEVEL=DEBUG` in .env for verbose logging
- Check Milvus Web UI: http://localhost:9091/webui
- Inspect Milvus logs: `docker compose logs -f milvus-standalone`
- Test individual modules in Python REPL

### Common Issues
- **Embeddings not loading**: Check `EMBEDDING_DEVICE=cpu` or `mps` (Apple Silicon)
- **Milvus unhealthy**: Run `docker compose down -v && docker compose up -d`
- **No models in Ollama**: Run `ollama pull llama3.1`
- **Query returns no results**: Verify ingestion completed, check Milvus collection size

---

## Summary

You now have:

✅ **Complete technical specification** (5500+ lines, step-by-step code)  
✅ **Architecture documented** (components, data flow, decisions)  
✅ **Infrastructure as code** (docker-compose, requirements, .env)  
✅ **Project scaffolding** (11 packages, proper structure)  
✅ **API & CLI templates** (FastAPI, argparse)  
✅ **Test framework** (pytest skeleton)  
✅ **Documentation** (README, PLAN, SUMMARY)  

**Ready for implementation.**

Estimated effort: **13-20 hours** for full Phase 1 (2500 LOC)

---

**Next Step: Start with [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) Step 1-3 (config + embeddings + vectorstore).**

---

*Kubernetes Failure Intelligence Copilot — Phase 1 Planning Complete*  
*February 18, 2026*
