# K8s Failure Intelligence Copilot

RAG-powered Kubernetes failure diagnosis. Ask questions about pod crashes, OOMKills, scheduling failures — get root-cause analysis backed by your own runbooks, events, and manifests.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue) ![Gemini](https://img.shields.io/badge/LLM-Gemini-blueviolet) ![Chroma](https://img.shields.io/badge/VectorDB-Chroma-orange)

---

## How It Works

```
User question
     ↓
Query Classifier (LLM + heuristics)
     ├── diagnostic     → RAG pipeline (retrieve → LLM → structured JSON)
     ├── conversational  → memory lookup ("what did I ask before?")
     ├── operational     → kubectl command suggestions
     └── out_of_scope   → polite refusal

RAG pipeline:
  1. Chroma vector search (BGE embeddings, 384-dim)
  2. Gemini 2.0 Flash generation
  3. Structured JSON diagnosis with confidence + sources
```

---

## Quick Start (Local)

```bash
# 1. Clone & enter project
cd Week-2

# 2. Run setup (installs deps, runs tests, downloads embeddings, ingests sample data)
chmod +x setup.sh
./setup.sh

# 3. Set your Google API key
export GOOGLE_API_KEY=your-key-here   # or add to .env

# 4a. Web UI
python -m uvicorn src.api:app --reload
# → http://localhost:8000

# 4b. CLI chat
python scripts/chat.py
```

---

## Docker

Single container — everything (FastAPI + Chroma + embeddings) runs inside one image.

```bash
# Build & run
docker compose up --build

# Or standalone
docker build -t k8s-copilot .
docker run -p 8000:8000 -v copilot_data:/data -e GOOGLE_API_KEY=your-key k8s-copilot
```

The embedded model is pre-downloaded during `docker build`, so startup is instant.
Chroma data and chat memory persist in the `/data` volume.

---

## Deploy (Render / Fly.io)

### Render (free tier)

1. Push repo to GitHub
2. Create a **Web Service** on [render.com](https://render.com), connect the repo
3. Set **Environment** → Docker
4. Add env var: `GOOGLE_API_KEY`
5. Add a **Disk** mounted at `/data` (1 GB free)
6. Deploy — Render builds the Dockerfile automatically

### Fly.io

```bash
fly launch --no-deploy
fly secrets set GOOGLE_API_KEY=your-key
fly volumes create copilot_data --size 1 --region ord
# Edit fly.toml: [mounts] source="copilot_data" destination="/data"
fly deploy
```

---

## Project Structure

```
src/
  config.py          Settings (Pydantic BaseSettings, reads .env)
  vectorstore.py     Embeddings + Chroma wrapper
  ingestion.py       File loaders (YAML, JSON events, Markdown, logs) + chunking
  memory.py          Per-session chat memory (LRU eviction, disk persistence)
  rag_chain.py       Core RAG: classify → retrieve → generate → parse
  api.py             FastAPI server (web UI + REST + SSE streaming + CORS)

scripts/
  chat.py            Interactive CLI chat (rich output)
  ingest.py          CLI document ingestion

static/              Frontend assets (Three.js cinematic UI)
templates/           Jinja2 HTML template
data/sample/         Sample K8s manifests, events, runbooks
tests/               Offline unit tests (no external services needed)
```

---

## API

| Method | Endpoint | What it does |
|--------|----------|-------------|
| `GET`  | `/` | Web UI |
| `GET`  | `/health` | Chroma connection status |
| `POST` | `/diagnose` | Diagnose a failure → structured JSON |
| `POST` | `/diagnose/stream` | Same, but SSE token streaming |
| `POST` | `/ingest` | Ingest documents into Chroma |
| `POST` | `/memory/clear` | Clear a session's memory |

**Example — diagnose:**

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is my pod OOMKilled?", "session_id": "user1"}'
```

---

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | What |
|----------|---------|------|
| `GOOGLE_API_KEY` | *(required)* | Gemini API key |
| `MODEL_NAME` | `gemini-2.0-flash` | Gemini model name |
| `CHROMA_PERSIST_DIR` | `/data/chroma` | Chroma storage path |
| `CHROMA_COLLECTION` | `k8s_failures` | Chroma collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim) |
| `EMBEDDING_DEVICE` | `cpu` | `mps` for Apple Silicon, `cpu` for Docker |
| `RETRIEVAL_TOP_K` | `4` | Documents returned per query |
| `CHUNK_SIZE` | `1000` | Text chunk size for ingestion |
| `MEMORY_PATH` | `/data/chat_memory.json` | Chat history file |

---

## Requirements

- **Python 3.10+**
- **Google Gemini API key** (free tier or paid)
- ~512 MB RAM for embeddings + Chroma

---

## Tests

```bash
# Offline tests — no Gemini or Chroma server needed
pytest tests/test_basic.py -v
```

Covers: config defaults, memory (LRU, eviction, persistence), JSON parser, ingestion (all file types), query classifier, conversational handler.

---

## Adding Your Own Data

Drop files into `data/sample/` (or any directory):

- `.yaml` / `.yml` — K8s manifests
- `.json` — K8s events
- `.md` — Runbooks / documentation
- `.log` / `.txt` — Plain text logs

Then ingest:

```bash
python scripts/ingest.py --path data/sample/
```

Use `--no-drop` to append instead of replacing the collection.
