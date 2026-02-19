"""FastAPI server for K8s Failure Intelligence Copilot.

Endpoints:
  POST /diagnose           → Diagnose a Kubernetes failure
  POST /query-analysis     → Analyze a query (debug endpoint)
  POST /ingest             → Ingest documents into knowledge base
  GET  /health             → Health check
"""

import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional

from src.config import get_settings
from src.rag_chain import RAGChain, FailureDiagnosis, estimate_complexity
from src.ingestion import ingest_file, ingest_directory
from src.vectorstore import MilvusStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Enable debug logging for rag chain during troubleshooting
logging.getLogger("src.rag_chain").setLevel(logging.DEBUG)
logging.getLogger("src.vectorstore").setLevel(logging.DEBUG)

# ── Pydantic schemas ─────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    """Request to diagnose a K8s failure."""
    question: str = Field(..., min_length=1, description="Question or failure description")
    namespace: Optional[str] = Field(None, description="Optional namespace filter")
    force_model: Optional[str] = Field(None, description="Override model selection")
    session_id: Optional[str] = Field(default="default", description="Session ID for memory")


class DiagnoseResponse(BaseModel):
    """Response with diagnosis."""
    diagnosis: FailureDiagnosis
    session_id: Optional[str] = None
    complexity_score: float = Field(..., description="Query complexity 0.0-1.0")


class QueryAnalysisRequest(BaseModel):
    """Request to analyze a query."""
    question: str = Field(..., min_length=1, description="Question or failure description")


class QueryAnalysis(BaseModel):
    """Query analysis result (debug)."""
    question: str
    complexity_score: float
    estimated_model: str


class QueryAnalysisResponse(BaseModel):
    """Response with query analysis."""
    analysis: QueryAnalysis


class IngestRequest(BaseModel):
    """Request to ingest documents."""
    path: str = Field(..., description="File or directory path")
    doc_type: Optional[str] = Field(None, description="Override type detection")
    no_drop: bool = Field(False, description="Don't wipe collection, append instead")


class IngestResponse(BaseModel):
    """Response with ingestion stats."""
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    errors: list = Field(default_factory=list)


class MemoryClearRequest(BaseModel):
    """Request to clear a session's memory."""
    session_id: str = Field(..., description="Session ID to clear")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Kubernetes Failure Intelligence Copilot",
    version="0.1.0",
    description="RAG-powered diagnosis of Kubernetes failures",
)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize services (lazy; will connect on first request)
_rag_chain: Optional[RAGChain] = None
_milvus: Optional[MilvusStore] = None


def get_rag_chain() -> RAGChain:
    """Get cached RAG chain."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def get_milvus_store(drop_old: bool = False) -> MilvusStore:
    """Get cached Milvus store."""
    global _milvus
    if _milvus is None:
        _milvus = MilvusStore(drop_old=drop_old)
    return _milvus


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Web UI for interactive diagnosis."""
    s = get_settings()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "simple_model": s.simple_model,
            "complex_model": s.complex_model,
        },
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        store = get_milvus_store()
        is_healthy = await run_in_threadpool(store.health_check)
        payload = {
            "status": "ok" if is_healthy else "degraded",
            "milvus": "connected" if is_healthy else "disconnected",
        }
        if not is_healthy:
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
        return payload
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "detail": str(e)},
        )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """
    Diagnose a Kubernetes failure.

    Request:
    ```json
    {
      "question": "Why is my pod crashing with OOMKilled?",
      "namespace": "prod",
      "force_model": null,
      "session_id": "user_123"
    }
    ```

    Response:
    ```json
    {
      "diagnosis": {
        "root_cause": "Container exceeded memory limit",
        "explanation": "The app allocated more memory than the 512Mi limit",
        "recommended_fix": "1. Increase limit to 1Gi\n2. Profile app memory usage",
        "confidence": 0.85,
        "sources": ["data/sample/docs/oomkilled-runbook.md"],
        "model_used": "llama3.1"
      },
      "session_id": "user_123",
      "complexity_score": 0.35
    }
    ```
    """
    try:
        chain = get_rag_chain()

        # Estimate complexity
        complexity = estimate_complexity(request.question)

        # Diagnose
        force_model = request.force_model
        if force_model in (None, "", "string"):
            force_model = None

        diagnosis = await run_in_threadpool(
            chain.diagnose,
            request.question,
            request.session_id or "default",
            force_model,
        )

        return DiagnoseResponse(
            diagnosis=diagnosis,
            session_id=request.session_id,
            complexity_score=complexity,
        )
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnose/stream")
async def diagnose_stream(request: DiagnoseRequest):
    """Stream diagnosis via SSE. Tokens arrive as `data: {"token": "..."}` events,
    final structured result as `data: {"done": true, "diagnosis": {...}}`."""
    try:
        chain = get_rag_chain()
        force_model = request.force_model
        if force_model in (None, "", "string"):
            force_model = None

        async def event_generator():
            try:
                async for event in chain.diagnose_stream(
                    request.question,
                    request.session_id or "default",
                    force_model,
                ):
                    yield event
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                import json as _json
                yield f"data: {_json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        logger.error(f"Stream setup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-analysis", response_model=QueryAnalysisResponse)
async def analyze_query(request: QueryAnalysisRequest):
    """
    Analyze a query to understand routing and complexity (debug endpoint).

    Request:
    ```json
    {
      "question": "Why is my pod crashing and how do I fix it?"
    }
    ```

    Response:
    ```json
    {
      "analysis": {
        "question": "Why is my pod crashing and how do I fix it?",
        "complexity_score": 0.65,
        "estimated_model": "llama3.1"
      }
    }
    ```
    """
    try:
        complexity = estimate_complexity(request.question)
        settings = get_settings()
        model = (
            settings.complex_model
            if complexity >= settings.query_complexity_threshold
            else settings.simple_model
        )

        return QueryAnalysisResponse(
            analysis=QueryAnalysis(
                question=request.question,
                complexity_score=complexity,
                estimated_model=model,
            )
        )
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest documents into the knowledge base.

    Request:
    ```json
    {
      "path": "data/sample/",
      "doc_type": null,
      "no_drop": false
    }
    ```

    Response:
    ```json
    {
      "documents_loaded": 12,
      "chunks_created": 12,
      "chunks_stored": 12,
      "errors": []
    }
    ```
    """
    try:
        target = Path(request.path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")

        logger.info(f"Ingesting: {target}")

        # Get Milvus store (drop_old = not no_drop)
        drop_old = not request.no_drop
        store = MilvusStore(drop_old=drop_old)

        # Load documents
        if target.is_dir():
            docs = await run_in_threadpool(ingest_directory, target)
        else:
            docs = await run_in_threadpool(ingest_file, target)

        if not docs:
            logger.warning(f"No documents found in {target}")
            return IngestResponse(
                documents_loaded=0,
                chunks_created=0,
                chunks_stored=0,
                errors=["No documents found"],
            )

        # Store in Milvus
        ids = await run_in_threadpool(store.add_documents, docs)

        result = IngestResponse(
            documents_loaded=len(docs),
            chunks_created=len(docs),
            chunks_stored=len(ids),
            errors=[],
        )

        logger.info(f"Ingestion complete: {result}")
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear")
async def clear_memory(request: MemoryClearRequest):
    """Clear chat memory for a session ID."""
    try:
        chain = get_rag_chain()
        chain.memory.clear(request.session_id)
        return {"status": "ok", "session_id": request.session_id}
    except Exception as e:
        logger.error(f"Memory clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Startup / Shutdown ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Kubernetes Failure Intelligence Copilot API starting")
    logger.info(f"Settings: {get_settings()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down")


# ── CLI / Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
