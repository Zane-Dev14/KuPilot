import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path
from src.config import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.chains.rag_chain import RAGChain
from src.retrieval.query_analyzer import QueryAnalyzer
from src.models.schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
    IngestRequest,
    IngestResponse,
    QueryAnalysisResponse,
)

logger = logging.getLogger(__name__)

# Lazy-initialized service references
rag_chain: RAGChain | None = None
query_analyzer: QueryAnalyzer | None = None
ingestion_pipeline: IngestionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy services on startup."""
    global rag_chain, query_analyzer, ingestion_pipeline
    logger.info("Initializing services...")
    query_analyzer = QueryAnalyzer()
    ingestion_pipeline = IngestionPipeline()
    rag_chain = RAGChain()
    logger.info("Services initialized")
    yield
    logger.info("Shutting down services")


# Initialize FastAPI app
app = FastAPI(
    title="Kubernetes Failure Intelligence Copilot",
    version="0.1.0",
    description="RAG-powered diagnosis of Kubernetes failures",
    lifespan=lifespan,
)

settings = get_settings()

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """
    Diagnose a Kubernetes failure.
    
    - **question**: Description of the failure or question
    - **namespace**: Optional namespace filter
    - **force_model**: Override model selection
    """
    try:
        if rag_chain is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        diagnosis = rag_chain.diagnose(
            query=request.question,
            force_model=request.force_model,
        )
        
        return DiagnoseResponse(
            diagnosis=diagnosis,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-analysis", response_model=QueryAnalysisResponse)
async def analyze_query(request: DiagnoseRequest):
    """
    Analyze a query (debug endpoint).
    
    Returns extracted metadata and decomposed sub-queries.
    """
    try:
        if query_analyzer is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        metadata, sub_queries = query_analyzer.analyze(request.question)
        
        return QueryAnalysisResponse(
            metadata=metadata,
            sub_queries=sub_queries,
        )
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest documents into the knowledge base.
    
    - **path**: File or directory path
    - **doc_type**: Override type detection (yaml, log, event, markdown)
    """
    try:
        if ingestion_pipeline is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        result = ingestion_pipeline.ingest(
            path=Path(request.path),
            doc_type=request.doc_type,
        )
        
        return IngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)