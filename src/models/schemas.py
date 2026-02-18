from pydantic import BaseModel, Field
from typing import Optional

class QueryMetadata(BaseModel):
    """Extracted K8s metadata from a query."""
    namespace: Optional[str] = None
    pod: Optional[str] = None
    container: Optional[str] = None
    node: Optional[str] = None
    error_type: Optional[str] = None
    labels_dict: dict[str, str] = Field(default_factory=dict)

class FailureDiagnosis(BaseModel):
    """Structured diagnosis output."""
    root_cause: str = Field(description="Root cause of the failure")
    explanation: str = Field(description="Detailed explanation of what went wrong")
    recommended_fix: str = Field(description="Recommended fix or mitigation steps")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    sources: list[str] = Field(default_factory=list, description="Source documents")
    reasoning_model_used: Optional[str] = None
    thinking_chain: Optional[str] = None

class DiagnoseRequest(BaseModel):
    """Request to diagnose a K8s failure."""
    question: str = Field(description="Question or failure description")
    namespace: Optional[str] = None
    force_model: Optional[str] = None
    session_id: Optional[str] = None

class DiagnoseResponse(BaseModel):
    """Response with diagnosis."""
    diagnosis: FailureDiagnosis
    session_id: Optional[str] = None

class QueryAnalysisResponse(BaseModel):
    """Debug response showing query analysis."""
    metadata: QueryMetadata
    sub_queries: list[str]

class IngestRequest(BaseModel):
    """Request to ingest documents."""
    path: str = Field(description="File or directory path")
    doc_type: Optional[str] = None

class IngestResponse(BaseModel):
    """Response with ingestion stats."""
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    errors: list[str] = Field(default_factory=list)