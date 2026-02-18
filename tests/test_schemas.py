"""Tests for src.models.schemas — Pydantic model validation."""

from src.models.schemas import (
    QueryMetadata,
    FailureDiagnosis,
    DiagnoseRequest,
    DiagnoseResponse,
    QueryAnalysisResponse,
    IngestRequest,
    IngestResponse,
)


class TestQueryMetadata:
    def test_defaults(self):
        m = QueryMetadata()
        assert m.namespace is None
        assert m.pod is None
        assert m.labels_dict == {}

    def test_with_values(self):
        m = QueryMetadata(namespace="prod", pod="web-123", error_type="crashed")
        assert m.namespace == "prod"
        assert m.pod == "web-123"


class TestFailureDiagnosis:
    def test_required_fields(self):
        d = FailureDiagnosis(
            root_cause="DB down",
            explanation="Connection refused",
            recommended_fix="Restart DB",
            confidence=0.9,
        )
        assert d.root_cause == "DB down"
        assert d.confidence == 0.9
        assert d.sources == []
        assert d.reasoning_model_used is None

    def test_all_fields(self):
        d = FailureDiagnosis(
            root_cause="OOM",
            explanation="Memory exceeded",
            recommended_fix="Increase limits",
            confidence=0.85,
            sources=["manifest.yaml"],
            reasoning_model_used="deepseek-r1:32b",
            thinking_chain="[reasoning]",
        )
        assert d.reasoning_model_used == "deepseek-r1:32b"
        assert d.thinking_chain == "[reasoning]"


class TestDiagnoseRequest:
    def test_minimal(self):
        r = DiagnoseRequest(question="Why is my pod crashing?")
        assert r.question == "Why is my pod crashing?"
        assert r.force_model is None

    def test_with_force_model(self):
        r = DiagnoseRequest(question="test", force_model="deepseek-r1:32b")
        assert r.force_model == "deepseek-r1:32b"


class TestIngestResponse:
    def test_with_errors(self):
        r = IngestResponse(documents_loaded=5, chunks_created=10, chunks_stored=10, errors=["bad file"])
        assert r.errors == ["bad file"]

    def test_defaults(self):
        r = IngestResponse(documents_loaded=0, chunks_created=0, chunks_stored=0)
        assert r.errors == []
