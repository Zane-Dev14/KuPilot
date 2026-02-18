"""Tests for src.retrieval.query_analyzer — metadata extraction + decomposition."""

from src.retrieval.query_analyzer import QueryAnalyzer
from src.models.schemas import QueryMetadata


class TestExtractK8sMetadata:
    def setup_method(self):
        self.analyzer = QueryAnalyzer()

    def test_extract_namespace(self):
        meta = self.analyzer.extract_k8s_metadata("pod crashing in namespace prod")
        assert meta.namespace == "prod"

    def test_extract_pod(self):
        meta = self.analyzer.extract_k8s_metadata("pod web-app-123 keeps restarting")
        assert meta.pod == "web-app-123"

    def test_extract_error_type_crash(self):
        meta = self.analyzer.extract_k8s_metadata("my container keeps crashing")
        assert meta.error_type == "crashed"

    def test_extract_error_type_oom(self):
        meta = self.analyzer.extract_k8s_metadata("container was oom killed")
        assert meta.error_type == "oom_killed"

    def test_extract_error_type_image_pull(self):
        meta = self.analyzer.extract_k8s_metadata("image pull error on deploy")
        assert meta.error_type == "image_pull_error"

    def test_extract_node(self):
        meta = self.analyzer.extract_k8s_metadata("node worker-1.internal is not ready")
        assert meta.node == "worker-1.internal"

    def test_extract_multiple(self):
        meta = self.analyzer.extract_k8s_metadata(
            "pod web-123 crashed in namespace staging on node node-2"
        )
        assert meta.namespace == "staging"
        assert meta.pod == "web-123"
        assert meta.error_type == "crashed"

    def test_no_metadata(self):
        meta = self.analyzer.extract_k8s_metadata("hello world")
        assert meta.namespace is None
        assert meta.pod is None
        assert meta.error_type is None

    def test_returns_pydantic_model(self):
        meta = self.analyzer.extract_k8s_metadata("test")
        assert isinstance(meta, QueryMetadata)


class TestDecomposeQuery:
    def setup_method(self):
        self.analyzer = QueryAnalyzer()

    def test_single_query(self):
        parts = self.analyzer.decompose_query("Why is my pod crashing?")
        assert parts == ["Why is my pod crashing?"]

    def test_semicolon_separator(self):
        parts = self.analyzer.decompose_query("check pod logs; also check events")
        assert len(parts) == 2
        assert parts[0] == "check pod logs"
        assert parts[1] == "also check events"

    def test_and_also_separator(self):
        parts = self.analyzer.decompose_query("fix the crash and also check memory")
        assert len(parts) == 2


class TestAnalyze:
    def setup_method(self):
        self.analyzer = QueryAnalyzer()

    def test_returns_tuple(self):
        result = self.analyzer.analyze("pod web-123 crashed in namespace prod")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_metadata_and_subqueries(self):
        meta, subs = self.analyzer.analyze("pod web-123 crashed in namespace prod")
        assert meta.namespace == "prod"
        assert meta.pod == "web-123"
        assert len(subs) >= 1
