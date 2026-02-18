"""Tests for ingestion loaders — YAML, Log, Events, Markdown, Helm, Pipeline."""

import json
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.ingestion.yaml_loader import KubernetesYAMLLoader
from src.ingestion.log_loader import KubernetesLogLoader
from src.ingestion.events_loader import KubernetesEventsLoader
from src.ingestion.markdown_loader import MarkdownDocumentLoader
from src.ingestion.helm_loader import HelmChartLoader
from src.ingestion.pipeline import IngestionPipeline
from src.config import get_settings


# ---------------------------------------------------------------------------
# YAML Loader
# ---------------------------------------------------------------------------

class TestYAMLLoader:
    def test_load_single_file(self, tmp_path: Path):
        f = tmp_path / "deploy.yaml"
        f.write_text(
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: test\n  namespace: default\n"
        )
        loader = KubernetesYAMLLoader()
        docs = loader.load(f)
        assert len(docs) == 1
        assert docs[0].metadata["kind"] == "Deployment"
        assert docs[0].metadata["name"] == "test"

    def test_load_multidoc(self, tmp_path: Path):
        f = tmp_path / "multi.yaml"
        f.write_text(
            "apiVersion: v1\nkind: Service\nmetadata:\n  name: svc1\n---\n"
            "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cm1\n"
        )
        loader = KubernetesYAMLLoader()
        docs = loader.load(f)
        assert len(docs) == 2

    def test_load_directory(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("kind: Pod\nmetadata:\n  name: p1\n")
        (tmp_path / "b.yml").write_text("kind: Pod\nmetadata:\n  name: p2\n")
        loader = KubernetesYAMLLoader()
        docs = loader.load(tmp_path)
        assert len(docs) == 2

    def test_chunk(self, tmp_path: Path):
        long_content = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: big\ndata:\n  key: " + "x" * 3000
        f = tmp_path / "big.yaml"
        f.write_text(long_content)
        loader = KubernetesYAMLLoader(chunk_size=500, chunk_overlap=50)
        docs = loader.load(f)
        chunks = loader.chunk(docs)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Log Loader
# ---------------------------------------------------------------------------

class TestLogLoader:
    def test_load_single_file(self, tmp_path: Path):
        f = tmp_path / "app.log"
        f.write_text(
            "2025-01-01T10:00:00 [ERROR] something failed\n"
            "2025-01-01T10:00:01 [ERROR] details here\n"
        )
        loader = KubernetesLogLoader()
        docs = loader.load(f)
        assert len(docs) >= 1
        assert docs[0].metadata["doc_type"] == "kubernetes_log"

    def test_time_window_grouping(self, tmp_path: Path):
        f = tmp_path / "app.log"
        f.write_text(
            "2025-01-01T10:00:00 line1\n"
            "2025-01-01T10:00:05 line2\n"
            "2025-01-01T10:05:00 line3\n"  # 5 minutes later → new group
        )
        loader = KubernetesLogLoader(time_window=30)
        docs = loader.load(f)
        assert len(docs) == 2

    def test_load_directory(self, tmp_path: Path):
        (tmp_path / "a.log").write_text("2025-01-01T10:00:00 log line\n")
        (tmp_path / "b.txt").write_text("2025-01-01T11:00:00 another line\n")
        loader = KubernetesLogLoader()
        docs = loader.load(tmp_path)
        assert len(docs) == 2


# ---------------------------------------------------------------------------
# Events Loader
# ---------------------------------------------------------------------------

class TestEventsLoader:
    def test_load_json(self, tmp_path: Path):
        events = [
            {
                "reason": "CrashLoopBackOff",
                "type": "Warning",
                "message": "Back-off restarting",
                "involvedObject": {"kind": "Pod", "name": "web-1", "namespace": "prod"},
                "count": 5,
            }
        ]
        f = tmp_path / "test.events.json"
        f.write_text(json.dumps(events))
        loader = KubernetesEventsLoader()
        docs = loader.load(f)
        assert len(docs) == 1
        assert docs[0].metadata["reason"] == "CrashLoopBackOff"
        assert docs[0].metadata["doc_type"] == "kubernetes_event"

    def test_load_directory(self, tmp_path: Path):
        events = [{"reason": "OOMKilled", "type": "Warning", "message": "killed",
                    "involvedObject": {"kind": "Pod", "name": "p1", "namespace": "ns"}}]
        f = tmp_path / "oom.events.json"
        f.write_text(json.dumps(events))
        loader = KubernetesEventsLoader()
        docs = loader.load(tmp_path)
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# Markdown Loader
# ---------------------------------------------------------------------------

class TestMarkdownLoader:
    def test_load_single(self, tmp_path: Path):
        f = tmp_path / "runbook.md"
        f.write_text("# Title\n\nSome content\n\n## Section\n\nMore content\n")
        loader = MarkdownDocumentLoader()
        docs = loader.load(f)
        assert len(docs) == 1
        assert docs[0].metadata["doc_type"] == "markdown_document"

    def test_chunk_by_headers(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# H1\n\nParagraph 1\n\n## H2\n\nParagraph 2\n\n### H3\n\nParagraph 3\n")
        loader = MarkdownDocumentLoader()
        docs = loader.load(f)
        chunks = loader.chunk(docs)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Helm Loader
# ---------------------------------------------------------------------------

class TestHelmLoader:
    def test_load_template(self, tmp_path: Path):
        tpl = tmp_path / "deploy.yaml"
        tpl.write_text(
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
            "  name: {{ .Release.Name }}\n  namespace: default\nspec:\n  replicas: 1\n"
        )
        loader = HelmChartLoader()
        docs = loader.load(tpl)
        assert len(docs) >= 1

    def test_load_chart_dir(self, tmp_path: Path):
        # Create minimal chart structure
        chart_yaml = tmp_path / "Chart.yaml"
        chart_yaml.write_text("name: my-chart\nversion: 0.1.0\n")

        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "deploy.yaml").write_text(
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {{ .Values.name }}\n"
        )

        values = tmp_path / "values.yaml"
        values.write_text("name: my-app\nreplicas: 2\n")

        loader = HelmChartLoader()
        docs = loader.load(tmp_path)
        # Should have template doc + values doc
        assert len(docs) >= 2


# ---------------------------------------------------------------------------
# Pipeline type detection
# ---------------------------------------------------------------------------

class TestPipelineDetectType:
    """Test _detect_type without connecting to Milvus."""

    @staticmethod
    def _make_pipeline() -> IngestionPipeline:
        """Create a pipeline stub with real settings but no vectorstore init."""
        p = IngestionPipeline.__new__(IngestionPipeline)
        p.settings = get_settings()
        return p

    def test_yaml_file(self, tmp_path: Path):
        f = tmp_path / "test.yaml"
        f.touch()
        assert self._make_pipeline()._detect_type(f) == "yaml"

    def test_log_file(self, tmp_path: Path):
        f = tmp_path / "app.log"
        f.touch()
        assert self._make_pipeline()._detect_type(f) == "log"

    def test_event_file(self, tmp_path: Path):
        f = tmp_path / "crash.events.json"
        f.touch()
        assert self._make_pipeline()._detect_type(f) == "event"

    def test_markdown_file(self, tmp_path: Path):
        f = tmp_path / "runbook.md"
        f.touch()
        assert self._make_pipeline()._detect_type(f) == "markdown"

    def test_helm_dir(self, tmp_path: Path):
        (tmp_path / "Chart.yaml").touch()
        assert self._make_pipeline()._detect_type(tmp_path) == "helm"

    def test_events_dir(self, tmp_path: Path):
        d = tmp_path / "events"
        d.mkdir()
        assert self._make_pipeline()._detect_type(d) == "event"
