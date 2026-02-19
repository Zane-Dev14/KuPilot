"""Tests that run without Milvus, Ollama, or downloaded models.

Covers: config defaults, memory, model selector, ingestion parsing.
Run:  pytest tests/test_basic.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest


# ─── Config ──────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        from src.config import Settings
        s = Settings()  # no .env needed — uses defaults
        assert s.milvus_uri == "http://localhost:19530"
        assert s.milvus_collection == "k8s_failures"
        assert s.embedding_dimension == 384
        assert s.retrieval_top_k == 4
        assert 0.0 < s.query_complexity_threshold < 1.0

    def test_singleton(self):
        from src.config import get_settings
        a = get_settings()
        b = get_settings()
        assert a is b


# ─── Memory ──────────────────────────────────────────────────────────────────

class TestMemory:
    def _new_mem(self, **kw):
        from src.memory import ChatMemory
        return ChatMemory(**kw)

    def test_add_and_retrieve(self):
        mem = self._new_mem()
        mem.add_user_message("s1", "hello")
        mem.add_ai_message("s1", "hi")
        history = mem.get_history("s1")
        assert len(history) == 2
        assert history[0].content == "hello"
        assert history[1].content == "hi"

    def test_empty_session(self):
        mem = self._new_mem()
        assert mem.get_history("nonexistent") == []

    def test_trim(self):
        mem = self._new_mem(max_messages=4)
        for i in range(6):
            mem.add_user_message("s1", f"msg-{i}")
        assert len(mem.get_history("s1")) == 4

    def test_lru_eviction(self):
        mem = self._new_mem(max_sessions=2)
        mem.add_user_message("a", "x")
        mem.add_user_message("b", "x")
        mem.add_user_message("c", "x")  # should evict "a"
        assert mem.get_history("a") == []
        assert mem.active_sessions == 2

    def test_clear(self):
        mem = self._new_mem()
        mem.add_user_message("s1", "x")
        mem.clear("s1")
        assert mem.get_history("s1") == []
        assert mem.active_sessions == 0

    def test_clear_all(self):
        mem = self._new_mem()
        mem.add_user_message("a", "x")
        mem.add_user_message("b", "x")
        mem.clear_all()
        assert mem.active_sessions == 0


# ─── Model selector ─────────────────────────────────────────────────────────

class TestModelSelector:
    def test_simple_query_low_complexity(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity("list pods")
        assert score < 0.3

    def test_reasoning_query(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity("Why is my pod in CrashLoopBackOff?")
        assert score >= 0.3

    def test_complex_multi_question(self):
        from src.rag_chain import estimate_complexity
        score = estimate_complexity("Why is it crashing? How do I fix it? What caused the OOM?")
        assert score >= 0.5

    def test_force_model_override(self):
        from src.rag_chain import select_model
        result = select_model("hi", force="my-custom-model")
        assert result == "my-custom-model"


# ─── JSON parser ─────────────────────────────────────────────────────────────

class TestParseJson:
    def test_direct_json(self):
        from src.rag_chain import _parse_json
        raw = '{"root_cause": "OOM", "confidence": 0.9}'
        assert _parse_json(raw)["root_cause"] == "OOM"

    def test_fenced_code_block(self):
        from src.rag_chain import _parse_json
        raw = 'Some preamble\n```json\n{"root_cause": "X"}\n```\nAftermath'
        assert _parse_json(raw)["root_cause"] == "X"

    def test_embedded_braces(self):
        from src.rag_chain import _parse_json
        raw = 'Here is the answer: {"root_cause": "Y"}'
        assert _parse_json(raw)["root_cause"] == "Y"

    def test_garbage_returns_empty(self):
        from src.rag_chain import _parse_json
        assert _parse_json("no json here!!!") == {}


# ─── Ingestion ───────────────────────────────────────────────────────────────

class TestIngestion:
    def test_load_yaml(self):
        from src.ingestion import ingest_file

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test-app", "namespace": "dev"},
            "spec": {"replicas": 2},
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            import yaml
            yaml.dump(manifest, f)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        assert docs[0].metadata["kind"] == "Deployment"
        assert "test-app" in docs[0].page_content
        p.unlink()

    def test_load_json_events(self):
        from src.ingestion import ingest_file

        events = [
            {
                "reason": "OOMKilled",
                "type": "Warning",
                "message": "Container killed",
                "involvedObject": {"kind": "Pod", "name": "web", "namespace": "dev"},
                "count": 3,
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(events, f)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        assert "OOMKilled" in docs[0].page_content
        p.unlink()

    def test_load_markdown(self):
        from src.ingestion import ingest_file

        md = "# Title\nParagraph.\n## Section\nContent here."
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(md)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        p.unlink()

    def test_load_log(self):
        from src.ingestion import ingest_file

        log = "2024-01-01T00:00:00Z ERROR something went wrong\n" * 10
        with tempfile.NamedTemporaryFile(suffix=".log", mode="w", delete=False) as f:
            f.write(log)
            p = Path(f.name)
        docs = ingest_file(p)
        assert len(docs) >= 1
        assert "ERROR" in docs[0].page_content
        p.unlink()

    def test_unsupported_extension(self):
        from src.ingestion import ingest_file
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            p = Path(f.name)
        docs = ingest_file(p)
        assert docs == []
        p.unlink()
