"""Tests for src.config — settings loading and defaults."""

from src.config import Settings, get_settings


def test_default_settings():
    """All defaults should be set without .env file."""
    s = Settings()
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dimension == 384
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.simple_model == "llama3.1"
    assert s.complex_model == "deepseek-r1:32b"
    assert s.query_complexity_threshold == 0.7
    assert s.milvus_uri == "http://localhost:19530"
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.retrieval_top_k == 4
    assert s.retrieval_rerank_k == 10
    assert s.hnsw_m == 16
    assert s.hnsw_ef_construction == 256
    assert s.search_ef == 128


def test_get_settings_cached():
    """get_settings should return the same object (lru_cache)."""
    a = get_settings()
    b = get_settings()
    assert a is b
