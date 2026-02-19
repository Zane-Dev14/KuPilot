"""Embeddings, Milvus vector store, and cross-encoder reranker.

Combines three concerns into one module because they share lifecycle:
  get_embeddings()  → singleton HuggingFace embedding model
  get_reranker()    → singleton CrossEncoder reranker
  MilvusStore       → thin wrapper around langchain-milvus
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from sentence_transformers import CrossEncoder

from src.config import get_settings

logger = logging.getLogger(__name__)

# ── Embeddings singleton ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load embedding model once and cache it."""
    s = get_settings()
    logger.info("Loading embeddings: %s (device=%s)", s.embedding_model, s.embedding_device)
    emb = HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": s.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    dim = len(emb.embed_query("hello"))
    logger.info("Embeddings ready — %d-dim", dim)
    return emb


# ── Reranker singleton ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load cross-encoder reranker once and cache it."""
    s = get_settings()
    logger.info("Loading reranker: %s", s.reranker_model)
    model = CrossEncoder(s.reranker_model)
    logger.info("Reranker ready")
    return model


def rerank(docs: list[Document], query: str, top_k: int = 4) -> list[Document]:
    """Score each doc against the query and return the top-k."""
    if not docs:
        return []
    model = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ── Milvus wrapper ───────────────────────────────────────────────────────────

class MilvusStore:
    """Thin wrapper: add documents, search, health check."""

    def __init__(self, drop_old: bool = False) -> None:
        self._settings = get_settings()
        self._drop_old = drop_old
        self._vs: Optional[Milvus] = None

    # -- lazy init ----------------------------------------------------------
    def _get_vs(self) -> Milvus:
        if self._vs is None:
            s = self._settings
            logger.info("Connecting to Milvus: %s / %s", s.milvus_uri, s.milvus_collection)
            self._vs = Milvus(
                embedding_function=get_embeddings(),
                collection_name=s.milvus_collection,
                connection_args={"uri": s.milvus_uri},
                drop_old=self._drop_old,
                auto_id=True,
            )
            logger.info("Milvus connected")
        return self._vs

    # -- public API ---------------------------------------------------------
    def add_documents(self, docs: list[Document]) -> list[str]:
        logger.info("Storing %d chunks in Milvus …", len(docs))
        ids = self._get_vs().add_documents(docs)
        logger.info("Stored %d chunks", len(ids))
        return ids

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Similarity search → rerank → return top_k."""
        k = k or self._settings.retrieval_top_k
        # Fetch extra candidates for reranking
        fetch_k = min(k * 3, 20)
        docs = self._get_vs().similarity_search(query, k=fetch_k)
        return rerank(docs, query, top_k=k)

    def health_check(self) -> bool:
        try:
            self._get_vs()
            return True
        except Exception as exc:
            logger.error("Milvus health check failed: %s", exc)
            return False
