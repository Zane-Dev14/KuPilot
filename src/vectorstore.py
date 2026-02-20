"""Embeddings and Chroma vector store."""

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    logger.info("Loading embeddings: %s (device=%s)", s.embedding_model, s.embedding_device)
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": s.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStore:
    """Thin wrapper around Chroma with persistent storage."""

    def __init__(self, drop_old: bool = False):
        self._settings = get_settings()
        self._drop_old = drop_old
        self._vs: Chroma | None = None

    def _get_vs(self) -> Chroma:
        if self._vs is None:
            s = self._settings
            if self._drop_old:
                import chromadb
                client = chromadb.PersistentClient(path=s.chroma_persist_dir)
                try:
                    client.delete_collection(s.chroma_collection)
                    logger.info("Dropped existing collection '%s'", s.chroma_collection)
                except ValueError:
                    pass
            self._vs = Chroma(
                collection_name=s.chroma_collection,
                embedding_function=get_embeddings(),
                persist_directory=s.chroma_persist_dir,
            )
        return self._vs

    def add_documents(self, docs: list[Document]) -> list[str]:
        logger.info("Storing %d chunks in Chroma", len(docs))
        return self._get_vs().add_documents(docs)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self._settings.retrieval_top_k
        return self._get_vs().similarity_search(query, k=k)

    def health_check(self) -> bool:
        try:
            self._get_vs()
            return True
        except Exception:
            return False
