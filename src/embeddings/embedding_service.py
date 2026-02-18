import logging
from typing import Optional
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import get_settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Singleton wrapper around HuggingFaceEmbeddings for bge-m3."""
    
    _instance: Optional["EmbeddingService"] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingService._embeddings is None:
            self._load_embeddings()
    
    def _load_embeddings(self) -> None:
        """Load bge-m3 embeddings model."""
        settings = get_settings()
        logger.info(f"Loading embeddings model: {settings.embedding_model}")
        
        EmbeddingService._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Verify dimension
        test_embedding = EmbeddingService._embeddings.embed_query("test")
        actual_dim = len(test_embedding)
        logger.info(f"Embeddings loaded: {actual_dim}-dimensional")
        
        if actual_dim != settings.embedding_dimension:
            logger.warning(
                f"Embedding dimension mismatch: expected {settings.embedding_dimension}, got {actual_dim}"
            )
    
    def get(self) -> HuggingFaceEmbeddings:
        """Get the embeddings instance."""
        if self._embeddings is None:
            raise RuntimeError("EmbeddingService not initialized")
        return self._embeddings

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Get cached embeddings service."""
    return EmbeddingService().get()