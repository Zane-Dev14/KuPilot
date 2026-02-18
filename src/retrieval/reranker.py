import logging
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.config import get_settings

logger = logging.getLogger(__name__)

class RerankerService:
    """Cross-encoder reranker using bge-reranker-v2-m3."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        settings = get_settings()
        logger.info(f"Loading reranker model: {settings.reranker_model}")
        
        self._model = CrossEncoder(settings.reranker_model)
        logger.info("Reranker model loaded successfully")
    
    def rerank(
        self,
        documents: list[Document],
        query: str,
        top_k: int = 4,
    ) -> list[Document]:
        """
        Rerank documents by relevance to query.
        
        Args:
            documents: List of documents to rerank
            query: Query string
            top_k: Return top K documents
        
        Returns:
            Reranked documents (top_k)
        """
        if not documents:
            return []
        
        logger.debug(f"Reranking {len(documents)} documents for query: {query[:100]}...")
        
        # Prepare document texts
        doc_texts = [doc.page_content for doc in documents]
        
        # Score document-query pairs
        scores = self._model.predict([[query, doc] for doc in doc_texts])
        
        # Sort by score (descending)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        # Return top K
        reranked = [doc for doc, score in ranked[:top_k]]
        logger.debug(f"Reranking complete. Returned {len(reranked)} documents")
        
        return reranked