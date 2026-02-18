import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class BaseDocumentLoader(ABC):
    """Abstract base loader for all document sources."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load raw documents from path."""
        pass
    
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk documents into smaller pieces."""
        pass
    
    def load_and_chunk(self, path: Path) -> list[Document]:
        """Template method: load and chunk."""
        logger.info(f"Loading documents from: {path}")
        documents = self.load(path)
        logger.info(f"Loaded {len(documents)} documents. Chunking...")
        chunked = self.chunk(documents)
        logger.info(f"Chunked into {len(chunked)} pieces")
        
        # Add metadata to all chunks
        for i, doc in enumerate(chunked):
            doc.metadata.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())
            doc.metadata.setdefault("chunk_index", i)
        
        return chunked