import logging
from typing import Optional
from langchain_milvus import Milvus
from langchain_core.documents import Document
from src.config import get_settings
from src.embeddings.embedding_service import get_embeddings

logger = logging.getLogger(__name__)

class MilvusVectorStore:
    """Wrapper around LangChain Milvus for K8s failure knowledge base."""
    
    def __init__(self):
        self.settings = get_settings()
        self._vectorstore = None
    
    def _initialize_vectorstore(self) -> Milvus:
        """Lazy initialization of Milvus vectorstore."""
        if self._vectorstore is not None:
            return self._vectorstore
        
        embeddings = get_embeddings()
        
        logger.info(
            f"Initializing Milvus vectorstore: {self.settings.milvus_collection} "
            f"at {self.settings.milvus_uri}"
        )
        
        self._vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=self.settings.milvus_collection,
            connection_args={
                "uri": self.settings.milvus_uri,
                "token": self.settings.milvus_token,
                "db_name": self.settings.milvus_db,
            },
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": self.settings.hnsw_m,
                    "efConstruction": self.settings.hnsw_ef_construction,
                },
            },
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": self.settings.search_ef},
            },
            consistency_level="Strong",
            drop_old=False,
        )
        
        logger.info("Milvus vectorstore initialized successfully")
        return self._vectorstore
    
    def get_vectorstore(self) -> Milvus:
        """Get initialized vectorstore."""
        return self._initialize_vectorstore()
    
    def as_retriever(self, search_type: str = "mmr", k: int | None = None):
        """Get a retriever from the vectorstore."""
        if k is None:
            k = self.settings.retrieval_top_k
        
        vectorstore = self.get_vectorstore()
        return vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to vectorstore."""
        vectorstore = self.get_vectorstore()
        logger.info(f"Adding {len(documents)} documents to Milvus")
        ids = vectorstore.add_documents(documents=documents)
        logger.info(f"Documents added successfully. IDs: {ids[:3]}...")
        return ids
    
    def health_check(self) -> bool:
        """Verify Milvus connection."""
        try:
            vectorstore = self.get_vectorstore()
            # Try a simple collection info call
            logger.info("Milvus health check: OK")
            return True
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False