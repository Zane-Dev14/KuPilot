from pydantic_settings import BaseSettings
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = "root:Milvus"
    milvus_db: str = "default"
    milvus_collection: str = "k8s_failures"
    
    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 384  # bge-m3 outputs 384-dim
    embedding_device: str = "cpu"  # "mps" for Apple Silicon
    
    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 128
    
    # LLMs
    simple_model: str = "llama3.1"
    complex_model: str = "deepseek-r1:32b"
    ollama_base_url: str = "http://localhost:11434"
    
    # Model Selection
    query_complexity_threshold: float = 0.7  # Score 0-1
    
    # Milvus Index Parameters (HNSW)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 256
    search_ef: int = 128
    
    # Document Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval
    retrieval_top_k: int = 4
    retrieval_rerank_k: int = 10  # Retrieve 10, rerank to 4
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()