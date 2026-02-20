"""Application settings — loaded from .env via Pydantic BaseSettings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        protected_namespaces=(), extra="ignore",
    )

    # Groq
    groq_api_key: str = ""
    model_name: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_device: str = "cpu"

    # Chroma
    chroma_persist_dir: str = "/tmp/chroma"
    chroma_collection: str = "k8s_failures"

    # Retrieval & chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 4

    # Memory
    memory_path: str = "/tmp/chat_memory.json"
    memory_max_messages: int = 40
    memory_max_sessions: int = 256

    # Logging
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()