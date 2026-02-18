import logging
from pathlib import Path
from langchain_core.documents import Document
from src.config import get_settings
from src.embeddings.embedding_service import get_embeddings
from src.vectorstore.milvus_client import MilvusVectorStore
from src.ingestion.yaml_loader import KubernetesYAMLLoader
from src.ingestion.log_loader import KubernetesLogLoader
from src.ingestion.events_loader import KubernetesEventsLoader
from src.ingestion.markdown_loader import MarkdownDocumentLoader
from src.ingestion.helm_loader import HelmChartLoader

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Orchestrate document loading, chunking, embedding, and storage."""
    
    def __init__(self):
        self.settings = get_settings()
        self.vectorstore = MilvusVectorStore()
        self.embeddings = get_embeddings()
    
    def ingest(self, path: Path, doc_type: str | None = None) -> dict:
        """
        Ingest documents from path.
        
        Args:
            path: File or directory path
            doc_type: Override document type detection ("yaml", "log", "event", "markdown")
        
        Returns:
            Dictionary with ingestion stats
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        logger.info(f"Starting ingestion: {path}")
        
        # Detect or override document type
        if doc_type:
            doc_type = doc_type.lower()
        else:
            doc_type = self._detect_type(path)
        
        logger.info(f"Document type: {doc_type}")
        
        # Select loader
        loader = self._get_loader(doc_type)
        
        # Load and chunk
        documents = loader.load_and_chunk(path)
        
        if not documents:
            logger.warning(f"No documents loaded from {path}")
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "chunks_stored": 0,
                "errors": [],
            }
        
        # Store in Milvus
        chunk_ids = self.vectorstore.add_documents(documents)
        
        logger.info(
            f"Ingestion complete: {len(documents)} chunks stored. "
            f"First 3 IDs: {chunk_ids[:3]}"
        )
        
        return {
            "documents_loaded": len(documents),
            "chunks_created": len(documents),
            "chunks_stored": len(chunk_ids),
            "errors": [],
        }
    
    def _detect_type(self, path: Path) -> str:
        """Detect document type from path."""
        if path.is_file():
            suffix = path.suffix.lower()
            name_lower = path.name.lower()
            if "events" in name_lower:
                return "event"
            elif suffix in [".yaml", ".yml"]:
                return "yaml"
            elif suffix in [".log", ".txt"]:
                return "log"
            elif suffix == ".md":
                return "markdown"
            elif suffix == ".tpl":
                return "helm"
        elif path.is_dir():
            # Check for Helm chart structure
            if (path / "Chart.yaml").exists() or (path / "Chart.yml").exists():
                return "helm"
            # Check directory name hints
            dir_name = path.name.lower()
            if dir_name in ("events", "event"):
                return "event"
            elif dir_name in ("logs", "log"):
                return "log"
            elif dir_name in ("manifests", "manifest", "k8s", "kubernetes"):
                return "yaml"
            elif dir_name in ("docs", "documentation", "runbooks"):
                return "markdown"
            # Check subdirectories
            if (path / "manifests").exists():
                return "yaml"
            elif (path / "logs").exists():
                return "log"
            elif (path / "events").exists():
                return "event"
            elif (path / "templates").exists():
                return "helm"
        
        # Default
        return "yaml"
    
    def _get_loader(self, doc_type: str):
        """Get appropriate loader for document type."""
        loaders = {
            "yaml": KubernetesYAMLLoader,
            "log": KubernetesLogLoader,
            "event": KubernetesEventsLoader,
            "markdown": MarkdownDocumentLoader,
            "helm": HelmChartLoader,
        }
        
        loader_class = loaders.get(doc_type, KubernetesYAMLLoader)
        return loader_class(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )