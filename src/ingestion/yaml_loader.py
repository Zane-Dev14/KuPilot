import logging
import yaml
from itertools import chain
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)

class KubernetesYAMLLoader(BaseDocumentLoader):
    """Load Kubernetes YAML manifests and Helm templates."""
    
    def load(self, path: Path) -> list[Document]:
        """Load YAML file(s), supporting multi-doc format."""
        documents = []
        
        if path.is_dir():
            # Load all YAML files in directory
            for yaml_file in chain(path.glob("**/*.yaml"), path.glob("**/*.yml")):
                documents.extend(self._load_single_file(yaml_file))
        else:
            documents.extend(self._load_single_file(path))
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> list[Document]:
        """Load a single YAML file, handling multi-doc format."""
        documents = []
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Split on document separator
        try:
            docs = list(yaml.safe_load_all(content))
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []
        
        for doc_obj in docs:
            if not doc_obj:
                continue  # Skip empty docs
            
            # Extract K8s metadata
            kind = doc_obj.get("kind", "Unknown")
            name = doc_obj.get("metadata", {}).get("name", "unknown")
            namespace = doc_obj.get("metadata", {}).get("namespace", "default")
            api_version = doc_obj.get("apiVersion", "v1")
            labels = doc_obj.get("metadata", {}).get("labels", {})
            
            # Convert to readable text (not raw YAML)
            text = self._render_k8s_object(doc_obj)
            
            metadata = {
                "source": str(file_path),
                "doc_type": "kubernetes_manifest",
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "api_version": api_version,
                "labels": labels,
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        logger.debug(f"Loaded {len(documents)} K8s objects from {file_path}")
        return documents
    
    def _render_k8s_object(self, obj: dict) -> str:
        """Render Kubernetes object as readable text."""
        lines = []
        kind = obj.get("kind", "Unknown")
        name = obj.get("metadata", {}).get("name", "unknown")
        namespace = obj.get("metadata", {}).get("namespace", "default")
        
        lines.append(f"Kind: {kind}")
        lines.append(f"Name: {name}")
        lines.append(f"Namespace: {namespace}")
        lines.append(f"API Version: {obj.get('apiVersion', 'v1')}")
        
        if "spec" in obj:
            lines.append("Spec:")
            lines.extend(self._render_dict(obj["spec"], indent=2))
        
        if "status" in obj:
            lines.append("Status:")
            lines.extend(self._render_dict(obj["status"], indent=2))
        
        return "\n".join(lines)
    
    def _render_dict(self, d: dict, indent: int = 0) -> list[str]:
        """Recursively render dict as text."""
        lines = []
        prefix = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}{k}:")
                lines.extend(self._render_dict(v, indent + 2))
            elif isinstance(v, list):
                lines.append(f"{prefix}{k}: [{len(v)} items]")
            else:
                lines.append(f"{prefix}{k}: {v}")
        return lines
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk documents by size."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\nSpec:\n", "\n\n", "\n", " "],
        )
        
        chunked = []
        for doc in documents:
            chunks = splitter.split_documents([doc])
            chunked.extend(chunks)
        
        return chunked