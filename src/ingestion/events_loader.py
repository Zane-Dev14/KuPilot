import logging
import json
import yaml
from itertools import chain
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)

class KubernetesEventsLoader(BaseDocumentLoader):
    """Load Kubernetes Event objects (JSON or YAML)."""
    
    def load(self, path: Path) -> list[Document]:
        """Load event file(s)."""
        documents = []
        
        if path.is_dir():
            # Load all event files
            for event_file in chain(path.glob("**/*.events.json"), path.glob("**/*.events.yaml")):
                documents.extend(self._load_single_file(event_file))
        else:
            documents.extend(self._load_single_file(path))
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> list[Document]:
        """Load a single event file."""
        documents = []
        
        try:
            if file_path.suffix == ".json":
                with open(file_path, "r") as f:
                    events = json.load(f)
                    if not isinstance(events, list):
                        events = [events]
            else:
                with open(file_path, "r") as f:
                    events = list(yaml.safe_load_all(f.read()))
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []
        
        for event_obj in events:
            if not event_obj:
                continue
            
            # Extract event fields
            reason = event_obj.get("reason", "Unknown")
            event_type = event_obj.get("type", "Normal")
            message = event_obj.get("message", "")
            
            involved = event_obj.get("involvedObject", {})
            involved_kind = involved.get("kind", "Unknown")
            involved_name = involved.get("name", "unknown")
            namespace = involved.get("namespace", "default")
            
            first_timestamp = event_obj.get("firstTimestamp", "")
            last_timestamp = event_obj.get("lastTimestamp", "")
            count = event_obj.get("count", 1)
            
            # Render as readable text
            text = self._render_event(event_obj)
            
            metadata = {
                "source": str(file_path),
                "doc_type": "kubernetes_event",
                "kind": "Event",
                "reason": reason,
                "event_type": event_type,
                "involved_object_kind": involved_kind,
                "involved_object_name": involved_name,
                "namespace": namespace,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "count": count,
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        logger.debug(f"Loaded {len(documents)} events from {file_path}")
        return documents
    
    def _render_event(self, event: dict) -> str:
        """Render event as readable text."""
        lines = []
        
        reason = event.get("reason", "Unknown")
        event_type = event.get("type", "Normal")
        message = event.get("message", "")
        
        involved = event.get("involvedObject", {})
        involved_kind = involved.get("kind", "Unknown")
        involved_name = involved.get("name", "unknown")
        namespace = involved.get("namespace", "default")
        
        count = event.get("count", 1)
        first_ts = event.get("firstTimestamp", "")
        last_ts = event.get("lastTimestamp", "")
        
        lines.append(f"Event: {reason}")
        lines.append(f"Type: {event_type}")
        lines.append(f"Involved Object: {involved_kind}/{involved_name} (namespace: {namespace})")
        lines.append(f"Count: {count}")
        lines.append(f"First Occurrence: {first_ts}")
        lines.append(f"Last Occurrence: {last_ts}")
        lines.append(f"Message: {message}")
        
        return "\n".join(lines)
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Events are typically short; minimal chunking."""
        # Group by reason + involved object
        grouped = {}
        for doc in documents:
            key = (doc.metadata.get("reason"), doc.metadata.get("involved_object_name"))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(doc)
        
        # Light chunking if any event is large
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        chunked = []
        for docs in grouped.values():
            for doc in docs:
                if len(doc.page_content) > self.chunk_size:
                    chunks = splitter.split_documents([doc])
                    chunked.extend(chunks)
                else:
                    chunked.append(doc)
        
        return chunked