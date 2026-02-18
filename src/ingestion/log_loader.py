import logging
import re
from itertools import chain
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)

class KubernetesLogLoader(BaseDocumentLoader):
    """Load Kubernetes logs with time-window grouping."""
    
    # Timestamp pattern (ISO 8601)
    TIMESTAMP_PATTERN = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, time_window: int = 30):
        super().__init__(chunk_size, chunk_overlap)
        self.time_window = time_window  # seconds
    
    def load(self, path: Path) -> list[Document]:
        """Load log file(s)."""
        documents = []
        
        if path.is_dir():
            for log_file in chain(path.glob("**/*.log"), path.glob("**/*.txt")):
                documents.extend(self._load_single_file(log_file))
        else:
            documents.extend(self._load_single_file(path))
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> list[Document]:
        """Load a single log file."""
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        documents = []
        current_group = []
        current_timestamp = None
        
        for line in lines:
            ts_match = re.search(self.TIMESTAMP_PATTERN, line)
            
            if ts_match:
                ts_str = ts_match.group(1)
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    ts = None
            else:
                ts = None
            
            # Group lines by time window
            if ts and current_timestamp:
                time_diff = abs((ts - current_timestamp).total_seconds())
                if time_diff > self.time_window:
                    # New time window
                    if current_group:
                        documents.append(self._make_document(
                            current_group,
                            file_path,
                            current_timestamp
                        ))
                    current_group = [line]
                    current_timestamp = ts
                else:
                    current_group.append(line)
            else:
                if ts:
                    if current_group:
                        documents.append(self._make_document(
                            current_group,
                            file_path,
                            current_timestamp
                        ))
                    current_group = [line]
                    current_timestamp = ts
                else:
                    if current_group:
                        current_group.append(line)
        
        # Flush last group
        if current_group:
            documents.append(self._make_document(
                current_group,
                file_path,
                current_timestamp
            ))
        
        logger.debug(f"Loaded {len(documents)} log groups from {file_path}")
        return documents
    
    def _make_document(self, lines: list[str], file_path: Path, timestamp: datetime | None) -> Document:
        """Create a document from log lines."""
        text = "".join(lines)
        
        metadata = {
            "source": str(file_path),
            "doc_type": "kubernetes_log",
            "timestamp": timestamp.isoformat() if timestamp else "",
        }
        
        return Document(page_content=text, metadata=metadata)
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk logs using custom separators."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " "],
        )
        
        chunked = []
        for doc in documents:
            chunks = splitter.split_documents([doc])
            chunked.extend(chunks)
        
        return chunked