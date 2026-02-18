import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.ingestion.base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)

class MarkdownDocumentLoader(BaseDocumentLoader):
    """Load Markdown runbooks and documentation."""
    
    def load(self, path: Path) -> list[Document]:
        """Load markdown file(s)."""
        documents = []
        
        if path.is_dir():
            for md_file in path.glob("**/*.md"):
                documents.extend(self._load_single_file(md_file))
        else:
            documents.extend(self._load_single_file(path))
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> list[Document]:
        """Load a single markdown file."""
        with open(file_path, "r") as f:
            content = f.read()
        
        metadata = {
            "source": str(file_path),
            "doc_type": "markdown_document",
        }
        
        documents = [Document(page_content=content, metadata=metadata)]
        return documents
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk by markdown headers."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        chunked = []
        for doc in documents:
            try:
                splits = md_splitter.split_text(doc.page_content)
                for split in splits:
                    # Preserve original metadata + header metadata
                    split.metadata.update(doc.metadata)
                    chunked.append(split)
            except Exception as e:
                logger.warning(f"Failed to split {doc.metadata.get('source')}: {e}")
                # Fallback to character-based splitting
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunks = fallback_splitter.split_documents([doc])
                chunked.extend(chunks)
        
        return chunked