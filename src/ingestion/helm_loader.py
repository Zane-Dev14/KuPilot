import logging
import re
from itertools import chain
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.yaml_loader import KubernetesYAMLLoader

logger = logging.getLogger(__name__)


class HelmChartLoader(KubernetesYAMLLoader):
    """Load Helm chart templates with template-expression awareness.

    Extends the YAML loader to handle Helm-specific features:
    - Strips Go template expressions ({{ }}) before YAML parsing
    - Extracts Chart.yaml / values.yaml metadata
    - Preserves template context in document metadata
    """

    # Helm template expression pattern
    TEMPLATE_EXPR = re.compile(r"\{\{.*?\}\}")

    def load(self, path: Path) -> list[Document]:
        """Load Helm chart from a chart directory or a single template file."""
        documents: list[Document] = []

        if path.is_dir():
            # Load Chart.yaml metadata if present
            chart_meta = self._load_chart_meta(path)

            # Load templates/ directory
            templates_dir = path / "templates"
            if templates_dir.is_dir():
                for tpl_file in chain(
                    templates_dir.glob("**/*.yaml"),
                    templates_dir.glob("**/*.yml"),
                    templates_dir.glob("**/*.tpl"),
                ):
                    documents.extend(
                        self._load_helm_template(tpl_file, chart_meta)
                    )

            # Also load values.yaml as a document
            for values_file in ["values.yaml", "values.yml"]:
                vf = path / values_file
                if vf.exists():
                    documents.extend(self._load_values_file(vf, chart_meta))
        else:
            # Single file — treat as a template
            documents.extend(self._load_helm_template(path, {}))

        logger.info(f"Loaded {len(documents)} Helm documents from {path}")
        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_chart_meta(self, chart_dir: Path) -> dict:
        """Extract metadata from Chart.yaml."""
        import yaml

        for name in ("Chart.yaml", "Chart.yml"):
            chart_file = chart_dir / name
            if chart_file.exists():
                try:
                    with open(chart_file, "r") as f:
                        data = yaml.safe_load(f) or {}
                    return {
                        "chart_name": data.get("name", "unknown"),
                        "chart_version": data.get("version", "0.0.0"),
                        "chart_description": data.get("description", ""),
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse {chart_file}: {e}")
        return {}

    def _load_helm_template(
        self, file_path: Path, chart_meta: dict
    ) -> list[Document]:
        """Load a single Helm template, stripping Go template expressions."""
        with open(file_path, "r") as f:
            raw = f.read()

        # Strip {{ ... }} expressions so yaml.safe_load can parse the document
        cleaned = self.TEMPLATE_EXPR.sub("HELM_TEMPLATE_EXPR", raw)

        # Try to parse as K8s YAML
        docs = self._load_single_file_from_text(cleaned, file_path, chart_meta)
        if not docs:
            # Fallback: store as raw text
            metadata = {
                "source": str(file_path),
                "doc_type": "helm_template",
                **chart_meta,
            }
            docs = [Document(page_content=raw, metadata=metadata)]

        return docs

    def _load_single_file_from_text(
        self, text: str, file_path: Path, chart_meta: dict
    ) -> list[Document]:
        """Parse cleaned YAML text into Documents."""
        import yaml

        documents: list[Document] = []
        try:
            objs = list(yaml.safe_load_all(text))
        except yaml.YAMLError:
            return []

        for obj in objs:
            if not isinstance(obj, dict):
                continue

            kind = obj.get("kind", "Unknown")
            name = obj.get("metadata", {}).get("name", "unknown")
            namespace = obj.get("metadata", {}).get("namespace", "default")

            rendered = self._render_k8s_object(obj)

            metadata = {
                "source": str(file_path),
                "doc_type": "helm_template",
                "kind": kind,
                "name": name,
                "namespace": namespace,
                **chart_meta,
            }
            documents.append(Document(page_content=rendered, metadata=metadata))

        return documents

    def _load_values_file(
        self, file_path: Path, chart_meta: dict
    ) -> list[Document]:
        """Load values.yaml as a document."""
        with open(file_path, "r") as f:
            content = f.read()

        metadata = {
            "source": str(file_path),
            "doc_type": "helm_values",
            **chart_meta,
        }
        return [Document(page_content=content, metadata=metadata)]

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk Helm documents."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n---\n", "\nSpec:\n", "\n\n", "\n", " "],
        )

        chunked: list[Document] = []
        for doc in documents:
            chunks = splitter.split_documents([doc])
            chunked.extend(chunks)

        return chunked
