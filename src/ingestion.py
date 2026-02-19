"""Unified document ingestion — auto-detects file type, chunks, adds metadata.

Supports:  .yaml / .yml  |  .json (K8s events)  |  .md  |  .log / .txt
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

_CHAR_SPLITTER: RecursiveCharacterTextSplitter | None = None


def _splitter() -> RecursiveCharacterTextSplitter:
    global _CHAR_SPLITTER
    if _CHAR_SPLITTER is None:
        s = get_settings()
        _CHAR_SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
        )
    return _CHAR_SPLITTER


def _stamp(docs: list[Document]) -> list[Document]:
    """Normalize metadata + add ingestion timestamp.

    Milvus requires all documents to have the same fields,
    so we ensure a canonical set of metadata keys.
    """
    now = datetime.now(timezone.utc).isoformat()
    canonical = ("source", "doc_type", "kind", "name", "namespace", "reason")
    for i, d in enumerate(docs):
        d.metadata.setdefault("ingested_at", now)
        d.metadata["chunk_index"] = i
        for key in canonical:
            d.metadata.setdefault(key, "")
        # Remove any extra/non-standard keys that would widen the schema
        allowed = set(canonical) | {"ingested_at", "chunk_index"}
        d.metadata = {k: v for k, v in d.metadata.items() if k in allowed}
    return docs


# ── YAML manifests ────────────────────────────────────────────────────────────

def _render_k8s(obj: dict) -> str:
    """Render a parsed K8s object as human-readable text."""
    lines = [
        f"Kind: {obj.get('kind', 'Unknown')}",
        f"Name: {obj.get('metadata', {}).get('name', 'unknown')}",
        f"Namespace: {obj.get('metadata', {}).get('namespace', 'default')}",
        f"API Version: {obj.get('apiVersion', 'v1')}",
    ]
    for section in ("spec", "status"):
        if section in obj:
            lines.append(f"{section.title()}:")
            lines.extend(_flat(obj[section], indent=2))
    return "\n".join(lines)


def _flat(d: dict, indent: int = 0) -> list[str]:
    prefix = " " * indent
    lines: list[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.extend(_flat(v, indent + 2))
        elif isinstance(v, list):
            lines.append(f"{prefix}{k}: [{len(v)} items]")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return lines


def _load_yaml(path: Path) -> list[Document]:
    docs: list[Document] = []
    try:
        objs = list(yaml.safe_load_all(path.read_text()))
    except yaml.YAMLError as exc:
        logger.error("YAML parse error in %s: %s", path, exc)
        return []
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        meta = {
            "source": str(path),
            "doc_type": "kubernetes_manifest",
            "kind": obj.get("kind", "Unknown"),
            "name": obj.get("metadata", {}).get("name", "unknown"),
            "namespace": obj.get("metadata", {}).get("namespace", "default"),
        }
        docs.append(Document(page_content=_render_k8s(obj), metadata=meta))
    return docs


# ── JSON events ───────────────────────────────────────────────────────────────

def _render_event(ev: dict) -> str:
    inv = ev.get("involvedObject", {})
    return "\n".join([
        f"Event: {ev.get('reason', 'Unknown')}",
        f"Type: {ev.get('type', 'Normal')}",
        f"Object: {inv.get('kind', '?')}/{inv.get('name', '?')} (ns: {inv.get('namespace', 'default')})",
        f"Count: {ev.get('count', 1)}",
        f"Message: {ev.get('message', '')}",
    ])


def _load_events(path: Path) -> list[Document]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error in %s: %s", path, exc)
        return []
    events = raw if isinstance(raw, list) else [raw]
    docs: list[Document] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        inv = ev.get("involvedObject", {})
        meta = {
            "source": str(path),
            "doc_type": "kubernetes_event",
            "reason": ev.get("reason", "Unknown"),
            "namespace": inv.get("namespace", "default"),
            "name": inv.get("name", "unknown"),
        }
        docs.append(Document(page_content=_render_event(ev), metadata=meta))
    return docs


# ── Markdown ──────────────────────────────────────────────────────────────────

_MD_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)


def _load_markdown(path: Path) -> list[Document]:
    text = path.read_text()
    splits = _MD_SPLITTER.split_text(text)
    for s in splits:
        s.metadata["source"] = str(path)
        s.metadata["doc_type"] = "markdown_document"
    return splits


# ── Logs / plain text ────────────────────────────────────────────────────────

def _load_log(path: Path) -> list[Document]:
    text = path.read_text()
    meta = {"source": str(path), "doc_type": "kubernetes_log"}
    chunks = _splitter().split_text(text)
    return [Document(page_content=c, metadata={**meta}) for c in chunks]


# ── public API ────────────────────────────────────────────────────────────────

_LOADERS = {
    ".yaml": _load_yaml,
    ".yml": _load_yaml,
    ".json": _load_events,
    ".md": _load_markdown,
    ".log": _load_log,
    ".txt": _load_log,
}


def ingest_file(path: Path) -> list[Document]:
    """Load + chunk a single file. Returns list of Document chunks."""
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        logger.warning("Unsupported file type: %s — skipping", path)
        return []
    docs = loader(path)
    # For YAML / events, apply character splitter on large docs
    if path.suffix.lower() in (".yaml", ".yml", ".json"):
        out: list[Document] = []
        for d in docs:
            if len(d.page_content) > get_settings().chunk_size:
                for chunk in _splitter().split_documents([d]):
                    out.append(chunk)
            else:
                out.append(d)
        docs = out
    return _stamp(docs)


def ingest_directory(directory: Path) -> list[Document]:
    """Recursively ingest all supported files under *directory*."""
    all_docs: list[Document] = []
    for ext in _LOADERS:
        for fp in sorted(directory.rglob(f"*{ext}")):
            logger.info("Ingesting %s", fp)
            all_docs.extend(ingest_file(fp))
    return _stamp(all_docs)
