#!/usr/bin/env python3
"""Ingest sample data into the Kubernetes Failure Intelligence knowledge base.

Usage:
    python -m scripts.ingest                    # Ingest all sample data
    python -m scripts.ingest --path data/sample/events --type event
    python -m scripts.ingest --path data/sample/manifests --type yaml
    python -m scripts.ingest --path data/sample/logs --type log
    python -m scripts.ingest --path data/sample/docs --type markdown
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default sample directories (relative to project root)
SAMPLE_DIRS = [
    ("data/sample/manifests", "yaml"),
    ("data/sample/logs", "log"),
    ("data/sample/events", "event"),
    ("data/sample/docs", "markdown"),
]


def ingest_path(pipeline: IngestionPipeline, path: Path, doc_type: str | None) -> dict:
    """Ingest a single path and return stats."""
    if not path.exists():
        logger.warning(f"Path does not exist, skipping: {path}")
        return {"documents_loaded": 0, "chunks_created": 0, "chunks_stored": 0, "errors": [str(path)]}

    logger.info(f"Ingesting: {path} (type={doc_type or 'auto'})")
    result = pipeline.ingest(path=path, doc_type=doc_type)
    logger.info(
        f"  -> loaded={result['documents_loaded']}, "
        f"chunks={result['chunks_created']}, "
        f"stored={result['chunks_stored']}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into K8s Failure Intelligence KB")
    parser.add_argument("--path", type=str, help="File or directory to ingest")
    parser.add_argument("--type", type=str, choices=["yaml", "log", "event", "markdown", "helm"],
                        help="Document type (auto-detected if omitted)")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Ingest all sample data directories")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = IngestionPipeline()

    totals = {"documents_loaded": 0, "chunks_created": 0, "chunks_stored": 0, "errors": []}

    if args.path:
        result = ingest_path(pipeline, Path(args.path), args.type)
        for k in ("documents_loaded", "chunks_created", "chunks_stored"):
            totals[k] += result[k]
        totals["errors"].extend(result.get("errors", []))

    elif args.all or (not args.path):
        # Ingest all sample directories
        for rel_path, doc_type in SAMPLE_DIRS:
            full_path = project_root / rel_path
            result = ingest_path(pipeline, full_path, doc_type)
            for k in ("documents_loaded", "chunks_created", "chunks_stored"):
                totals[k] += result[k]
            totals["errors"].extend(result.get("errors", []))

    logger.info("=" * 60)
    logger.info("Ingestion Summary:")
    logger.info(f"  Documents loaded : {totals['documents_loaded']}")
    logger.info(f"  Chunks created   : {totals['chunks_created']}")
    logger.info(f"  Chunks stored    : {totals['chunks_stored']}")
    if totals["errors"]:
        logger.warning(f"  Errors           : {len(totals['errors'])}")
    else:
        logger.info("  Errors           : 0")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
