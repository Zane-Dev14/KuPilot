"""LangChain tools for K8s failure diagnosis.

Each tool is decorated with ``@tool`` so it can be bound to an LLM or
used inside a LangGraph agent.
"""

import logging
from langchain_core.tools import tool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the Kubernetes failure knowledge base for relevant documents.

    Use this tool when you need to look up information about Kubernetes
    failures, error messages, manifests, logs, or events.

    Args:
        query: Natural-language search query describing the failure or topic.

    Returns:
        Formatted string of the most relevant document excerpts.
    """
    # Import lazily to avoid circular imports at module load time
    from src.retrieval.retriever import EnhancedRetriever

    retriever = EnhancedRetriever()
    chain = retriever.get_chain()
    docs: list[Document] = chain.invoke(query)

    if not docs:
        return "No relevant documents found in the knowledge base."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        kind = doc.metadata.get("kind", "")
        header = f"[{i}] {kind} — {source}" if kind else f"[{i}] {source}"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


@tool
def list_recent_events(namespace: str = "default") -> str:
    """List recent Kubernetes events stored in the knowledge base.

    Use this tool to find Warning or Normal events that may indicate
    failures for pods or nodes in a specific namespace.

    Args:
        namespace: Kubernetes namespace to filter events by (default: "default").

    Returns:
        Formatted list of matching event documents.
    """
    from src.vectorstore.milvus_client import MilvusVectorStore

    vs = MilvusVectorStore()
    retriever = vs.as_retriever(search_type="similarity", k=10)

    # Use namespace as the search text — the vector search is semantic,
    # so including the namespace in the query helps surface relevant events.
    docs: list[Document] = retriever.invoke(
        f"kubernetes events namespace {namespace}"
    )

    # Post-filter by namespace metadata
    filtered = [
        d for d in docs
        if d.metadata.get("namespace") == namespace
        and d.metadata.get("doc_type") == "kubernetes_event"
    ]

    if not filtered:
        return f"No events found for namespace '{namespace}'."

    parts: list[str] = []
    for doc in filtered:
        reason = doc.metadata.get("reason", "")
        obj = doc.metadata.get("involved_object_name", "")
        parts.append(f"• {reason} on {obj}\n  {doc.page_content[:200]}")

    return "\n".join(parts)
