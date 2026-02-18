"""LangChain tools for the K8s Failure Intelligence Copilot.

These tools can be bound to ChatOllama via `llm.bind_tools()` or used
in a LangGraph agent node.
"""

from src.tools.k8s_tools import search_knowledge_base, list_recent_events

__all__ = [
    "search_knowledge_base",
    "list_recent_events",
]
