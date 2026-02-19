"""RAG diagnosis chain — retrieval + adaptive model selection + generation.

Key features:
  * Multi-model: simple queries → llama3.1, complex → deepseek-r1:32b
  * Reranking:   vector search → cross-encoder rerank → top-K context
  * Memory:      includes recent chat history in the prompt
  * Structured:  returns FailureDiagnosis pydantic model
"""

import json
import logging
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.config import get_settings
from src.memory import get_chat_memory
from src.vectorstore import MilvusStore

logger = logging.getLogger(__name__)


# ── Pydantic schemas (inline — no separate models package) ────────────────────

class FailureDiagnosis(BaseModel):
    """Structured output returned by the RAG chain."""
    root_cause: str = Field(default="Unknown", description="Root cause of the failure")
    explanation: str = Field(default="", description="Detailed explanation")
    recommended_fix: str = Field(default="", description="Actionable fix steps")
    confidence: float = Field(default=0.5, description="0.0–1.0")
    sources: list[str] = Field(default_factory=list)
    model_used: Optional[str] = None


# ── Model selector (heuristic) ───────────────────────────────────────────────

_REASONING_KW = ["why", "explain", "diagnose", "troubleshoot", "root cause", "how"]
_UNCERTAIN_KW = ["maybe", "could", "might", "possibly", "not sure"]


def estimate_complexity(query: str) -> float:
    """Score 0.0–1.0 based on keyword heuristics."""
    q = query.lower()
    score = 0.0
    if any(kw in q for kw in _REASONING_KW):
        score += 0.3
    if q.count("?") > 1:
        score += min(0.2 * (q.count("?") - 1), 0.4)
    if len(query) > 200:
        score += 0.1
    if any(kw in q for kw in _UNCERTAIN_KW):
        score += 0.2
    return min(score, 1.0)


def select_model(query: str, force: str | None = None) -> str:
    """Pick llama3.1 or deepseek-r1:32b based on complexity."""
    if force:
        return force
    s = get_settings()
    c = estimate_complexity(query)
    model = s.complex_model if c >= s.query_complexity_threshold else s.simple_model
    logger.info("Complexity %.2f → %s", c, model)
    return model


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a Kubernetes failure diagnosis expert.
Given the CONTEXT (retrieved documents) and the USER QUERY, respond with a JSON object:
{{
  "root_cause": "...",
  "explanation": "...",
  "recommended_fix": "...",
  "confidence": 0.0-1.0
}}
Be concise, technical, and actionable.  Return ONLY the JSON object."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "CONTEXT:\n{context}\n\nCHAT HISTORY:\n{history}\n\nUSER QUERY:\n{query}"),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no documents found)"
    parts: list[str] = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        kind = d.metadata.get("kind") or d.metadata.get("doc_type", "")
        parts.append(f"[{i}] {kind} — {src}\n{d.page_content}")
    return "\n---\n".join(parts)


def _format_history(messages: list[BaseMessage]) -> str:
    if not messages:
        return "(none)"
    lines = []
    for m in messages[-6:]:                       # last 3 turns
        role = "User" if m.type == "human" else "AI"
        lines.append(f"{role}: {m.content[:200]}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from LLM output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Look for ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Look for first { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ── RAG Chain class ───────────────────────────────────────────────────────────

class RAGChain:
    """Retrieve → select model → generate diagnosis."""

    def __init__(self) -> None:
        self.store = MilvusStore()
        self.memory = get_chat_memory()

    def diagnose(
        self,
        query: str,
        session_id: str = "default",
        force_model: str | None = None,
    ) -> FailureDiagnosis:
        # 1. Retrieve + rerank
        docs = self.store.search(query)
        logger.info("Retrieved %d docs for query", len(docs))

        # 2. Select model
        model_name = select_model(query, force_model)

        # 3. Build prompt inputs
        history = self.memory.get_history(session_id)
        context_str = _format_docs(docs)
        history_str = _format_history(history)

        # 4. Call LLM
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=get_settings().ollama_base_url,
        )
        chain = _PROMPT | llm
        result = chain.invoke({
            "context": context_str,
            "history": history_str,
            "query": query,
        })
        raw_text = result.content if hasattr(result, "content") else str(result)

        # 5. Parse output → FailureDiagnosis
        parsed = _parse_json(raw_text)
        diagnosis = FailureDiagnosis(
            root_cause=parsed.get("root_cause", raw_text[:200]),
            explanation=parsed.get("explanation", ""),
            recommended_fix=parsed.get("recommended_fix", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            sources=[d.metadata.get("source", "?") for d in docs],
            model_used=model_name,
        )

        # 6. Update memory
        self.memory.add_user_message(session_id, query)
        self.memory.add_ai_message(session_id, diagnosis.root_cause)

        return diagnosis
