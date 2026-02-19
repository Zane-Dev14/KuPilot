"""RAG diagnosis chain — retrieval + adaptive model selection + generation.

Key features:
  * Multi-model: simple queries → llama3.1, complex → Qwen3-Coder:30b
  * Reranking:   vector search → cross-encoder rerank → top-K context
  * Memory:      includes recent chat history in the prompt
  * Structured:  returns FailureDiagnosis pydantic model
"""

import json
import logging
import re
from pathlib import PurePosixPath
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

_REASONING_KW = ["why", "explain", "diagnose", "troubleshoot", "root cause",
                 "how", "analyze", "investigate", "debug", "cause"]
_UNCERTAIN_KW = ["maybe", "could", "might", "possibly", "not sure",
                 "intermittent", "sometimes", "random"]
_COMPOUND_KW  = ["and", "but", "also", "plus", "additionally", "as well",
                 "however", "although", "even though", "while"]
_TECHNICAL_KW = ["oomkilled", "oom", "crashloopbackoff", "crashloop",
                 "imagepullbackoff", "imagepull", "failedscheduling",
                 "evicted", "cordon", "taint", "affinity", "pdb",
                 "hpa", "vpa", "resource quota", "limitrange",
                 "network policy", "init container", "sidecar",
                 "liveness", "readiness", "startup probe",
                 "persistent volume", "configmap", "secret"]


def _word_in(word: str, text: str) -> bool:
    """Check if a multi-word keyword appears in text (word-boundary-aware)."""
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))


def estimate_complexity(query: str) -> float:
    """Score 0.0–1.0 based on keyword heuristics.

    Each matching signal adds to the score independently so that
    multi-faceted queries reliably cross the routing threshold.
    """
    q = query.lower()
    score = 0.0

    # Reasoning keywords — each match adds weight (capped)
    reasoning_hits = sum(1 for kw in _REASONING_KW if _word_in(kw, q))
    score += min(reasoning_hits * 0.20, 0.50)

    # Uncertainty / ambiguity
    if any(_word_in(kw, q) for kw in _UNCERTAIN_KW):
        score += 0.15

    # Multiple question marks → multi-part question
    qmarks = q.count("?")
    if qmarks > 1:
        score += min(qmarks * 0.10, 0.30)

    # Compound / multi-clause
    compound_hits = sum(1 for kw in _COMPOUND_KW if _word_in(kw, q))
    if compound_hits:
        score += min(compound_hits * 0.10, 0.20)

    # Technical depth
    tech_hits = sum(1 for kw in _TECHNICAL_KW if kw in q)
    score += min(tech_hits * 0.10, 0.30)

    # Length signals (longer = more context = harder)
    words = len(query.split())
    if words > 15:
        score += 0.10
    if words > 30:
        score += 0.10

    return round(min(score, 1.0), 2)


def select_model(query: str, force: str | None = None) -> str:
    """Pick llama3.1 or Qwen3-Coder:30b based on complexity."""
    if force:
        return force
    s = get_settings()
    c = estimate_complexity(query)
    model = s.complex_model if c >= s.query_complexity_threshold else s.simple_model
    logger.info("Complexity %.2f → %s", c, model)
    return model


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a Kubernetes failure diagnosis expert integrated with a RAG knowledge base.

RULES:
1. Base your answers ONLY on the CONTEXT documents and CHAT HISTORY provided.
   Do NOT invent pod names, namespaces, or details that are not in the context.
2. If the user asks a conversational question (e.g. "what did I ask before?",
   "summarise our chat", yes/no questions), answer it naturally using CHAT HISTORY.
   Still use the JSON format but put your natural answer in "root_cause" and
   set "recommended_fix" to "N/A".
3. If the question is completely outside Kubernetes operations, reply helpfully
   but set confidence to 0.0 and recommended_fix to "N/A".
4. When diagnosing failures, cite specific evidence from the context (event
   messages, resource values, error strings, timestamps) rather than giving
   generic advice.
5. For the fix, give concrete steps or commands, not vague suggestions.

Always respond with ONLY this JSON (no markdown fences, no extra text):
{{
  "root_cause": "<concise root cause or direct answer>",
  "explanation": "<detailed explanation citing evidence from context>",
  "recommended_fix": "<specific actionable steps>",
  "confidence": 0.0-1.0
}}"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",
     "CONTEXT (retrieved documents):\n{context}\n\n"
     "CHAT HISTORY (recent conversation):\n{history}\n\n"
     "USER QUERY:\n{query}"),
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
        return "(no previous conversation)"
    lines = []
    for m in messages[-10:]:                      # last 5 turns
        role = "User" if m.type == "human" else "AI"
        lines.append(f"{role}: {m.content[:500]}")
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
        parsed = _parse_json(str(raw_text))

        # Null-safe extraction — LLMs sometimes return null for fields
        root_cause = parsed.get("root_cause") or str(raw_text)[:300]
        explanation = parsed.get("explanation") or ""
        fix = parsed.get("recommended_fix") or ""
        try:
            conf = float(parsed.get("confidence") or 0.5)
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = 0.5

        # Clean source paths — show just the relative data/ portion
        raw_sources = [d.metadata.get("source", "?") for d in docs]
        clean_sources: list[str] = []
        for s in raw_sources:
            idx = s.find("data/")
            clean_sources.append(s[idx:] if idx != -1 else s)

        diagnosis = FailureDiagnosis(
            root_cause=root_cause,
            explanation=explanation,
            recommended_fix=fix,
            confidence=conf,
            sources=clean_sources,
            model_used=model_name,
        )

        # 6. Update memory — store the full answer, not just root_cause
        self.memory.add_user_message(session_id, query)
        summary = f"{diagnosis.root_cause}. {diagnosis.explanation}"
        self.memory.add_ai_message(session_id, summary[:600])

        return diagnosis
