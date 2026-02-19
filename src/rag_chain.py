"""RAG diagnosis chain — retrieval + adaptive model selection + generation.

Key features:
  * Multi-model: simple queries → llama3.1, complex → Qwen3-Coder:30b
  * Reranking:   vector search → cross-encoder rerank → top-K context
  * Memory:      includes recent chat history in the prompt
  * Structured:  returns FailureDiagnosis pydantic model
"""

import json
import logging
import os
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
    evidence_snippets: list[str] = Field(default_factory=list)
    response_type: str = Field(default="diagnostic", description="diagnostic|conversational|operational|out_of_scope")
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

_K8S_TERMS = [
    "k8s", "kubernetes", "pod", "pods", "deployment", "replicaset",
    "statefulset", "daemonset", "namespace", "node", "cluster",
    "crashloop", "oomkilled", "imagepull", "ingress", "service",
    "configmap", "secret", "pvc", "pv", "kube-system", "k3d",
]

_CONVERSATION_PATTERNS = [
    r"\bfirst (question|message)\b",
    r"\bwhat did i ask\b",
    r"\bwhat was my (first|last|previous) question\b",
    r"\bhave i asked\b",
    r"\bdid i ask\b",
    r"\bsummary\b",
    r"\bsummarise\b",
    r"\brecap\b",
    r"\bremember\b",
]

_OPERATIONAL_PATTERNS = [
    r"\b(list|show|get)\s+pods\b",
    r"\bpods\s+in\s+k3d\b",
    r"\bk3d\b.*\bpods\b",
    r"\bpods\b.*\bk3d\b",
    r"\bkubectl\b",
]


def _word_in(word: str, text: str) -> bool:
    """Check if a multi-word keyword appears in text (word-boundary-aware)."""
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))


def _term_in(term: str, text: str) -> bool:
    """Word-safe match for short terms (avoids matching 'pod' in 'podcast')."""
    return bool(re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text))


def _fuzzy_term_in(term: str, text: str) -> bool:
    """Allow near-miss matches for longer terms (e.g., crashloopback -> crashloopbackoff)."""
    tokens = re.findall(r"[a-z0-9-]+", text)
    for tok in tokens:
        if len(tok) < 6:
            continue
        if term.startswith(tok) or tok.startswith(term):
            return True
    return False


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
   Still use the JSON format but put your natural answer in "root_cause",
   set "recommended_fix" to "N/A", and do NOT cite sources.
3. If the question is completely outside Kubernetes operations, reply helpfully
   but set confidence to 0.0, recommended_fix to "N/A", and do NOT cite sources.
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

_CLASSIFIER_SYSTEM = """\
You are a router for a Kubernetes diagnosis assistant.

Decide the response type:
- diagnostic: Kubernetes troubleshooting, failure analysis, or follow-up to a Kubernetes issue.
- conversational: Questions about the conversation itself (e.g., what did I ask before?).
- operational: Requests to run commands or list resources (kubectl, list pods, etc.).
- out_of_scope: Not related to Kubernetes or cluster operations.

Use chat history to interpret short follow-ups (e.g., "What else could cause this?").
Return ONLY JSON: {"response_type": "diagnostic|conversational|operational|out_of_scope"}.
"""

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CLASSIFIER_SYSTEM),
    ("human",
     "CHAT HISTORY:\n{history}\n\n"
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
        snippet = d.page_content[:1200]
        parts.append(f"[{i}] {kind} — {src}\n{snippet}")
    return "\n---\n".join(parts)


def _format_history(messages: list[BaseMessage]) -> str:
    if not messages:
        return "(no previous conversation)"
    lines = []
    for m in messages[-10:]:                      # last 5 turns
        role = "User" if m.type == "human" else "AI"
        lines.append(f"{role}: {m.content[:500]}")
    return "\n".join(lines)


def _format_memory_text(dx: FailureDiagnosis) -> str:
    parts = [f"Root Cause: {dx.root_cause}"]
    if dx.explanation:
        parts.append(f"Explanation: {dx.explanation}")
    if dx.recommended_fix and dx.recommended_fix != "N/A":
        parts.append(f"Recommended Fix: {dx.recommended_fix}")
    return "\n".join(parts)


def _classify_query(query: str, history: list[BaseMessage] | None = None) -> str:
    history = history or []
    q = query.lower().strip()

    # Always try LLM classification first (skip in pytest to keep tests offline).
    if "PYTEST_CURRENT_TEST" not in os.environ:
        try:
            llm = ChatOllama(
                model=get_settings().simple_model,
                temperature=0,
                base_url=get_settings().ollama_base_url,
            )
            chain = _CLASSIFIER_PROMPT | llm
            result = chain.invoke({
                "history": _format_history(history),
                "query": query,
            })
            raw = result.content if hasattr(result, "content") else str(result)
            parsed = _parse_json(str(raw))
            response_type = (parsed.get("response_type") or parsed.get("type") or "").strip()
            if response_type in {"diagnostic", "conversational", "operational", "out_of_scope"}:
                return response_type
        except Exception as exc:
            logger.warning("LLM classifier failed, falling back: %s", exc)

    # Fallback heuristics (used when LLM is unavailable)
    if any(re.search(p, q) for p in _CONVERSATION_PATTERNS):
        return "conversational"
    if any(re.search(p, q) for p in _OPERATIONAL_PATTERNS):
        return "operational"
    if any(_term_in(term, q) for term in _K8S_TERMS):
        return "diagnostic"
    if any(_fuzzy_term_in(term, q) for term in _TECHNICAL_KW):
        return "diagnostic"

    if history:
        recent = history[-6:]
        recent_text = " ".join(
            (m.content if isinstance(m.content, str) else str(m.content)).lower()
            for m in recent
        )
        has_k8s_context = any(_term_in(t, recent_text) for t in _K8S_TERMS) or \
                          any(kw in recent_text for kw in _TECHNICAL_KW)
        if has_k8s_context:
            return "diagnostic"

    return "out_of_scope"


def _user_messages(messages: list[BaseMessage]) -> list[str]:
    results: list[str] = []
    for m in messages:
        if m.type != "human":
            continue
        content = m.content
        results.append(content if isinstance(content, str) else str(content))
    return results


def _find_mentions(messages: list[BaseMessage], topic: str) -> list[str]:
    """Return lines where the topic appears in session messages (user+AI).

    Format: "[role #] <snippet>" where role is User/AI and # is 1-based index in session.
    """
    out: list[str] = []
    t = topic.lower()
    for i, m in enumerate(messages, start=1):
        # content may be non-string; coerce safely
        content = m.content if isinstance(m.content, str) else str(m.content)
        if t in content.lower():
            role = "User" if m.type == "human" else "AI"
            snippet = content.strip().replace("\n", " ")[:300]
            out.append(f"[{role} #{i}] {snippet}")
    return out


def _answer_conversational(
    query: str,
    history: list[BaseMessage],
    store: "MilvusStore | None" = None,
) -> tuple[str, str, float, list[str], list[str]]:
    q = query.lower().strip()
    # Defensive: ensure history is a list
    if history is None:
        history = []
    user_msgs = _user_messages(history)
    logger.debug("_answer_conversational: q=%s user_msgs=%d", q, len(user_msgs))

    # first / last
    if re.search(r"\bfirst (question|message)\b", q):
        if user_msgs:
            return f"Your first question was: {user_msgs[0]}", "", 0.9, [], []
        return "I do not have any prior questions in this session.", "", 0.3, [], []

    if re.search(r"\b(last|previous) question\b", q) or "what did i ask" in q:
        if user_msgs:
            return f"Your last question was: {user_msgs[-1]}", "", 0.9, [], []
        return "I do not have any prior questions in this session.", "", 0.3, [], []

    # have I asked you about <topic>
    # Handles: "have i asked you about X", "have i asked about X",
    #          "did i ask you about X", "did i ask about X",
    #          "have i mentioned X", "did i mention X"
    logger.debug("_answer_conversational: checking 'have I asked' pattern")
    m = re.search(
        r"(?:have i|did i)"
        r"\s+(?:ask(?:ed)?|mention(?:ed)?|talk(?:ed)?\s+about|bring\s+up|discuss(?:ed)?)"
        r"\s+(?:you\s+)?(?:about\s+)?(.+)",
        q,
    )
    logger.debug("_answer_conversational: have_i_match=%s", bool(m))
    if m:
        topic = m.group(1).strip("?.! ")
        # Split at first question mark to handle multiple question clauses
        # e.g., "crashloopback? where, when..." -> "crashloopback"
        topic = topic.split("?")[0].strip()
        # strip trailing noise words like 'today' or 'where'
        topic = re.sub(r"\b(today|now|recently|where|when|what|how)\b", "", topic)
        # truncate at trailing conjunction clauses ("and what did you say", "or where", etc.)
        topic = re.split(r"\s+(?:and|or|but|nor)\b", topic)[0]
        # strip any remaining punctuation/whitespace
        topic = re.sub(r"[?.!,]+", "", topic).strip()
        logger.debug("_answer_conversational: extracted topic='%s'", topic)

        mentions = _find_mentions(history, topic)
        logger.debug("_answer_conversational: mentions_found=%d", len(mentions))
        if mentions:
            # If user asks 'where', return the locations (message snippets)
            if "where" in q or (q.endswith("?") and "where" in query.lower()):
                joined = "; ".join(mentions[:5])
                return f"Yes — mentioned in session.", f"{joined}", 0.9, [], mentions
            # Otherwise give a short yes/no plus the first mention
            return f"Yes, you mentioned {topic} earlier.", f"{mentions[0]}", 0.9, [], [mentions[0]]

        # No session mentions — fall back to KB search (reuse existing store if provided)
        docs = []
        try:
            logger.debug("_answer_conversational: falling back to KB search for '%s'", topic)
            _store = store if store is not None else MilvusStore()
            docs = _store.search(topic)
            logger.debug("_answer_conversational: kb_docs_found=%d", len(docs))
        except Exception as e:
            logger.exception("_answer_conversational: KB search error: %s", e)
            docs = []

        if docs:
            sources: list[str] = []
            evidence: list[str] = []
            for d in docs[:4]:
                src = d.metadata.get("source", "?")
                idx = src.find("data/")
                src_clean = src[idx:] if idx != -1 else src
                snippet = d.page_content.strip().replace("\n", " ")[:220]
                sources.append(src_clean)
                evidence.append(f"{src_clean}: {snippet}")
            return (
                f"No prior mentions in session. I found related documents in the KB.",
                "See evidence and sources.",
                0.7,
                sources,
                evidence,
            )

        return f"No, I don't see any prior mentions of {topic} in this session.", "", 0.6, [], []

    return "I can answer questions about our conversation in this session.", "", 0.4, [], []


def _answer_operational(query: str) -> tuple[str, str, str, float]:
    q = query.lower().strip()
    context_hint = ""
    if "k3d" in q:
        context_hint = " If you are using k3d, select the correct context with `kubectl config get-contexts` and `kubectl config use-context <name>`."

    root = "I cannot run cluster commands from here, but you can list pods locally."
    explanation = (
        "Use kubectl to query the cluster." + context_hint
    )
    fix = "Run: kubectl get pods -A (or kubectl get pods -n <namespace>)"
    return root, explanation, fix, 0.2


def _answer_out_of_scope() -> tuple[str, str, float]:
    root = "I specialize in Kubernetes failure diagnosis and cannot help with that topic."
    explanation = "Ask a Kubernetes or cluster troubleshooting question to continue."
    return root, explanation, 0.0


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
        # 1. Classify request type (pass history for context-aware follow-ups)
        history = self.memory.get_history(session_id)
        response_type = _classify_query(query, history)

        if response_type == "conversational":
            root, explanation, conf, sources, evidence = _answer_conversational(query, history, self.store)
            diagnosis = FailureDiagnosis(
                root_cause=root,
                explanation=explanation,
                recommended_fix="N/A",
                confidence=conf,
                sources=sources,
                evidence_snippets=evidence,
                response_type=response_type,
                model_used="rules",
            )
        elif response_type == "operational":
            root, explanation, fix, conf = _answer_operational(query)
            diagnosis = FailureDiagnosis(
                root_cause=root,
                explanation=explanation,
                recommended_fix=fix,
                confidence=conf,
                sources=[],
                evidence_snippets=[],
                response_type=response_type,
                model_used="rules",
            )
        elif response_type == "out_of_scope":
            root, explanation, conf = _answer_out_of_scope()
            diagnosis = FailureDiagnosis(
                root_cause=root,
                explanation=explanation,
                recommended_fix="N/A",
                confidence=conf,
                sources=[],
                evidence_snippets=[],
                response_type=response_type,
                model_used="rules",
            )
        else:
            # 2. Retrieve + rerank
            docs = self.store.search(query)
            logger.info("Retrieved %d docs for query", len(docs))

            # 3. Select model
            model_name = select_model(query, force_model)

            # 4. Build prompt inputs
            context_str = _format_docs(docs)
            history_str = _format_history(history)

            # 5. Call LLM
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

            # 6. Parse output → FailureDiagnosis
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

            # Evidence snippets for UI
            evidence_snippets: list[str] = []
            for d in docs[:3]:
                src = d.metadata.get("source", "?")
                idx = src.find("data/")
                src = src[idx:] if idx != -1 else src
                snippet = d.page_content.strip().replace("\n", " ")[:220]
                evidence_snippets.append(f"{src}: {snippet}")

            diagnosis = FailureDiagnosis(
                root_cause=root_cause,
                explanation=explanation,
                recommended_fix=fix,
                confidence=conf,
                sources=clean_sources,
                evidence_snippets=evidence_snippets,
                response_type=response_type,
                model_used=model_name,
            )

        # 7. Update memory — store a richer summary for follow-ups
        self.memory.add_user_message(session_id, query)
        self.memory.add_ai_message(session_id, _format_memory_text(diagnosis)[:1200])

        return diagnosis

    async def diagnose_stream(
        self,
        query: str,
        session_id: str = "default",
        force_model: str | None = None,
    ):
        """Async generator that yields SSE-formatted strings.

        For non-diagnostic queries (rule-based), yields the full answer at once.
        For diagnostic queries, streams tokens from the LLM then yields metadata.
        """
        import asyncio

        history = self.memory.get_history(session_id)
        response_type = _classify_query(query, history)

        # ── Non-diagnostic paths: yield complete answer instantly ──
        if response_type != "diagnostic":
            if response_type == "conversational":
                root, explanation, conf, sources, evidence = _answer_conversational(
                    query, history, self.store
                )
                diagnosis = FailureDiagnosis(
                    root_cause=root, explanation=explanation,
                    recommended_fix="N/A", confidence=conf,
                    sources=sources, evidence_snippets=evidence,
                    response_type=response_type, model_used="rules",
                )
            elif response_type == "operational":
                root, explanation, fix, conf = _answer_operational(query)
                diagnosis = FailureDiagnosis(
                    root_cause=root, explanation=explanation,
                    recommended_fix=fix, confidence=conf,
                    sources=[], evidence_snippets=[],
                    response_type=response_type, model_used="rules",
                )
            else:
                root, explanation, conf = _answer_out_of_scope()
                diagnosis = FailureDiagnosis(
                    root_cause=root, explanation=explanation,
                    recommended_fix="N/A", confidence=conf,
                    sources=[], evidence_snippets=[],
                    response_type=response_type, model_used="rules",
                )

            self.memory.add_user_message(session_id, query)
            self.memory.add_ai_message(session_id, _format_memory_text(diagnosis)[:1200])

            # Yield the full text as a single token event, then done
            full_text = f"{diagnosis.root_cause}\n\n{diagnosis.explanation}"
            if diagnosis.recommended_fix and diagnosis.recommended_fix != "N/A":
                full_text += f"\n\n**Recommended Fix:**\n{diagnosis.recommended_fix}"
            yield f"data: {json.dumps({'token': full_text})}\n\n"
            yield f"data: {json.dumps({'done': True, 'diagnosis': diagnosis.model_dump()})}\n\n"
            return

        # ── Diagnostic path: stream LLM tokens ──
        docs = await asyncio.to_thread(self.store.search, query)
        logger.info("Retrieved %d docs for streaming query", len(docs))

        model_name = select_model(query, force_model)
        context_str = _format_docs(docs)
        history_str = _format_history(history)

        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=get_settings().ollama_base_url,
        )
        chain = _PROMPT | llm

        raw_text = ""
        async for chunk in chain.astream({
            "context": context_str,
            "history": history_str,
            "query": query,
        }):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                raw_text += str(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        # Parse accumulated text into structured diagnosis
        parsed = _parse_json(str(raw_text))
        root_cause = parsed.get("root_cause") or str(raw_text)[:300]
        explanation = parsed.get("explanation") or ""
        fix = parsed.get("recommended_fix") or ""
        try:
            conf = float(parsed.get("confidence") or 0.5)
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = 0.5

        raw_sources = [d.metadata.get("source", "?") for d in docs]
        clean_sources: list[str] = []
        for s in raw_sources:
            idx = s.find("data/")
            clean_sources.append(s[idx:] if idx != -1 else s)

        evidence_snippets: list[str] = []
        for d in docs[:3]:
            src = d.metadata.get("source", "?")
            idx = src.find("data/")
            src = src[idx:] if idx != -1 else src
            snippet = d.page_content.strip().replace("\n", " ")[:220]
            evidence_snippets.append(f"{src}: {snippet}")

        diagnosis = FailureDiagnosis(
            root_cause=root_cause, explanation=explanation,
            recommended_fix=fix, confidence=conf,
            sources=clean_sources, evidence_snippets=evidence_snippets,
            response_type=response_type, model_used=model_name,
        )

        self.memory.add_user_message(session_id, query)
        self.memory.add_ai_message(session_id, _format_memory_text(diagnosis)[:1200])

        yield f"data: {json.dumps({'done': True, 'diagnosis': diagnosis.model_dump()})}\n\n"
