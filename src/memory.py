"""Per-session conversation memory.

Stores the last *max_messages* messages per session.
Sessions are evicted LRU when *max_sessions* is reached.

Usage:
    mem = get_chat_memory()
    mem.add_user_message("s1", "Why is my pod crashing?")
    mem.add_ai_message("s1", "The container is OOMKilled …")
    history = mem.get_history("s1")   # [HumanMessage, AIMessage]
"""

import logging
from collections import OrderedDict
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


class ChatMemory:
    """In-memory, per-session conversation history."""

    def __init__(self, max_messages: int = 20, max_sessions: int = 256) -> None:
        self.max_messages = max_messages
        self.max_sessions = max_sessions
        self._store: OrderedDict[str, list[BaseMessage]] = OrderedDict()

    # ── public ────────────────────────────────────────────────────────────
    def add_user_message(self, session_id: str, content: str) -> None:
        self._ensure(session_id)
        self._store[session_id].append(HumanMessage(content=content))
        self._trim(session_id)

    def add_ai_message(self, session_id: str, content: str) -> None:
        self._ensure(session_id)
        self._store[session_id].append(AIMessage(content=content))
        self._trim(session_id)

    def get_history(self, session_id: str) -> list[BaseMessage]:
        return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def clear_all(self) -> None:
        self._store.clear()

    @property
    def active_sessions(self) -> int:
        return len(self._store)

    # ── internals ─────────────────────────────────────────────────────────
    def _ensure(self, sid: str) -> None:
        if sid not in self._store:
            if len(self._store) >= self.max_sessions:
                evicted, _ = self._store.popitem(last=False)
                logger.debug("Evicted session %s", evicted)
            self._store[sid] = []
        else:
            self._store.move_to_end(sid)

    def _trim(self, sid: str) -> None:
        msgs = self._store.get(sid)
        if msgs and len(msgs) > self.max_messages:
            self._store[sid] = msgs[-self.max_messages:]


# ── module-level singleton ────────────────────────────────────────────────
_memory: Optional[ChatMemory] = None


def get_chat_memory() -> ChatMemory:
    global _memory
    if _memory is None:
        _memory = ChatMemory()
    return _memory
