"""Session-based conversation memory for multi-turn K8s diagnosis."""

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


class ChatMemory:
    """In-memory, per-session conversation history.

    Stores the last ``max_messages`` messages for each session so that
    the RAG chain can include recent conversation context in its prompt.
    Sessions are evicted in LRU order when ``max_sessions`` is reached.
    """

    def __init__(
        self,
        max_messages: int = 20,
        max_sessions: int = 256,
    ):
        self.max_messages = max_messages
        self.max_sessions = max_sessions
        # session_id → list[BaseMessage]
        self._store: OrderedDict[str, list[BaseMessage]] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, session_id: str, content: str) -> None:
        """Record a user message."""
        self._ensure_session(session_id)
        self._store[session_id].append(
            HumanMessage(content=content)
        )
        self._trim(session_id)

    def add_ai_message(self, session_id: str, content: str) -> None:
        """Record an AI response."""
        self._ensure_session(session_id)
        self._store[session_id].append(
            AIMessage(content=content)
        )
        self._trim(session_id)

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return conversation history for a session (may be empty)."""
        return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Clear history for a specific session."""
        self._store.pop(session_id, None)
        logger.debug(f"Cleared memory for session {session_id}")

    def clear_all(self) -> None:
        """Clear all sessions."""
        self._store.clear()
        logger.info("Cleared all chat memory sessions")

    @property
    def active_sessions(self) -> int:
        """Number of active sessions."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_session(self, session_id: str) -> None:
        """Create session if needed; move to end for LRU tracking."""
        if session_id not in self._store:
            # Evict oldest session if at capacity
            if len(self._store) >= self.max_sessions:
                evicted_id, _ = self._store.popitem(last=False)
                logger.debug(f"Evicted oldest session: {evicted_id}")
            self._store[session_id] = []
        else:
            # Touch — move to end
            self._store.move_to_end(session_id)

    def _trim(self, session_id: str) -> None:
        """Keep only the last ``max_messages`` messages."""
        messages = self._store.get(session_id)
        if messages and len(messages) > self.max_messages:
            self._store[session_id] = messages[-self.max_messages :]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_memory: Optional[ChatMemory] = None


def get_chat_memory() -> ChatMemory:
    """Get or create the global ChatMemory instance."""
    global _memory
    if _memory is None:
        _memory = ChatMemory()
    return _memory
