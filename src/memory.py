"""Per-session conversation memory.

Stores the last *max_messages* messages per session.
Sessions are evicted LRU when *max_sessions* is reached.

Usage:
    mem = get_chat_memory()
    mem.add_user_message("s1", "Why is my pod crashing?")
    mem.add_ai_message("s1", "The container is OOMKilled …")
    history = mem.get_history("s1")   # [HumanMessage, AIMessage]
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import get_settings

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


class DiskChatMemory(ChatMemory):
    """Disk-backed memory with JSON persistence."""

    def __init__(self, path: Path, max_messages: int = 20, max_sessions: int = 256) -> None:
        super().__init__(max_messages=max_messages, max_sessions=max_sessions)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def add_user_message(self, session_id: str, content: str) -> None:
        super().add_user_message(session_id, content)
        self._save()

    def add_ai_message(self, session_id: str, content: str) -> None:
        super().add_ai_message(session_id, content)
        self._save()

    def clear(self, session_id: str) -> None:
        super().clear(session_id)
        self._save()

    def clear_all(self) -> None:
        super().clear_all()
        self._save()

    def _save(self) -> None:
        data: dict[str, list[dict[str, str]]] = {}
        for sid, msgs in self._store.items():
            data[sid] = [
                {"type": m.type, "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in msgs
            ]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.path)
        except Exception as exc:
            logger.warning("Failed to persist memory: %s", exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception as exc:
            logger.warning("Failed to load memory: %s", exc)
            return
        if not isinstance(raw, dict):
            return

        for sid, msgs in raw.items():
            if not isinstance(msgs, list):
                continue
            restored: list[BaseMessage] = []
            for item in msgs:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "")
                msg_type = item.get("type")
                if msg_type == "human":
                    restored.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    restored.append(AIMessage(content=content))
            if restored:
                self._store[sid] = restored[-self.max_messages:]


# ── module-level singleton ────────────────────────────────────────────────
_memory: Optional[ChatMemory] = None


def get_chat_memory() -> ChatMemory:
    global _memory
    if _memory is None:
        settings = get_settings()
        base = Path(__file__).resolve().parents[1]
        path = Path(settings.memory_path)
        if not path.is_absolute():
            path = base / path
        _memory = DiskChatMemory(
            path=path,
            max_messages=settings.memory_max_messages,
            max_sessions=settings.memory_max_sessions,
        )
    return _memory
