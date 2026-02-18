"""Tests for src.memory.chat_memory — session-based conversation memory."""

from src.memory.chat_memory import ChatMemory
from langchain_core.messages import HumanMessage, AIMessage


class TestChatMemory:
    def setup_method(self):
        self.mem = ChatMemory(max_messages=5, max_sessions=3)

    def test_empty_history(self):
        assert self.mem.get_history("nonexistent") == []

    def test_add_and_retrieve(self):
        self.mem.add_user_message("s1", "hello")
        self.mem.add_ai_message("s1", "hi there")
        history = self.mem.get_history("s1")
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert history[0].content == "hello"

    def test_max_messages_trim(self):
        for i in range(10):
            self.mem.add_user_message("s1", f"msg-{i}")
        history = self.mem.get_history("s1")
        assert len(history) == 5  # max_messages
        assert history[0].content == "msg-5"  # Oldest kept

    def test_max_sessions_eviction(self):
        self.mem.add_user_message("s1", "a")
        self.mem.add_user_message("s2", "b")
        self.mem.add_user_message("s3", "c")
        # Adding a 4th session should evict s1 (oldest)
        self.mem.add_user_message("s4", "d")
        assert self.mem.get_history("s1") == []
        assert len(self.mem.get_history("s4")) == 1

    def test_clear_session(self):
        self.mem.add_user_message("s1", "test")
        self.mem.clear("s1")
        assert self.mem.get_history("s1") == []

    def test_clear_all(self):
        self.mem.add_user_message("s1", "a")
        self.mem.add_user_message("s2", "b")
        self.mem.clear_all()
        assert self.mem.active_sessions == 0

    def test_active_sessions_count(self):
        assert self.mem.active_sessions == 0
        self.mem.add_user_message("s1", "a")
        assert self.mem.active_sessions == 1
        self.mem.add_user_message("s2", "b")
        assert self.mem.active_sessions == 2
