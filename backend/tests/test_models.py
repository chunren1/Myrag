import pytest
from pydantic import ValidationError
from api.chat import ChatRequest


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(query="test question", stream=True, mode="agentic")
        assert req.query == "test question"
        assert req.stream is True
        assert req.mode == "agentic"

    def test_defaults(self):
        req = ChatRequest(query="test")
        assert req.stream is True
        assert req.mode == "agentic"

    def test_empty_query_fails(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="", stream=True)

    def test_invalid_mode_fails(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="test", mode="invalid")

    def test_simple_mode(self):
        req = ChatRequest(query="test", mode="simple")
        assert req.mode == "simple"

    def test_max_length_query(self):
        long_query = "x" * 5000
        req = ChatRequest(query=long_query)
        assert len(req.query) == 5000

    def test_over_max_length_fails(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="x" * 5001)
