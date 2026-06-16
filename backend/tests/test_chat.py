import pytest
from api.chat import ThinkingFilter, _filter_thinking_from_text, _smart_mode


class TestThinkingFilter:
    def test_no_thinking_tag(self):
        f = ThinkingFilter()
        assert f.feed("hello world") == "hello world"

    def test_complete_thinking_tag(self):
        f = ThinkingFilter()
        result = f.feed("<thinking>draft</thinking>real answer")
        assert "draft" not in result
        assert "real answer" in result

    def test_thinking_tag_across_chunks(self):
        f = ThinkingFilter()
        f.feed("<thin")
        result = f.feed("king>draft</thinking>answer")
        assert "draft" not in result
        assert "answer" in result

    def test_flush_after_feed_clears_buffer(self):
        f = ThinkingFilter()
        f.feed("hello")
        assert f.flush() == ""

    def test_flush_inside_thinking_returns_empty(self):
        f = ThinkingFilter()
        f.feed("<thinking>draft")
        assert f.flush() == ""

    def test_multiple_thinking_blocks(self):
        f = ThinkingFilter()
        text = "<thinking>a</thinking>text1<thinking>b</thinking>text2"
        result = f.feed(text)
        assert "a" not in result
        assert "b" not in result
        assert "text1" in result
        assert "text2" in result


class TestFilterThinkingFromText:
    def test_removes_thinking_tags(self):
        text = "<thinking>analysis</thinking>Final answer"
        result = _filter_thinking_from_text(text)
        assert "analysis" not in result
        assert "Final answer" in result

    def test_removes_xml_wrapped_thinking(self):
        text = "```xml\n<thinking>draft</thinking>\n```Final"
        result = _filter_thinking_from_text(text)
        assert "draft" not in result
        assert "Final" in result


class TestSmartMode:
    def test_short_query_returns_simple(self):
        assert _smart_mode("什么是RAG", "agentic") == "simple"

    def test_explicit_simple_stays_simple(self):
        assert _smart_mode("complex question", "simple") == "simple"

    def test_long_complex_query_stays_agentic(self):
        assert _smart_mode(
            "Transformer架构相比RNN有哪些优势？在NLP中的应用现状如何？",
            "agentic"
        ) == "agentic"

    def test_simple_trigger_word(self):
        assert _smart_mode("什么是RAG系统？", "agentic") == "simple"
