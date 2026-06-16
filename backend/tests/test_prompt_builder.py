from core.prompt_builder import build_generator_prompt


class TestPromptBuilder:
    def test_basic_prompt(self):
        docs = [
            {
                "content": "test content",
                "score": 0.9,
                "metadata": {"title_path": "Test Title"},
            }
        ]
        result = build_generator_prompt(query="test query", retrieved_docs=docs)
        assert "system" in result
        assert "user" in result
        assert "test query" in result["user"]
        assert "test content" in result["system"]

    def test_no_docs(self):
        result = build_generator_prompt(query="test", retrieved_docs=[])
        assert "未检索到任何参考资料" in result["system"]

    def test_with_reflection_logs(self):
        docs = [{"content": "c", "score": 0.5, "metadata": {"title_path": "t"}}]
        logs = [{"round": 1, "is_sufficient": False, "supplementary_query": "extra"}]
        result = build_generator_prompt(query="q", retrieved_docs=docs, reflection_logs=logs)
        assert "检索质量说明" in result["system"]

    def test_context_truncation(self):
        long_docs = [
            {
                "content": "x" * 5000,
                "score": 0.9,
                "metadata": {"title_path": "long"},
            }
            for _ in range(5)
        ]
        result = build_generator_prompt(query="q", retrieved_docs=long_docs, max_context_chars=2000)
        assert len(result["system"]) < 5000
