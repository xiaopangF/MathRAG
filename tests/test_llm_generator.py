from types import SimpleNamespace

import pytest

from src.generation import llm_generator as llm_module
from src.generation.llm_generator import LLMGenerationError, LLMGenerator


def test_normalize_context_supports_structured_context():
    context = {
        "content": "导数表示函数在某一点的变化率。",
        "score": 0.91,
        "title": "导数的定义",
        "chapter": "第二章 一元函数微分学",
        "section": "导数",
        "chunk_type": "definition",
        "source_file": "高等数学.pdf",
        "page_start": 47,
        "page_end": 48,
    }

    result = LLMGenerator._normalize_context(context, 1)

    assert result == {
        "index": 1,
        "content": "导数表示函数在某一点的变化率。",
        "score": 0.91,
        "title": "导数的定义",
        "chapter": "第二章 一元函数微分学",
        "section": "导数",
        "chunk_type": "definition",
        "source_file": "高等数学.pdf",
        "page_start": 47,
        "page_end": 48,
    }


def test_normalize_context_supports_legacy_tuple_context():
    result = LLMGenerator._normalize_context(("导数是变化率。", 0.8), 2)

    assert result == {
        "index": 2,
        "content": "导数是变化率。",
        "score": 0.8,
        "title": "",
        "chapter": "",
        "section": "",
        "chunk_type": "",
        "source_file": "",
        "page_start": None,
        "page_end": None,
    }


def test_format_page_range_supports_single_page_and_ranges():
    assert LLMGenerator._format_page_range(47, 47) == "47"
    assert LLMGenerator._format_page_range(47, 48) == "47-48"
    assert LLMGenerator._format_page_range(None, 48) == ""


def test_llm_client_uses_explicit_timeout_and_retries(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "system-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("MATHRAG_LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("MATHRAG_LLM_THINKING_ENABLED", "true")
    monkeypatch.setattr(llm_module, "OpenAI", fake_openai)

    generator = LLMGenerator(timeout_seconds=12.5, max_retries=4)

    assert generator.api_key == "system-key"
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 4
    assert generator.model == "deepseek-v4-pro"
    assert generator.thinking_enabled is True


def test_llm_runtime_limits_are_validated(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MATHRAG_LLM_TIMEOUT_SECONDS", "0")

    try:
        LLMGenerator()
    except ValueError as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("invalid timeout should fail")


def test_complete_chat_passes_model_thinking_and_tool_options():
    captured = {}
    expected_message = SimpleNamespace(content=None, tool_calls=[object()])

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=expected_message,
                    )
                ]
            )

    generator = object.__new__(LLMGenerator)
    generator.model = "deepseek-v4-flash"
    generator.thinking_enabled = False
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    result = generator.complete_chat(
        messages=[{"role": "user", "content": "计算 1+1"}],
        tools=[{"type": "function", "function": {"name": "calculate_math"}}],
        tool_choice="required",
    )

    assert result is expected_message
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["tool_choice"] == "required"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_complete_chat_rejects_truncated_responses():
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="partial"),
                    )
                ]
            )

    generator = object.__new__(LLMGenerator)
    generator.model = "deepseek-v4-flash"
    generator.thinking_enabled = False
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    with pytest.raises(LLMGenerationError, match="未能完整生成"):
        generator.complete_chat(messages=[{"role": "user", "content": "test"}])
