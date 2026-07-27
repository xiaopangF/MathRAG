import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.agent.math_agent import INSUFFICIENT_AGENT_ANSWER, MathAgent
from src.agent.math_tools import (
    CalculatorToolArguments,
    ToolInputError,
    calculate_math,
    calculate_math_isolated,
    parse_safe_expression,
)


@dataclass
class FakeChunk:
    content: str = "导数表示函数在一点处的瞬时变化率。"
    rerank_score: float = 0.91
    embedding_score: float = 0.82
    bm25_score: float = 3.5
    fusion_score: float = 0.03
    retrieval_score: float = 0.94
    embedding_rank: int | None = 2
    bm25_rank: int | None = 1
    title: str = "导数的定义"
    chapter: str = "第二章 一元函数微分学"
    section: str = "导数"
    chunk_type: str = "definition"
    file: str = "data/chunks/children/child_0007.txt"
    source_file: str = "高等数学.pdf"
    page_start: int | None = 47
    page_end: int | None = 48
    vector_id: int = 7


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = list(chunks if chunks is not None else [FakeChunk()])
        self.calls = []

    def retrieve(self, query, top_k=3):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class FakeGenerator:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("fake generator ran out of responses")
        return self.responses.pop(0)


def _tool_call(name, arguments, *, call_id="call-1"):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _message(content=None, *, tool_calls=None, reasoning_content=None):
    return SimpleNamespace(
        content=content,
        tool_calls=list(tool_calls or []),
        reasoning_content=reasoning_content,
    )


@pytest.mark.parametrize(
    "payload",
    [
        "x.real",
        "__import__('os').system('whoami')",
        "open('secrets.txt')",
        "import os",
    ],
)
def test_safe_parser_rejects_attributes_and_import_like_payloads(payload):
    with pytest.raises(ToolInputError):
        parse_safe_expression(payload)


@pytest.mark.parametrize(
    ("arguments", "expected_result"),
    [
        (
            CalculatorToolArguments(
                operation="differentiate",
                expression="sin(x) + x^3",
            ),
            "3*x**2 + cos(x)",
        ),
        (
            CalculatorToolArguments(
                operation="integrate",
                expression="x",
                lower_bound="0",
                upper_bound="1",
            ),
            "1/2",
        ),
        (
            CalculatorToolArguments(operation="evaluate", expression="2 + 3*4"),
            "14.0000000000",
        ),
    ],
)
def test_calculate_math_handles_basic_calculus_and_evaluation(
    arguments,
    expected_result,
):
    assert calculate_math(arguments)["result"] == expected_result


def test_calculate_math_isolated_runs_a_simple_calculation():
    result = calculate_math_isolated(
        CalculatorToolArguments(operation="evaluate", expression="6*7"),
        timeout_seconds=8.0,
    )

    assert result["result"] == "42.0000000000"


@pytest.mark.parametrize(
    ("question", "result", "expected"),
    [
        (
            "求 sin(x) + x^3 的导数",
            {
                "operation": "differentiate",
                "expression": "sin(x)+x^3",
                "variable": "x",
            },
            True,
        ),
        (
            "求 sin(x) + x^3 的导数",
            {
                "operation": "evaluate",
                "expression": "sin(x)+x^3",
                "variable": "x",
            },
            False,
        ),
        (
            "计算 x 从 0 到 1 的定积分",
            {
                "operation": "integrate",
                "expression": "x",
                "variable": "x",
                "lower_bound": "0",
                "upper_bound": "1",
            },
            True,
        ),
        (
            "解方程 x^2-1=0",
            {
                "operation": "solve",
                "expression": "x^2-1",
                "variable": "x",
            },
            True,
        ),
    ],
)
def test_calculation_alignment_checks_operation_expression_and_parameters(
    question,
    result,
    expected,
):
    assert MathAgent._calculation_matches_question(question, result) is expected


def test_agent_search_flow_preserves_citation_and_clamps_top_k():
    retriever = FakeRetriever()
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    {"query": "导数定义", "top_k": 5},
                )
            ]
        ),
        _message("导数描述函数的瞬时变化率。[1]"),
        _message("导数描述函数的瞬时变化率。[1]"),
    )
    agent = MathAgent(retriever=retriever, generator=generator)

    result = agent.run("什么是导数？", top_k=2)

    assert retriever.calls == [("导数定义", 2)]
    assert result["answer"] == "导数描述函数的瞬时变化率。[1]"
    assert result["contexts"][0]["vector_id"] == 7
    assert result["contexts"][0]["source_file"] == "高等数学.pdf"
    assert result["agent_steps"] == [
        {
            "tool": "search_textbook",
            "label": "教材检索",
            "status": "success",
            "input": {"query": "导数定义", "top_k": 2},
            "summary": "检索到 1 个教材片段",
        }
    ]
    assert result["confidence"]["is_sufficient"] is True
    assert result["confidence"]["citations_valid"] is True
    assert result["confidence"]["grounding_checked"] is True
    assert len(generator.calls) == 3


def test_agent_skips_repeated_call_and_stops_at_bound():
    retriever = FakeRetriever()
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    '{"query":"导数定义","top_k":5}',
                    call_id="call-1",
                )
            ]
        ),
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    '{ "top_k": 5, "query": "导数定义" }',
                    call_id="call-2",
                )
            ]
        ),
        _message("导数是瞬时变化率。[1]"),
        _message("导数是瞬时变化率。[1]"),
    )
    agent = MathAgent(
        retriever=retriever,
        generator=generator,
        max_tool_calls=2,
    )

    result = agent.run("解释导数的定义", top_k=3)

    assert retriever.calls == [("导数定义", 3)]
    assert len(generator.calls) == 4
    assert [step["status"] for step in result["agent_steps"]] == [
        "success",
        "skipped",
    ]
    assert "重复工具调用" in result["agent_steps"][1]["summary"]
    assert result["confidence"]["is_sufficient"] is True


def test_agent_overrides_wrong_llm_arithmetic_with_calculator_result(monkeypatch):
    captured = []

    def fake_calculate(arguments):
        captured.append(arguments)
        return {
            "operation": arguments.operation,
            "expression": arguments.expression,
            "variable": arguments.variable,
            "result": "14.0000000000",
            "latex": "14.0",
        }

    monkeypatch.setattr("src.agent.math_agent.calculate_math_isolated", fake_calculate)
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "calculate_math",
                    {"operation": "evaluate", "expression": "2+3*4"},
                )
            ]
        ),
        _message("2+3*4 的结果是 20。"),
    )
    agent = MathAgent(retriever=FakeRetriever([]), generator=generator)

    result = agent.run("计算 2+3*4")

    assert captured[0].operation == "evaluate"
    assert result["answer"] == "计算结果（2+3*4）：14"
    assert result["contexts"] == []
    assert result["confidence"]["is_sufficient"] is True
    assert result["confidence"]["reason"] == "agent_tools"
    assert result["confidence"]["calculator_aligned"] is True


def test_agent_refuses_calculator_expression_that_does_not_match_question(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.agent.math_agent.calculate_math_isolated",
        lambda arguments: {
            "operation": arguments.operation,
            "expression": arguments.expression,
            "variable": arguments.variable,
            "result": "42.0000000000",
            "latex": "42.0",
        },
    )
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "calculate_math",
                    {"operation": "evaluate", "expression": "6*7"},
                )
            ]
        ),
        _message("计算结果为 42。"),
    )
    agent = MathAgent(retriever=FakeRetriever([]), generator=generator)

    result = agent.run("计算 2+3*4")

    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["confidence"]["is_sufficient"] is False
    assert result["confidence"]["calculator_aligned"] is False
    assert result["confidence"]["reason"] == "no_evidence"


@pytest.mark.parametrize(
    "question",
    [
        "导数的定义是什么？",
        "What is the product rule?",
        "请解释洛必达法则。",
    ],
)
def test_rule_or_concept_question_cannot_use_irrelevant_calculation_as_evidence(
    monkeypatch,
    question,
):
    monkeypatch.setattr(
        "src.agent.math_agent.calculate_math_isolated",
        lambda arguments: {
            "operation": arguments.operation,
            "expression": arguments.expression,
            "variable": arguments.variable,
            "result": "2*x",
            "latex": "2 x",
        },
    )
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "calculate_math",
                    {"operation": "differentiate", "expression": "x^2"},
                )
            ]
        ),
        _message("导数定义为变化率。"),
    )
    agent = MathAgent(retriever=FakeRetriever([]), generator=generator)

    result = agent.run(question)

    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["confidence"]["is_sufficient"] is False
    assert result["confidence"]["reason"] == "no_evidence"


def test_agent_refuses_cited_context_below_score_threshold():
    low_score_chunk = FakeChunk(rerank_score=0.1)
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    {"query": "导数定义", "top_k": 1},
                )
            ]
        ),
        _message("导数表示函数的瞬时变化率。[1]"),
    )
    agent = MathAgent(
        retriever=FakeRetriever([low_score_chunk]),
        generator=generator,
        min_rerank_score=0.2,
    )

    result = agent.run("什么是导数？")

    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["confidence"]["is_sufficient"] is False
    assert result["confidence"]["citations_valid"] is False
    assert result["confidence"]["grounding_checked"] is False


def test_agent_refuses_in_range_citation_to_unrelated_context_after_grounding():
    unrelated_chunk = FakeChunk(
        content="矩阵乘法要求左矩阵的列数等于右矩阵的行数。",
        title="矩阵乘法",
        chapter="线性代数",
        section="矩阵运算",
    )
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    {"query": "导数定义", "top_k": 1},
                )
            ]
        ),
        _message("导数描述函数的瞬时变化率。[1]"),
        _message("导数描述函数的瞬时变化率。[1]"),
    )
    agent = MathAgent(
        retriever=FakeRetriever([unrelated_chunk]),
        generator=generator,
    )

    result = agent.run("什么是导数？")

    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["confidence"]["is_sufficient"] is False
    assert result["confidence"]["reason"] == "citation_validation"
    assert result["confidence"]["citations_valid"] is False
    assert result["confidence"]["grounding_checked"] is True
    assert len(generator.calls) == 3


@pytest.mark.parametrize(
    "grounded_answer",
    [
        "导数表示函数的瞬时变化率。[1] 火星上存在高等数学教材。[1]",
        "导数表示函数的瞬时变化率。[1] 火星上存在高等数学教材。",
    ],
)
def test_agent_rejects_repeated_or_uncited_unsupported_claims(
    grounded_answer,
):
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    {"query": "导数定义", "top_k": 1},
                )
            ]
        ),
        _message(grounded_answer),
        _message(grounded_answer),
    )
    agent = MathAgent(retriever=FakeRetriever(), generator=generator)

    result = agent.run("什么是导数？")

    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["confidence"]["is_sufficient"] is False
    assert result["confidence"]["reason"] == "citation_validation"
    assert result["confidence"]["citations_valid"] is False


def test_agent_rejects_weak_overlap_and_unsupported_absolute_claim():
    unsupported_answer = "导数在所有点都连续。[1]"
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    {"query": "导数定义", "top_k": 1},
                )
            ]
        ),
        _message(unsupported_answer),
        _message(unsupported_answer),
    )
    agent = MathAgent(retriever=FakeRetriever(), generator=generator)

    result = agent.run("什么是导数？")

    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["confidence"]["is_sufficient"] is False
    assert result["confidence"]["reason"] == "citation_validation"
    assert result["confidence"]["citations_valid"] is False


def test_agent_hides_runtime_tool_failure_details(monkeypatch, caplog):
    leaked_path = r"C:\Users\Lenovo\private\calculator.sock"

    def fail_calculation(_arguments):
        raise OSError(f"cannot open {leaked_path}")

    monkeypatch.setattr(
        "src.agent.math_agent.calculate_math_isolated",
        fail_calculation,
    )
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "calculate_math",
                    {"operation": "evaluate", "expression": "6*7"},
                )
            ]
        ),
        _message("The internal calculator failed."),
    )
    agent = MathAgent(retriever=FakeRetriever([]), generator=generator)

    with caplog.at_level("ERROR", logger="src.agent.math_agent"):
        result = agent.run("计算 6*7")

    serialized_result = json.dumps(result, ensure_ascii=False)
    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["agent_steps"][0]["status"] == "error"
    assert result["agent_steps"][0]["summary"] == "工具暂时不可用，请稍后重试"
    assert leaked_path not in serialized_result
    assert caplog.records[-1].getMessage() == "agent_tool_execution_failed"


def test_agent_repairs_missing_citation():
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    {"query": "导数定义", "top_k": 1},
                )
            ]
        ),
        _message("导数描述瞬时变化率。"),
        _message("导数描述瞬时变化率。[1]"),
    )
    agent = MathAgent(retriever=FakeRetriever(), generator=generator)

    result = agent.run("什么是导数？")

    assert result["answer"] == "导数描述瞬时变化率。[1]"
    assert result["confidence"]["is_sufficient"] is True
    assert result["confidence"]["citation_repaired"] is True
    assert result["confidence"]["citations_valid"] is True
    assert generator.calls[2]["temperature"] == 0.0


def test_agent_refuses_when_citation_repair_is_still_invalid():
    generator = FakeGenerator(
        _message(
            tool_calls=[
                _tool_call(
                    "search_textbook",
                    {"query": "导数定义", "top_k": 1},
                )
            ]
        ),
        _message("导数描述瞬时变化率。[9]"),
        _message("导数描述瞬时变化率，但没有引用。"),
    )
    agent = MathAgent(retriever=FakeRetriever(), generator=generator)

    result = agent.run("什么是导数？")

    assert result["answer"] == INSUFFICIENT_AGENT_ANSWER
    assert result["confidence"]["is_sufficient"] is False
    assert result["confidence"]["reason"] == "citation_validation"
    assert result["confidence"]["citation_repaired"] is False
    assert result["confidence"]["citations_valid"] is False


def test_assistant_payload_preserves_reasoning_content():
    call = _tool_call(
        "search_textbook",
        {"query": "极限定义", "top_k": 2},
    )
    message = _message(
        "",
        tool_calls=[call],
        reasoning_content="internal reasoning token payload",
    )

    payload = MathAgent._assistant_message_payload(message)

    assert payload == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_textbook",
                    "arguments": '{"query": "极限定义", "top_k": 2}',
                },
            }
        ],
        "reasoning_content": "internal reasoning token payload",
    }
