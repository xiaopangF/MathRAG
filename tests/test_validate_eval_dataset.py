import json

from validate_eval_dataset import validate_eval_dataset


def write_jsonl(path, items):
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )


def grounded_item(item_id, item_type="definition", question=None):
    return {
        "id": item_id,
        "question": question or f"Question {item_id}?",
        "expected_chunk_keywords": ["keyword"],
        "expected_page_ranges": [1],
        "expected_sections": ["Section"],
        "type": item_type,
        "difficulty": "easy",
    }


def test_grounded_smoke_accepts_committed_sample():
    summary = validate_eval_dataset(
        "eval/questions.sample.jsonl",
        profile="grounded-smoke",
    )

    assert summary["errors"] == []
    assert summary["question_count"] == 5
    assert summary["grounded_count"] == 5


def test_keyword_profile_accepts_committed_100_question_dataset():
    summary = validate_eval_dataset(
        "data/eval/questions.jsonl",
        profile="keyword-100",
    )

    assert summary["errors"] == []
    assert summary["question_count"] == 100
    assert summary["grounded_count"] == 0


def test_grounded_profile_rejects_missing_page_and_section_annotations(tmp_path):
    path = tmp_path / "questions.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "q001",
                "question": "What is a derivative?",
                "expected_chunk_keywords": ["derivative"],
                "type": "definition",
            }
            for _ in range(5)
        ],
    )

    summary = validate_eval_dataset(path, profile="grounded-smoke")

    assert any("expected_page_ranges is required" in error for error in summary["errors"])
    assert any("expected_sections is required" in error for error in summary["errors"])


def test_grounded_dev_enforces_category_balance(tmp_path):
    path = tmp_path / "questions.jsonl"
    write_jsonl(path, [grounded_item(f"q{i:03d}") for i in range(1, 31)])

    summary = validate_eval_dataset(path, profile="grounded-dev")

    assert any("theorem questions" in error for error in summary["errors"])
    assert any("cross_section questions" in error for error in summary["errors"])


def test_grounded_dev_accepts_balanced_30_question_dataset(tmp_path):
    path = tmp_path / "questions.jsonl"
    types = (
        ["definition"] * 6
        + ["theorem"] * 6
        + ["formula"] * 4
        + ["method"] * 6
        + ["cross_section"] * 3
        + ["trap"] * 3
        + ["out_of_scope"] * 2
    )
    write_jsonl(
        path,
        [
            grounded_item(f"q{i:03d}", item_type, question=f"Question {i}?")
            for i, item_type in enumerate(types, start=1)
        ],
    )

    summary = validate_eval_dataset(path, profile="grounded-dev")

    assert summary["errors"] == []
    assert summary["question_count"] == 30
    assert summary["grounded_count"] == 30
