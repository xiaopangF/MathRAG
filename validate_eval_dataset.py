"""
Validate MathRAG retrieval evaluation datasets.

This script checks JSONL structure, unique IDs, required annotations, and the
minimum coverage expected by each evaluation tier. It does not run retrieval.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluate_retrieval import (  # noqa: E402
    get_expected_keywords,
    get_expected_page_ranges,
    get_expected_sections,
)


TYPE_FAMILIES = {
    "concept": {"concept", "definition", "property"},
    "theorem": {"theorem"},
    "formula": {"formula"},
    "method": {"method", "application"},
    "cross_section": {"cross_section", "multi_hop"},
    "trap": {"trap", "counterexample", "edge_case"},
    "out_of_scope": {"out_of_scope", "unanswerable"},
}

DIFFICULTIES = {"easy", "medium", "hard"}

PROFILE_RULES = {
    "keyword-100": {
        "min_questions": 100,
        "require_grounded": False,
        "family_min_counts": {},
    },
    "grounded-smoke": {
        "min_questions": 5,
        "require_grounded": True,
        "family_min_counts": {},
    },
    "grounded-dev": {
        "min_questions": 30,
        "require_grounded": True,
        "family_min_counts": {
            "concept": 6,
            "theorem": 6,
            "formula": 4,
            "method": 6,
            "cross_section": 3,
            "trap": 3,
            "out_of_scope": 2,
        },
    },
    "grounded-locked": {
        "min_questions": 100,
        "require_grounded": True,
        "family_min_counts": {
            "concept": 20,
            "theorem": 20,
            "formula": 15,
            "method": 20,
            "cross_section": 10,
            "trap": 10,
            "out_of_scope": 5,
        },
    },
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    items: list[dict[str, Any]] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{dataset_path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{dataset_path}:{line_no} must contain a JSON object")
            item["_line_no"] = line_no
            items.append(item)

    return items


def type_family(item_type: str) -> str:
    normalized = item_type.strip()
    for family, aliases in TYPE_FAMILIES.items():
        if normalized in aliases:
            return family
    return "unknown"


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_answerable(item: dict[str, Any]) -> bool:
    """Return whether the item is expected to have supporting textbook evidence."""
    return item.get("expected_answerable", True) is not False


def validate_eval_dataset(
    path: str | Path,
    profile: str = "grounded-smoke",
) -> dict[str, Any]:
    if profile not in PROFILE_RULES:
        raise ValueError(f"unknown profile: {profile}")

    rules = PROFILE_RULES[profile]
    items = load_jsonl(path)
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_questions: dict[str, int] = {}
    type_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    answerable_count = 0
    grounded_count = 0

    min_questions = int(rules["min_questions"])
    if len(items) < min_questions:
        errors.append(
            f"dataset has {len(items)} questions; profile {profile} requires at least {min_questions}"
        )

    for item in items:
        line_no = int(item["_line_no"])
        item_id = item.get("id")
        question = item.get("question")

        if not _is_nonempty_string(item_id):
            errors.append(f"line {line_no}: id must be a non-empty string")
        elif item_id in seen_ids:
            errors.append(
                f"line {line_no}: duplicate id {item_id!r}; first seen on line {seen_ids[item_id]}"
            )
        else:
            seen_ids[item_id] = line_no

        if not _is_nonempty_string(question):
            errors.append(f"line {line_no}: question must be a non-empty string")
        else:
            normalized_question = question.strip()
            if normalized_question in seen_questions:
                errors.append(
                    f"line {line_no}: duplicate question; first seen on line {seen_questions[normalized_question]}"
                )
            else:
                seen_questions[normalized_question] = line_no

        answerable = is_answerable(item)
        if answerable:
            answerable_count += 1

        expected_keywords = get_expected_keywords(item)
        if answerable and not expected_keywords:
            errors.append(
                f"line {line_no}: expected_chunk_keywords or expected_keywords is required"
            )

        item_type = item.get("type")
        if not _is_nonempty_string(item_type):
            errors.append(f"line {line_no}: type must be a non-empty string")
            family = "unknown"
        else:
            type_counts[item_type.strip()] += 1
            family = type_family(item_type)
            family_counts[family] += 1
            if family == "unknown":
                errors.append(f"line {line_no}: unknown type {item_type!r}")
            if family == "out_of_scope" and answerable:
                errors.append(
                    f"line {line_no}: out_of_scope items must set expected_answerable to false"
                )
            if family != "out_of_scope" and not answerable:
                errors.append(
                    f"line {line_no}: only out_of_scope items may set expected_answerable to false"
                )

        difficulty = item.get("difficulty")
        if difficulty is not None:
            if not _is_nonempty_string(difficulty) or difficulty.strip() not in DIFFICULTIES:
                errors.append(f"line {line_no}: difficulty must be one of {sorted(DIFFICULTIES)}")
            else:
                difficulty_counts[difficulty.strip()] += 1

        try:
            page_ranges = get_expected_page_ranges(item)
            sections = get_expected_sections(item)
        except ValueError as exc:
            errors.append(f"line {line_no}: {exc}")
            page_ranges = []
            sections = []

        if answerable and page_ranges and sections:
            grounded_count += 1

        if answerable and rules["require_grounded"]:
            if not page_ranges:
                errors.append(f"line {line_no}: expected_page_ranges is required")
            if not sections:
                errors.append(f"line {line_no}: expected_sections is required")

    for family, min_count in rules["family_min_counts"].items():
        actual = family_counts[family]
        if actual < min_count:
            errors.append(
                f"profile {profile} requires at least {min_count} {family} questions; found {actual}"
            )

    return {
        "path": str(path),
        "profile": profile,
        "question_count": len(items),
        "answerable_count": answerable_count,
        "grounded_count": grounded_count,
        "type_counts": dict(sorted(type_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "errors": errors,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['path']}")
    print(f"Profile: {summary['profile']}")
    print(f"Questions: {summary['question_count']}")
    print(f"Answerable: {summary['answerable_count']}")
    print(f"Grounded: {summary['grounded_count']}")
    print(f"Type families: {summary['family_counts']}")

    errors = summary["errors"]
    if not errors:
        print("Validation: PASS")
        return

    print("Validation: FAIL")
    for error in errors:
        print(f"- {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MathRAG evaluation datasets.")
    parser.add_argument("--eval-path", default="eval/questions.sample.jsonl")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_RULES),
        default="grounded-smoke",
        help="Validation profile to apply.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = validate_eval_dataset(args.eval_path, profile=args.profile)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Validation failed: {exc}")
        return 1

    print_summary(summary)
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
