import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(default="", max_length=30000)
    rating: Literal["up", "down"]
    reason: str = Field(default="", max_length=500)
    comment: str = Field(default="", max_length=4000)
    contexts: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    top_rerank_score: float | None = None
    knowledge_base_id: str = Field(
        default="default",
        pattern=r"^(default|kb_[0-9a-f]{12})$",
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized

    @field_validator("reason", "comment")
    @classmethod
    def normalize_optional_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("contexts")
    @classmethod
    def limit_context_payload(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload_size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if payload_size > 256 * 1024:
            raise ValueError("反馈上下文不能超过 256 KB")
        return value


class FeedbackResponse(BaseModel):
    id: int
    status: str = "saved"


class FeedbackItem(BaseModel):
    id: int
    knowledge_base_id: str
    question: str
    answer: str
    rating: Literal["up", "down"]
    reason: str = ""
    comment: str = ""
    top_rerank_score: float | None = None
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str


class FeedbackListResponse(BaseModel):
    items: list[FeedbackItem]
    total: int
    limit: int
    offset: int


class FeedbackSummaryResponse(BaseModel):
    total: int
    up_count: int
    down_count: int
    commented_count: int
    average_top_rerank_score: float | None = None
    latest_created_at: str | None = None
    positive_rate: float
