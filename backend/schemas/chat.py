from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)
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


class ChatResponse(BaseModel):
    query: str
    answer: str
    contexts: list[dict[str, Any]]
    confidence: dict[str, Any] = Field(default_factory=dict)
    knowledge_base_id: str = "default"
