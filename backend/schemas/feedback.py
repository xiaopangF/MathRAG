from typing import Any, Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = ""
    rating: Literal["up", "down"]
    reason: str = ""
    comment: str = ""
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    top_rerank_score: float | None = None
    knowledge_base_id: str = "default"


class FeedbackResponse(BaseModel):
    id: int
    status: str = "saved"
