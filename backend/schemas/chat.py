from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    knowledge_base_id: str = Field(default="default")


class ChatResponse(BaseModel):
    query: str
    answer: str
    contexts: list[dict[str, Any]]
    confidence: dict[str, Any] = Field(default_factory=dict)
    knowledge_base_id: str = "default"
