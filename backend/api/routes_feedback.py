from typing import Literal

from fastapi import APIRouter, Query

from backend.schemas.feedback import (
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummaryResponse,
)
from backend.services.feedback_service import feedback_service


router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def save_feedback(request: FeedbackRequest):
    feedback_id = feedback_service.save(request.model_dump())
    return FeedbackResponse(id=feedback_id)


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def summarize_feedback(
    knowledge_base_id: str | None = Query(
        default=None,
        pattern=r"^(default|kb_[0-9a-f]{12})$",
    ),
):
    return FeedbackSummaryResponse(
        **feedback_service.summarize_feedback(knowledge_base_id=knowledge_base_id)
    )


@router.get("/feedback", response_model=FeedbackListResponse)
def list_feedback(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    rating: Literal["up", "down"] | None = None,
    knowledge_base_id: str | None = Query(
        default=None,
        pattern=r"^(default|kb_[0-9a-f]{12})$",
    ),
):
    return FeedbackListResponse(
        **feedback_service.list_feedback(
            limit=limit,
            offset=offset,
            rating=rating,
            knowledge_base_id=knowledge_base_id,
        )
    )
