from fastapi import APIRouter

from backend.schemas.feedback import FeedbackRequest, FeedbackResponse
from backend.services.feedback_service import feedback_service


router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def save_feedback(request: FeedbackRequest):
    feedback_id = feedback_service.save(request.model_dump())
    return FeedbackResponse(id=feedback_id)
