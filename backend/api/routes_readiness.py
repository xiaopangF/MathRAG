from fastapi import APIRouter

from backend.schemas.readiness import ReadinessResponse
from backend.services.readiness_service import readiness_service


router = APIRouter(prefix="/api", tags=["readiness"])


@router.get("/readiness", response_model=ReadinessResponse)
def get_readiness():
    return ReadinessResponse(**readiness_service.inspect())
