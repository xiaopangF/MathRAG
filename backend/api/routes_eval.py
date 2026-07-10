from fastapi import APIRouter, HTTPException, Query

from backend.schemas.eval import EvalLatestResponse
from backend.services.eval_service import eval_service


router = APIRouter(prefix="/api/eval", tags=["evaluation"])


@router.get("/latest", response_model=EvalLatestResponse)
def latest_eval(method: str = Query(default="hybrid", pattern="^(hybrid|vector_only)$")):
    try:
        return eval_service.latest(method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
