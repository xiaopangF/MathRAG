from fastapi import APIRouter, HTTPException

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.rag_service import rag_service
from src.generation.llm_generator import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMQuotaError,
    LLMRateLimitError,
)


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return rag_service.ask(
            question=request.question,
            top_k=request.top_k,
            knowledge_base_id=request.knowledge_base_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMAuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except LLMQuotaError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc
