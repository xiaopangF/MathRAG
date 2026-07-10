import os

from fastapi import APIRouter

from backend.schemas.settings import (
    DeepSeekKeyRequest,
    DeepSeekKeyResponse,
    SettingsStatusResponse,
)
from backend.services.rag_service import rag_service


router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsStatusResponse)
def get_settings():
    return SettingsStatusResponse(
        deepseek_api_key_configured=bool(os.environ.get("DEEPSEEK_API_KEY")),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


@router.post("/deepseek-key", response_model=DeepSeekKeyResponse)
def set_deepseek_key(request: DeepSeekKeyRequest):
    rag_service.configure_deepseek(
        api_key=request.api_key.strip(),
        base_url=request.base_url.strip() or "https://api.deepseek.com/v1",
    )
    return DeepSeekKeyResponse(
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
