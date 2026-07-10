from pydantic import BaseModel, Field


class SettingsStatusResponse(BaseModel):
    deepseek_api_key_configured: bool
    deepseek_base_url: str


class DeepSeekKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=1)
    base_url: str = "https://api.deepseek.com/v1"


class DeepSeekKeyResponse(BaseModel):
    status: str = "saved"
    deepseek_api_key_configured: bool = True
    deepseek_base_url: str
