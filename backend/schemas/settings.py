from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


class SettingsStatusResponse(BaseModel):
    deepseek_api_key_configured: bool
    deepseek_base_url: str


class DeepSeekKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=1, max_length=512)
    base_url: str = Field(
        default="https://api.deepseek.com/v1",
        min_length=1,
        max_length=2048,
    )

    @field_validator("api_key")
    @classmethod
    def normalize_api_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("API Key 不能为空")
        return normalized

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url 必须是有效的 HTTP(S) 地址")
        if parsed.username or parsed.password:
            raise ValueError("base_url 不能包含用户名或密码")
        return normalized


class DeepSeekKeyResponse(BaseModel):
    status: str = "saved"
    deepseek_api_key_configured: bool = True
    deepseek_base_url: str
