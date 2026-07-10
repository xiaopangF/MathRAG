from typing import Literal

from pydantic import BaseModel, Field


ReadinessState = Literal["ready", "missing", "download_required"]
OverallReadinessState = Literal["ready", "degraded", "blocked"]


class ReadinessCheck(BaseModel):
    status: ReadinessState
    detail: str
    configured_value: str | None = None


class ReadinessResponse(BaseModel):
    status: OverallReadinessState
    can_answer_default: bool
    can_build_index: bool
    checks: dict[str, ReadinessCheck]
    blockers: list[str] = Field(default_factory=list)
