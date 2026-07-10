from typing import Any

from pydantic import BaseModel


class EvalLatestResponse(BaseModel):
    method: str
    metrics: dict[str, Any]
    report_path: str
