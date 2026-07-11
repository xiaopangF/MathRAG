import json
from pathlib import Path
from typing import Any

from backend.core.paths import PROJECT_ROOT, REPORTS_DIR


REPORT_BY_METHOD = {
    "hybrid": REPORTS_DIR / "retrieval_metrics_100_hybrid.json",
    "vector_only": REPORTS_DIR / "retrieval_metrics_100_vector_only.json",
    "grounded_dev": REPORTS_DIR / "retrieval_metrics_grounded_dev.json",
    "grounded_sample": REPORTS_DIR / "retrieval_metrics_grounded_sample.json",
}


class EvalService:
    def latest(self, method: str = "hybrid") -> dict[str, Any]:
        report_path = REPORT_BY_METHOD.get(method)
        if report_path is None:
            raise ValueError(f"未知评测方法: {method}")
        if not report_path.exists():
            raise FileNotFoundError(f"评测结果不存在: {report_path}")

        metrics = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "method": method,
            "metrics": metrics,
            "report_path": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }


eval_service = EvalService()
