import os
from pathlib import Path
from typing import Any

from backend.core.paths import PROJECT_ROOT


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEEPSEEK_KEY_PLACEHOLDERS = {
    "your_deepseek_api_key_here",
    "your-api-key-here",
    "sk-...",
}


class ReadinessService:
    """Inspect runtime dependencies without loading models or using the network."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root

    @staticmethod
    def _ready(detail: str, configured_value: str | None = None) -> dict[str, Any]:
        return {
            "status": "ready",
            "detail": detail,
            "configured_value": configured_value,
        }

    @staticmethod
    def _missing(detail: str, configured_value: str | None = None) -> dict[str, Any]:
        return {
            "status": "missing",
            "detail": detail,
            "configured_value": configured_value,
        }

    @staticmethod
    def _download_required(detail: str, configured_value: str) -> dict[str, Any]:
        return {
            "status": "download_required",
            "detail": detail,
            "configured_value": configured_value,
        }

    def _check_default_index(self) -> dict[str, Any]:
        index_dir = self.project_root / "data" / "faiss_index"
        required_files = [index_dir / "index.faiss", index_dir / "chunks_meta.jsonl"]
        missing_files = [path.name for path in required_files if not path.is_file()]
        if missing_files:
            return self._missing(
                f"默认知识库缺少文件: {', '.join(missing_files)}",
                str(index_dir),
            )

        empty_files = [path.name for path in required_files if path.stat().st_size == 0]
        if empty_files:
            return self._missing(
                f"默认知识库文件为空: {', '.join(empty_files)}",
                str(index_dir),
            )

        return self._ready("默认知识库索引和元数据均可用", str(index_dir))

    @staticmethod
    def _looks_like_local_path(model_name: str) -> bool:
        return (
            Path(model_name).is_absolute()
            or model_name.startswith((".", "~"))
            or "\\" in model_name
        )

    @staticmethod
    def _huggingface_cache_roots() -> list[Path]:
        roots: list[Path] = []
        if os.getenv("HF_HUB_CACHE"):
            roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
        if os.getenv("HF_HOME"):
            roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
        if os.getenv("TRANSFORMERS_CACHE"):
            roots.append(Path(os.environ["TRANSFORMERS_CACHE"]).expanduser())
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        return list(dict.fromkeys(roots))

    @staticmethod
    def _sentence_transformers_cache_roots() -> list[Path]:
        roots: list[Path] = []
        if os.getenv("SENTENCE_TRANSFORMERS_HOME"):
            roots.append(Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]).expanduser())
        roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")
        return list(dict.fromkeys(roots))

    def _is_remote_model_cached(self, model_name: str) -> bool:
        def has_model_files(model_dir: Path) -> bool:
            if not (model_dir / "config.json").is_file():
                return False
            return any(model_dir.glob("*.safetensors")) or any(
                (model_dir / filename).is_file()
                for filename in ("pytorch_model.bin", "model.onnx")
            )

        repo_cache_name = f"models--{model_name.replace('/', '--')}"
        for cache_root in self._huggingface_cache_roots():
            snapshots_dir = cache_root / repo_cache_name / "snapshots"
            if snapshots_dir.is_dir() and any(
                path.is_dir() and has_model_files(path)
                for path in snapshots_dir.iterdir()
            ):
                return True

        legacy_cache_name = model_name.replace("/", "_")
        return any(
            has_model_files(cache_root / legacy_cache_name)
            for cache_root in self._sentence_transformers_cache_roots()
        )

    def _check_model(self, model_name: str, label: str) -> dict[str, Any]:
        expanded_path = Path(model_name).expanduser()
        if expanded_path.exists():
            if expanded_path.is_dir():
                return self._ready(f"{label} 本地目录可用", str(expanded_path.resolve()))
            return self._missing(f"{label} 配置必须指向模型目录", str(expanded_path))

        if self._looks_like_local_path(model_name):
            return self._missing(f"{label} 本地目录不存在", str(expanded_path))

        if self._is_remote_model_cached(model_name):
            return self._ready(f"{label} 已存在于本机缓存", model_name)

        return self._download_required(
            f"{label} 尚未缓存，首次使用需要可访问 HuggingFace 或镜像站",
            model_name,
        )

    def inspect(self) -> dict[str, Any]:
        embedding_model = os.getenv("MATHRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        reranker_model = os.getenv("MATHRAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        api_key_configured = bool(api_key) and api_key.lower() not in DEEPSEEK_KEY_PLACEHOLDERS

        checks = {
            "default_index": self._check_default_index(),
            "embedding_model": self._check_model(embedding_model, "Embedding 模型"),
            "reranker_model": self._check_model(reranker_model, "Reranker 模型"),
            "deepseek_api_key": (
                self._ready("DeepSeek API Key 已配置")
                if api_key_configured
                else self._missing("未配置有效的 DEEPSEEK_API_KEY")
            ),
        }

        blockers = [
            check["detail"]
            for check in checks.values()
            if check["status"] != "ready"
        ]
        all_ready = not blockers
        has_hard_blocker = any(check["status"] == "missing" for check in checks.values())

        return {
            "status": "ready" if all_ready else ("blocked" if has_hard_blocker else "degraded"),
            "can_answer_default": all_ready,
            "can_build_index": checks["embedding_model"]["status"] == "ready",
            "checks": checks,
            "blockers": blockers,
        }


readiness_service = ReadinessService()
