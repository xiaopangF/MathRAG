"""Download configured SentenceTransformers models into the local cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _model_setting(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} is empty.")
    return value


def _is_local_model(model_name: str) -> bool:
    path = Path(model_name).expanduser()
    return path.exists() or path.is_absolute() or "\\" in model_name or model_name.startswith((".", "~"))


def main() -> int:
    hf_endpoint = os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    embedding_model = _model_setting("MATHRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    reranker_model = _model_setting("MATHRAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)

    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"MATHRAG_EMBEDDING_MODEL={embedding_model}")
    print(f"MATHRAG_RERANKER_MODEL={reranker_model}")

    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        print("Loading embedding model...")
        SentenceTransformer(embedding_model)

        print("Loading reranker model...")
        CrossEncoder(reranker_model)
    except Exception as exc:
        print(f"Model prewarm failed: {exc}", file=sys.stderr)
        if _is_local_model(embedding_model) or _is_local_model(reranker_model):
            print(
                "If you use local model paths in Docker, mount the host directory "
                "and configure container paths instead of Windows paths.",
                file=sys.stderr,
            )
        return 1

    print("Model cache is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
