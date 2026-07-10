import logging
import os
import sys
import time
from pathlib import Path
from threading import BoundedSemaphore, RLock
from typing import Any

from backend.core.paths import PROJECT_ROOT
from backend.core.settings import get_settings
from backend.services.knowledge_base_service import knowledge_base_service


logger = logging.getLogger(__name__)
runtime_settings = get_settings()


class RAGBusyError(RuntimeError):
    """Raised when all configured inference slots are occupied."""


class RAGService:
    """Lazy, thread-safe wrapper around the existing MathRAG pipeline."""

    def __init__(
        self,
        *,
        max_concurrency: int | None = None,
        acquire_timeout_seconds: float | None = None,
    ):
        self._pipeline = None
        self._retrievers: dict[str, Any] = {}
        self._generator = None
        self._lock = RLock()
        self._max_concurrency = max_concurrency or runtime_settings.rag_max_concurrency
        self._acquire_timeout_seconds = (
            acquire_timeout_seconds
            if acquire_timeout_seconds is not None
            else runtime_settings.rag_acquire_timeout_seconds
        )
        self._inference_slots = BoundedSemaphore(self._max_concurrency)

    def _ensure_project_path(self) -> None:
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    started_at = time.perf_counter()
                    self._ensure_project_path()
                    from src.pipeline.qa_pipeline import MathRAGPipeline
                    from src.retriever.retriever import RAGConfig

                    config = RAGConfig(
                        faiss_index_dir=str(PROJECT_ROOT / "data" / "faiss_index")
                    )
                    self._pipeline = MathRAGPipeline(retriever_config=config)
                    logger.info(
                        "default_pipeline_loaded",
                        extra={
                            "knowledge_base_id": "default",
                            "duration_ms": round(
                                (time.perf_counter() - started_at) * 1000,
                                2,
                            ),
                        },
                    )
        return self._pipeline

    def _get_generator(self):
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    started_at = time.perf_counter()
                    self._ensure_project_path()
                    from src.generation.llm_generator import LLMGenerator

                    self._generator = LLMGenerator(
                        timeout_seconds=runtime_settings.llm_timeout_seconds,
                        max_retries=runtime_settings.llm_max_retries,
                    )
                    logger.info(
                        "llm_generator_loaded",
                        extra={
                            "duration_ms": round(
                                (time.perf_counter() - started_at) * 1000,
                                2,
                            )
                        },
                    )
        return self._generator

    def configure_deepseek(self, api_key: str, base_url: str) -> None:
        with self._lock:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["DEEPSEEK_BASE_URL"] = base_url
            self._generator = None
            if self._pipeline is not None:
                self._pipeline.generator = self._get_generator()
        logger.info("deepseek_runtime_configuration_updated")

    def invalidate_knowledge_base(self, knowledge_base_id: str) -> bool:
        with self._lock:
            removed = self._retrievers.pop(knowledge_base_id, None) is not None
        if removed:
            logger.info(
                "knowledge_base_retriever_invalidated",
                extra={"knowledge_base_id": knowledge_base_id},
            )
        return removed

    def _get_retriever(self, knowledge_base_id: str):
        with self._lock:
            if knowledge_base_id in self._retrievers:
                return self._retrievers[knowledge_base_id]

            kb = knowledge_base_service.get_knowledge_base(knowledge_base_id)
            if not kb:
                raise ValueError(f"知识库不存在: {knowledge_base_id}")
            if kb["status"] != "ready":
                raise ValueError(f"知识库尚未就绪: {knowledge_base_id} ({kb['status']})")

            self._ensure_project_path()
            started_at = time.perf_counter()
            from src.retriever.retriever import MathRAGRetriever, RAGConfig

            config = RAGConfig(faiss_index_dir=kb["index_dir"])
            retriever = MathRAGRetriever(config)
            self._retrievers[knowledge_base_id] = retriever
            logger.info(
                "knowledge_base_retriever_loaded",
                extra={
                    "knowledge_base_id": knowledge_base_id,
                    "duration_ms": round(
                        (time.perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
            )
            return retriever

    @staticmethod
    def _contexts_from_chunks(chunks) -> list[dict[str, Any]]:
        return [
            {
                "content": chunk.content,
                "score": chunk.rerank_score,
                "embedding_score": chunk.embedding_score,
                "bm25_score": chunk.bm25_score,
                "title": chunk.title,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type,
                "file": chunk.file,
                "source_file": chunk.source_file,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "vector_id": chunk.vector_id,
            }
            for chunk in chunks
        ]

    def ask(
        self,
        question: str,
        top_k: int = 3,
        knowledge_base_id: str = "default",
    ) -> dict[str, Any]:
        acquired = self._inference_slots.acquire(
            timeout=self._acquire_timeout_seconds,
        )
        if not acquired:
            logger.warning(
                "rag_capacity_exhausted",
                extra={"knowledge_base_id": knowledge_base_id},
            )
            raise RAGBusyError(
                "问答服务当前繁忙，请稍后重试"
            )
        try:
            return self._ask(
                question=question,
                top_k=top_k,
                knowledge_base_id=knowledge_base_id,
            )
        finally:
            self._inference_slots.release()

    def _ask(
        self,
        question: str,
        top_k: int = 3,
        knowledge_base_id: str = "default",
    ) -> dict[str, Any]:
        if knowledge_base_id == "default":
            result = self._get_pipeline().ask(question, top_k=top_k)
            result["knowledge_base_id"] = knowledge_base_id
            return result

        from src.pipeline.qa_pipeline import DEFAULT_MIN_RERANK_SCORE, INSUFFICIENT_CONTEXT_ANSWER

        retriever = self._get_retriever(knowledge_base_id)
        chunks = retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return {
                "query": question,
                "answer": "未找到相关知识，请检查教材内容。",
                "contexts": [],
                "confidence": {
                    "is_sufficient": False,
                    "top_rerank_score": None,
                    "min_rerank_score": DEFAULT_MIN_RERANK_SCORE,
                    "reason": "no_context",
                },
                "knowledge_base_id": knowledge_base_id,
            }

        contexts = self._contexts_from_chunks(chunks)
        top_rerank_score = max(chunk.rerank_score for chunk in chunks)
        confidence = {
            "is_sufficient": top_rerank_score >= DEFAULT_MIN_RERANK_SCORE,
            "top_rerank_score": top_rerank_score,
            "min_rerank_score": DEFAULT_MIN_RERANK_SCORE,
            "reason": "score_threshold",
        }
        if not confidence["is_sufficient"]:
            answer = INSUFFICIENT_CONTEXT_ANSWER
        else:
            answer = self._get_generator().generate(question, contexts)

        return {
            "query": question,
            "answer": answer,
            "contexts": contexts,
            "confidence": confidence,
            "knowledge_base_id": knowledge_base_id,
        }


rag_service = RAGService()
