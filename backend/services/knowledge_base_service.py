import logging
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import UploadFile

from backend.core.paths import BACKEND_DB_PATH, DOCUMENTS_DIR, INDEXES_DIR
from backend.core.settings import get_settings


logger = logging.getLogger(__name__)
runtime_settings = get_settings()
MAX_UPLOAD_BYTES = runtime_settings.max_upload_bytes
ALLOWED_PDF_CONTENT_TYPES = {
    "",
    "application/pdf",
    "application/octet-stream",
    "binary/octet-stream",
}


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ResourceConflictError(ValueError):
    """Raised when a valid resource cannot accept the requested operation."""


TERMINAL_JOB_STATUSES = {JobStatus.SUCCESS, JobStatus.FAILED}
ALLOWED_JOB_TRANSITIONS = {
    JobStatus.PENDING: {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.FAILED},
    JobStatus.RUNNING: {JobStatus.RUNNING, JobStatus.SUCCESS, JobStatus.FAILED},
    JobStatus.SUCCESS: {JobStatus.SUCCESS},
    JobStatus.FAILED: {JobStatus.FAILED, JobStatus.PENDING},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "document.pdf"
    safe = "".join(char if char.isalnum() or char in "._-()[] " else "_" for char in name)
    return safe[:160] or "document.pdf"


class KnowledgeBaseService:
    def __init__(
        self,
        db_path: Path = BACKEND_DB_PATH,
        *,
        sqlite_timeout_seconds: float | None = None,
        job_max_attempts: int | None = None,
    ):
        self.db_path = db_path
        self.sqlite_timeout_seconds = (
            sqlite_timeout_seconds
            if sqlite_timeout_seconds is not None
            else runtime_settings.sqlite_timeout_seconds
        )
        self.job_max_attempts = (
            job_max_attempts
            if job_max_attempts is not None
            else runtime_settings.job_max_attempts
        )
        self._db_lock = RLock()
        self._db_initialized = False

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.sqlite_timeout_seconds,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            f"PRAGMA busy_timeout = {int(self.sqlite_timeout_seconds * 1000)}"
        )
        return conn

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def ensure_db(self) -> None:
        if self._db_initialized:
            return
        with self._db_lock:
            if self._db_initialized:
                return
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
                )
                conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    knowledge_base_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    index_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
                )
                conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT ''
                )
                """
                )
                self._ensure_column(
                    conn,
                    "index_jobs",
                    "attempt_count",
                    "INTEGER NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    conn,
                    "index_jobs",
                    "started_at",
                    "TEXT NOT NULL DEFAULT ''",
                )
                self._ensure_column(
                    conn,
                    "index_jobs",
                    "finished_at",
                    "TEXT NOT NULL DEFAULT ''",
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_index_jobs_status
                    ON index_jobs(status, updated_at)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_index_jobs_document_status
                    ON index_jobs(document_id, status, created_at)
                    """
                )
            self._db_initialized = True

    def save_document(self, upload: UploadFile, content: bytes) -> dict[str, Any]:
        self.ensure_db()
        filename = sanitize_filename(upload.filename or "")
        content_type = (upload.content_type or "").lower()
        if not filename.lower().endswith(".pdf"):
            raise ValueError("只支持上传 PDF 文件")
        if content_type not in ALLOWED_PDF_CONTENT_TYPES:
            raise ValueError(f"不支持的文件类型: {upload.content_type}")
        if len(content) > runtime_settings.max_upload_bytes:
            raise ValueError(
                f"PDF 文件过大，上限为 {runtime_settings.max_upload_bytes // 1024 // 1024} MB"
            )
        if not content.startswith(b"%PDF-"):
            raise ValueError("文件内容不是有效 PDF")

        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        storage_path = DOCUMENTS_DIR / f"{document_id}.pdf"
        storage_path.write_bytes(content)

        record = {
            "document_id": document_id,
            "filename": filename,
            "content_type": content_type or "application/pdf",
            "storage_path": str(storage_path),
            "size_bytes": len(content),
            "status": "uploaded",
            "created_at": utc_now(),
        }
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, filename, content_type, storage_path,
                        size_bytes, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["document_id"],
                        record["filename"],
                        record["content_type"],
                        record["storage_path"],
                        record["size_bytes"],
                        record["status"],
                        record["created_at"],
                    ),
                )
        except Exception:
            storage_path.unlink(missing_ok=True)
            logger.exception(
                "document_persistence_failed",
                extra={"document_id": document_id},
            )
            raise
        return record

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_index_job(self, document_id: str) -> dict[str, Any]:
        self.ensure_db()
        document = self.get_document(document_id)
        if not document:
            raise ValueError(f"文档不存在: {document_id}")

        now = utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active_job = conn.execute(
                """
                SELECT job_id, document_id, knowledge_base_id, status
                FROM index_jobs
                WHERE document_id = ? AND status IN ('pending', 'running')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
            if active_job:
                logger.info(
                    "index_job_reused",
                    extra={
                        "job_id": active_job["job_id"],
                        "document_id": document_id,
                        "knowledge_base_id": active_job["knowledge_base_id"],
                    },
                )
                return {
                    "job_id": active_job["job_id"],
                    "document_id": active_job["document_id"],
                    "knowledge_base_id": active_job["knowledge_base_id"],
                    "status": active_job["status"],
                    "reused": True,
                }

            knowledge_base_id = f"kb_{uuid.uuid4().hex[:12]}"
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            index_dir = INDEXES_DIR / knowledge_base_id
            conn.execute(
                """
                INSERT INTO knowledge_bases (
                    knowledge_base_id, document_id, name, index_dir,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    knowledge_base_id,
                    document_id,
                    document["filename"],
                    str(index_dir),
                    "building",
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO index_jobs (
                    job_id, document_id, knowledge_base_id, status, progress,
                    message, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    document_id,
                    knowledge_base_id,
                    "pending",
                    0,
                    "等待构建",
                    "",
                    now,
                    now,
                ),
            )

        return {
            "job_id": job_id,
            "document_id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "status": "pending",
            "reused": False,
        }

    def claim_job(self, job_id: str) -> dict[str, Any] | None:
        self.ensure_db()
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, progress = ?, message = ?, error = ?,
                    attempt_count = attempt_count + 1,
                    started_at = ?, finished_at = '', updated_at = ?
                WHERE job_id = ? AND status = ? AND attempt_count < ?
                """,
                (
                    JobStatus.RUNNING.value,
                    1,
                    "任务已认领",
                    "",
                    now,
                    now,
                    job_id,
                    JobStatus.PENDING.value,
                    self.job_max_attempts,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM index_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_job(
        self,
        job_id: str,
        status: str,
        progress: int,
        message: str = "",
        error: str = "",
    ) -> None:
        self.ensure_db()
        try:
            target_status = JobStatus(status)
        except ValueError as exc:
            raise ValueError(f"未知任务状态: {status}") from exc
        if not 0 <= progress <= 100:
            raise ValueError("任务进度必须在 0 到 100 之间")

        now = utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, progress, started_at, finished_at FROM index_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"任务不存在: {job_id}")

            current_status = JobStatus(row["status"])
            if target_status not in ALLOWED_JOB_TRANSITIONS[current_status]:
                raise ValueError(
                    f"非法任务状态转换: {current_status.value} -> {target_status.value}"
                )

            next_progress = max(row["progress"], progress)
            started_at = row["started_at"]
            finished_at = row["finished_at"]
            if target_status is JobStatus.RUNNING and not started_at:
                started_at = now
            if target_status in TERMINAL_JOB_STATUSES:
                next_progress = 100
                finished_at = now

            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, progress = ?, message = ?, error = ?,
                    started_at = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    target_status.value,
                    next_progress,
                    message,
                    error[:2000],
                    started_at,
                    finished_at,
                    now,
                    job_id,
                ),
            )

    def retry_index_job(self, job_id: str) -> dict[str, Any]:
        self.ensure_db()
        now = utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute(
                "SELECT * FROM index_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not job:
                raise ValueError(f"任务不存在: {job_id}")
            if job["status"] != JobStatus.FAILED.value:
                raise ResourceConflictError(
                    f"只有失败任务可以重试: {job_id} ({job['status']})"
                )
            if job["attempt_count"] >= self.job_max_attempts:
                raise ResourceConflictError(
                    f"任务已达到最大尝试次数 {self.job_max_attempts}: {job_id}"
                )
            document_exists = conn.execute(
                "SELECT 1 FROM documents WHERE document_id = ?",
                (job["document_id"],),
            ).fetchone()
            knowledge_base_exists = conn.execute(
                "SELECT 1 FROM knowledge_bases WHERE knowledge_base_id = ?",
                (job["knowledge_base_id"],),
            ).fetchone()
            if not document_exists or not knowledge_base_exists:
                raise ResourceConflictError("文档或知识库已删除，无法重试该任务")

            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, progress = ?, message = ?, error = '',
                    started_at = '', finished_at = '', updated_at = ?
                WHERE job_id = ?
                """,
                (JobStatus.PENDING.value, 0, "等待重试", now, job_id),
            )
            conn.execute(
                """
                UPDATE knowledge_bases
                SET status = ?, updated_at = ?
                WHERE knowledge_base_id = ?
                """,
                ("building", now, job["knowledge_base_id"]),
            )

        logger.info(
            "index_job_retry_scheduled",
            extra={
                "job_id": job_id,
                "document_id": job["document_id"],
                "knowledge_base_id": job["knowledge_base_id"],
                "attempt_count": job["attempt_count"],
            },
        )
        return {
            "job_id": job_id,
            "document_id": job["document_id"],
            "knowledge_base_id": job["knowledge_base_id"],
            "status": JobStatus.PENDING.value,
            "reused": False,
        }

    def finish_job(
        self,
        job_id: str,
        knowledge_base_id: str,
        *,
        success: bool,
        message: str = "",
        error: str = "",
    ) -> None:
        self.ensure_db()
        now = utc_now()
        job_status = JobStatus.SUCCESS if success else JobStatus.FAILED
        kb_status = "ready" if success else "failed"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, progress = 100, message = ?, error = ?,
                    finished_at = ?, updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (
                    job_status.value,
                    message,
                    error[:2000],
                    now,
                    now,
                    job_id,
                    JobStatus.RUNNING.value,
                ),
            )
            conn.execute(
                """
                UPDATE knowledge_bases
                SET status = ?, updated_at = ?
                WHERE knowledge_base_id = ?
                """,
                (kb_status, now, knowledge_base_id),
            )

    def update_knowledge_base_status(self, knowledge_base_id: str, status: str) -> None:
        self.ensure_db()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE knowledge_bases
                SET status = ?, updated_at = ?
                WHERE knowledge_base_id = ?
                """,
                (status, utc_now(), knowledge_base_id),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM index_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        self.ensure_db()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    j.job_id,
                    j.document_id,
                    j.knowledge_base_id,
                    j.status,
                    j.progress,
                    j.message,
                    j.error,
                    j.created_at,
                    j.updated_at,
                    j.attempt_count,
                    j.started_at,
                    j.finished_at,
                    d.filename,
                    kb.name AS knowledge_base_name,
                    kb.status AS knowledge_base_status
                FROM index_jobs j
                LEFT JOIN documents d ON d.document_id = j.document_id
                LEFT JOIN knowledge_bases kb ON kb.knowledge_base_id = j.knowledge_base_id
                ORDER BY j.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any] | None:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_bases WHERE knowledge_base_id = ?",
                (knowledge_base_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        self.ensure_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT knowledge_base_id, document_id, name, status
                FROM knowledge_bases
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any]:
        self.ensure_db()
        if knowledge_base_id == "default":
            raise ValueError("默认知识库不能删除")

        kb = self.get_knowledge_base(knowledge_base_id)
        if not kb:
            raise ValueError(f"知识库不存在: {knowledge_base_id}")
        if kb["status"] == "building":
            raise ResourceConflictError("知识库正在构建，不能删除")

        index_dir = Path(kb["index_dir"]).resolve()
        allowed_root = INDEXES_DIR.resolve()
        if allowed_root not in index_dir.parents and index_dir != allowed_root:
            raise ValueError(f"索引目录不在允许范围内: {index_dir}")

        staged_dir: Path | None = None
        if index_dir.exists():
            staged_dir = allowed_root / (
                f".deleting-{knowledge_base_id}-{uuid.uuid4().hex[:8]}"
            )
            index_dir.rename(staged_dir)

        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM knowledge_bases WHERE knowledge_base_id = ?",
                    (knowledge_base_id,),
                )
                conn.execute(
                    """
                    UPDATE index_jobs
                    SET message = ?, updated_at = ?
                    WHERE knowledge_base_id = ?
                    """,
                    ("知识库已删除，历史任务仅保留记录", utc_now(), knowledge_base_id),
                )
        except Exception:
            if staged_dir and staged_dir.exists() and not index_dir.exists():
                staged_dir.rename(index_dir)
            logger.exception(
                "knowledge_base_delete_rolled_back",
                extra={"knowledge_base_id": knowledge_base_id},
            )
            raise

        if staged_dir and staged_dir.exists():
            try:
                shutil.rmtree(staged_dir)
            except OSError:
                logger.exception(
                    "knowledge_base_staged_directory_cleanup_failed",
                    extra={"knowledge_base_id": knowledge_base_id},
                )

        return {
            "knowledge_base_id": knowledge_base_id,
            "deleted": True,
        }

    def recover_interrupted_jobs(self) -> int:
        self.ensure_db()
        now = utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, knowledge_base_id
                FROM index_jobs
                WHERE status IN ('pending', 'running')
                """
            ).fetchall()
            for row in rows:
                conn.execute(
                    """
                    UPDATE index_jobs
                    SET status = ?, progress = ?, message = ?, error = ?,
                        finished_at = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        "failed",
                        100,
                        "服务重启后任务已中断，请重试任务",
                        "interrupted_by_restart",
                        now,
                        now,
                        row["job_id"],
                    ),
                )
                conn.execute(
                    """
                    UPDATE knowledge_bases
                    SET status = ?, updated_at = ?
                    WHERE knowledge_base_id = ? AND status = ?
                    """,
                    ("failed", now, row["knowledge_base_id"], "building"),
                )
        recovered_count = len(rows)
        if recovered_count:
            logger.warning(
                "interrupted_index_jobs_recovered",
                extra={"recovered_jobs": recovered_count},
            )
        return recovered_count

    def recover_staged_deletions(self) -> dict[str, int]:
        self.ensure_db()
        restored = 0
        removed = 0
        if not INDEXES_DIR.is_dir():
            return {"restored": restored, "removed": removed}

        for staged_dir in INDEXES_DIR.glob(".deleting-kb_*-*"):
            if not staged_dir.is_dir():
                continue
            staged_name = staged_dir.name.removeprefix(".deleting-")
            knowledge_base_id = staged_name.rsplit("-", 1)[0]
            kb = self.get_knowledge_base(knowledge_base_id)
            if kb:
                index_dir = Path(kb["index_dir"])
                if not index_dir.exists():
                    staged_dir.rename(index_dir)
                    restored += 1
                    continue
            shutil.rmtree(staged_dir)
            removed += 1

        if restored or removed:
            logger.warning(
                "staged_knowledge_base_deletions_recovered",
                extra={
                    "restored_directories": restored,
                    "removed_staged_directories": removed,
                },
            )
        return {"restored": restored, "removed": removed}

    def build_index(self, job_id: str) -> None:
        job = self.claim_job(job_id)
        if not job:
            logger.warning("index_job_claim_skipped", extra={"job_id": job_id})
            return

        document = self.get_document(job["document_id"])
        kb = self.get_knowledge_base(job["knowledge_base_id"])
        if not document or not kb:
            self.finish_job(
                job_id,
                job["knowledge_base_id"],
                success=False,
                error="文档或知识库记录不存在",
            )
            return

        document_path = Path(document["storage_path"])
        index_dir = Path(kb["index_dir"])
        chunks_dir = index_dir / "chunks"
        processed_dir = index_dir / "processed"

        try:
            if job["attempt_count"] > 1 and index_dir.exists():
                resolved_index_dir = index_dir.resolve()
                allowed_root = INDEXES_DIR.resolve()
                if (
                    allowed_root not in resolved_index_dir.parents
                    and resolved_index_dir != allowed_root
                ):
                    raise ValueError(f"索引目录不在允许范围内: {resolved_index_dir}")
                shutil.rmtree(resolved_index_dir)

            logger.info(
                "index_job_started",
                extra={
                    "job_id": job_id,
                    "document_id": job["document_id"],
                    "knowledge_base_id": job["knowledge_base_id"],
                    "attempt_count": job["attempt_count"],
                },
            )
            self.update_job(job_id, "running", 10, "正在提取 PDF 文本")
            from src.loader.pdf_loader import PDFLoader
            from src.splitter.structural_splitter import smart_split_by_titles, save_chunks_to_files
            from src.retriever.vector_indexer import build_vector_index

            processed_dir.mkdir(parents=True, exist_ok=True)
            with PDFLoader(document_path) as loader:
                full_text = loader.extract_full_text(add_page_marker=True)
                (processed_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
                loader.save_pages_to_jsonl(processed_dir / "pages.jsonl")

            self.update_job(job_id, "running", 35, "正在结构化切分")
            chunks = smart_split_by_titles(full_text)
            save_chunks_to_files(
                chunks,
                output_dir=str(chunks_dir),
                clear_existing=True,
                source_file=document["filename"],
            )

            self.update_job(job_id, "running", 65, "正在构建向量索引")
            build_vector_index(
                chunk_dir=str(chunks_dir / "children"),
                metadata_path=str(chunks_dir / "metadata.jsonl"),
                index_path=str(index_dir / "index.faiss"),
                output_meta_path=str(index_dir / "chunks_meta.jsonl"),
            )

            self.finish_job(
                job_id,
                job["knowledge_base_id"],
                success=True,
                message="知识库构建完成",
            )
            logger.info(
                "index_job_completed",
                extra={
                    "job_id": job_id,
                    "document_id": job["document_id"],
                    "knowledge_base_id": job["knowledge_base_id"],
                    "attempt_count": job["attempt_count"],
                },
            )
        except Exception as exc:
            self.finish_job(
                job_id,
                job["knowledge_base_id"],
                success=False,
                error=str(exc),
            )
            logger.exception(
                "index_job_failed",
                extra={
                    "job_id": job_id,
                    "document_id": job["document_id"],
                    "knowledge_base_id": job["knowledge_base_id"],
                    "attempt_count": job["attempt_count"],
                },
            )


knowledge_base_service = KnowledgeBaseService()
