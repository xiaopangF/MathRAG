import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from backend.core.paths import BACKEND_DB_PATH, DOCUMENTS_DIR, INDEXES_DIR, STORAGE_DIR


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_PDF_CONTENT_TYPES = {
    "",
    "application/pdf",
    "application/octet-stream",
    "binary/octet-stream",
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
    def __init__(self, db_path: Path = BACKEND_DB_PATH):
        self.db_path = db_path

    def _connect(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def ensure_db(self) -> None:
        with self._connect() as conn:
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
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save_document(self, upload: UploadFile, content: bytes) -> dict[str, Any]:
        self.ensure_db()
        filename = sanitize_filename(upload.filename or "")
        content_type = (upload.content_type or "").lower()
        if not filename.lower().endswith(".pdf"):
            raise ValueError("只支持上传 PDF 文件")
        if content_type not in ALLOWED_PDF_CONTENT_TYPES:
            raise ValueError(f"不支持的文件类型: {upload.content_type}")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"PDF 文件过大，上限为 {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
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
        return record

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        self.ensure_db()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
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

        knowledge_base_id = f"kb_{uuid.uuid4().hex[:12]}"
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        index_dir = INDEXES_DIR / knowledge_base_id
        now = utc_now()
        with self._connect() as conn:
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
        }

    def update_job(
        self,
        job_id: str,
        status: str,
        progress: int,
        message: str = "",
        error: str = "",
    ) -> None:
        self.ensure_db()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, progress = ?, message = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, progress, message, error, utc_now(), job_id),
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
            conn.row_factory = sqlite3.Row
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
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM knowledge_bases WHERE knowledge_base_id = ?",
                (knowledge_base_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        self.ensure_db()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
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

        index_dir = Path(kb["index_dir"]).resolve()
        allowed_root = INDEXES_DIR.resolve()
        if allowed_root not in index_dir.parents and index_dir != allowed_root:
            raise ValueError(f"索引目录不在允许范围内: {index_dir}")

        if index_dir.exists():
            shutil.rmtree(index_dir)

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
            for job_id, knowledge_base_id in rows:
                conn.execute(
                    """
                    UPDATE index_jobs
                    SET status = ?, progress = ?, message = ?, error = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        "failed",
                        100,
                        "服务重启后任务已中断，请重新构建索引",
                        "interrupted_by_restart",
                        now,
                        job_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE knowledge_bases
                    SET status = ?, updated_at = ?
                    WHERE knowledge_base_id = ? AND status = ?
                    """,
                    ("failed", now, knowledge_base_id, "building"),
                )
        return len(rows)

    def build_index(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        document = self.get_document(job["document_id"])
        kb = self.get_knowledge_base(job["knowledge_base_id"])
        if not document or not kb:
            self.update_job(job_id, "failed", 100, error="文档或知识库记录不存在")
            return

        document_path = Path(document["storage_path"])
        index_dir = Path(kb["index_dir"])
        chunks_dir = index_dir / "chunks"
        processed_dir = index_dir / "processed"

        try:
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

            self.update_job(job_id, "success", 100, "知识库构建完成")
            self.update_knowledge_base_status(job["knowledge_base_id"], "ready")
        except Exception as exc:
            self.update_job(job_id, "failed", 100, error=str(exc))
            self.update_knowledge_base_status(job["knowledge_base_id"], "failed")


knowledge_base_service = KnowledgeBaseService()
