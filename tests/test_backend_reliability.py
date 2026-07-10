from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from backend.services import knowledge_base_service as kb_module
from backend.services.feedback_service import FeedbackService
from backend.services.knowledge_base_service import (
    MAX_UPLOAD_BYTES,
    KnowledgeBaseService,
    sanitize_filename,
)
from backend.services.rag_service import RAGBusyError, RAGService


def make_service(tmp_path, monkeypatch, *, job_max_attempts=3) -> KnowledgeBaseService:
    storage_dir = tmp_path / "storage"
    monkeypatch.setattr(kb_module, "DOCUMENTS_DIR", storage_dir / "documents")
    monkeypatch.setattr(kb_module, "INDEXES_DIR", storage_dir / "indexes")
    return KnowledgeBaseService(
        storage_dir / "mathrag_backend.db",
        job_max_attempts=job_max_attempts,
    )


def make_upload(filename="course.pdf", content_type="application/pdf"):
    return SimpleNamespace(filename=filename, content_type=content_type)


def test_sanitize_filename_strips_path_and_unsafe_characters():
    assert sanitize_filename("../unsafe:name?.pdf") == "unsafe_name_.pdf"
    assert sanitize_filename("") == "document.pdf"


def test_save_document_requires_pdf_filename_type_size_and_magic(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="PDF"):
        service.save_document(make_upload("notes.txt"), b"%PDF-1.7\n")

    with pytest.raises(ValueError, match="类型"):
        service.save_document(make_upload(content_type="text/plain"), b"%PDF-1.7\n")

    with pytest.raises(ValueError, match="过大"):
        service.save_document(make_upload(), b"%PDF-" + b"x" * MAX_UPLOAD_BYTES)

    with pytest.raises(ValueError, match="有效 PDF"):
        service.save_document(make_upload(), b"not a pdf")


def test_save_document_persists_sanitized_pdf(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)

    record = service.save_document(
        make_upload("../高数:教材?.pdf"),
        b"%PDF-1.7\ncontent",
    )

    assert record["filename"] == "高数_教材_.pdf"
    assert record["size_bytes"] == len(b"%PDF-1.7\ncontent")
    assert Path(record["storage_path"]).is_file()
    assert service.get_document(record["document_id"])["filename"] == "高数_教材_.pdf"


def test_save_document_removes_file_when_database_insert_fails(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    service.ensure_db()

    def fail_connect():
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(service, "_connect", fail_connect)

    with pytest.raises(sqlite3.OperationalError):
        service.save_document(make_upload(), b"%PDF-1.7\ncontent")

    assert list((tmp_path / "storage" / "documents").glob("*.pdf")) == []


def test_recover_interrupted_jobs_marks_pending_and_running_as_failed(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    first_document = service.save_document(make_upload("first.pdf"), b"%PDF-1.7\nfirst")
    second_document = service.save_document(make_upload("second.pdf"), b"%PDF-1.7\nsecond")
    pending_job = service.create_index_job(first_document["document_id"])
    running_job = service.create_index_job(second_document["document_id"])
    service.update_job(running_job["job_id"], "running", 50, "building")

    recovered = service.recover_interrupted_jobs()

    assert recovered == 2
    assert service.get_job(pending_job["job_id"])["status"] == "failed"
    assert service.get_job(running_job["job_id"])["error"] == "interrupted_by_restart"
    assert service.get_job(running_job["job_id"])["finished_at"]
    assert service.get_knowledge_base(pending_job["knowledge_base_id"])["status"] == "failed"


def test_create_index_job_reuses_active_job(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")

    first = service.create_index_job(document["document_id"])
    second = service.create_index_job(document["document_id"])

    assert first["reused"] is False
    assert second["reused"] is True
    assert second["job_id"] == first["job_id"]
    assert len(service.list_jobs()) == 1


def test_concurrent_index_requests_create_only_one_active_job(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")

    with ThreadPoolExecutor(max_workers=6) as executor:
        jobs = list(
            executor.map(
                lambda _: service.create_index_job(document["document_id"]),
                range(6),
            )
        )

    assert len({job["job_id"] for job in jobs}) == 1
    assert sum(not job["reused"] for job in jobs) == 1
    assert len(service.list_jobs()) == 1


def test_job_claim_is_atomic_and_records_attempt_metadata(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")
    job = service.create_index_job(document["document_id"])

    claimed = service.claim_job(job["job_id"])
    duplicate_claim = service.claim_job(job["job_id"])

    assert claimed["status"] == "running"
    assert claimed["attempt_count"] == 1
    assert claimed["started_at"]
    assert duplicate_claim is None


def test_job_state_machine_rejects_invalid_transition(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")
    job = service.create_index_job(document["document_id"])

    with pytest.raises(ValueError, match="非法任务状态转换"):
        service.update_job(job["job_id"], "success", 100)


def test_failed_job_can_be_retried_until_attempt_limit(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch, job_max_attempts=2)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")
    job = service.create_index_job(document["document_id"])
    claimed = service.claim_job(job["job_id"])
    service.finish_job(
        job["job_id"],
        claimed["knowledge_base_id"],
        success=False,
        error="first failure",
    )

    retry = service.retry_index_job(job["job_id"])
    second_claim = service.claim_job(job["job_id"])
    service.finish_job(
        job["job_id"],
        second_claim["knowledge_base_id"],
        success=False,
        error="second failure",
    )

    assert retry["status"] == "pending"
    assert second_claim["attempt_count"] == 2
    with pytest.raises(ValueError, match="最大尝试次数"):
        service.retry_index_job(job["job_id"])


def test_deleted_knowledge_base_job_cannot_be_retried(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")
    job = service.create_index_job(document["document_id"])
    claimed = service.claim_job(job["job_id"])
    service.finish_job(
        job["job_id"],
        claimed["knowledge_base_id"],
        success=False,
        error="failed",
    )
    service.delete_knowledge_base(job["knowledge_base_id"])

    with pytest.raises(ValueError, match="已删除"):
        service.retry_index_job(job["job_id"])


def test_delete_knowledge_base_restores_directory_when_database_fails(
    tmp_path,
    monkeypatch,
):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")
    job = service.create_index_job(document["document_id"])
    claimed = service.claim_job(job["job_id"])
    service.finish_job(
        job["job_id"],
        claimed["knowledge_base_id"],
        success=False,
        error="failed",
    )
    index_dir = Path(service.get_knowledge_base(job["knowledge_base_id"])["index_dir"])
    index_dir.mkdir(parents=True)
    (index_dir / "partial.index").write_text("partial", encoding="utf-8")

    original_connect = service._connect
    connect_count = 0

    def fail_delete_connect():
        nonlocal connect_count
        connect_count += 1
        if connect_count == 2:
            raise sqlite3.OperationalError("delete failed")
        return original_connect()

    monkeypatch.setattr(service, "_connect", fail_delete_connect)

    with pytest.raises(sqlite3.OperationalError):
        service.delete_knowledge_base(job["knowledge_base_id"])

    monkeypatch.setattr(service, "_connect", original_connect)
    assert index_dir.is_dir()
    assert (index_dir / "partial.index").is_file()
    assert service.get_knowledge_base(job["knowledge_base_id"]) is not None


def test_startup_recovers_staged_knowledge_base_deletions(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")
    job = service.create_index_job(document["document_id"])
    index_dir = Path(service.get_knowledge_base(job["knowledge_base_id"])["index_dir"])
    index_dir.mkdir(parents=True)
    (index_dir / "index.faiss").write_bytes(b"index")
    staged_dir = index_dir.parent / f".deleting-{job['knowledge_base_id']}-deadbeef"
    index_dir.rename(staged_dir)
    orphan_staged_dir = index_dir.parent / ".deleting-kb_aaaaaaaaaaaa-deadbeef"
    orphan_staged_dir.mkdir()

    result = service.recover_staged_deletions()

    assert result == {"restored": 1, "removed": 1}
    assert (index_dir / "index.faiss").is_file()
    assert not orphan_staged_dir.exists()


def test_feedback_database_uses_wal_mode(tmp_path):
    service = FeedbackService(tmp_path / "feedback" / "mathrag.db")
    feedback_id = service.save(
        {
            "question": "什么是导数？",
            "answer": "变化率",
            "rating": "up",
        }
    )

    with service._connect() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert feedback_id == 1
    assert journal_mode == "wal"


def test_backend_database_uses_wal_mode(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    service.ensure_db()

    with service._connect() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode == "wal"


def test_rag_service_rejects_when_inference_capacity_is_exhausted():
    service = RAGService(max_concurrency=1, acquire_timeout_seconds=0)
    assert service._inference_slots.acquire(blocking=False) is True
    try:
        with pytest.raises(RAGBusyError, match="繁忙"):
            service.ask("什么是导数？")
    finally:
        service._inference_slots.release()


def test_rag_service_invalidates_cached_knowledge_base():
    service = RAGService(max_concurrency=1)
    service._retrievers["kb_0123456789ab"] = object()

    assert service.invalidate_knowledge_base("kb_0123456789ab") is True
    assert service.invalidate_knowledge_base("kb_0123456789ab") is False


def test_rag_service_does_not_change_process_working_directory():
    source = Path("backend/services/rag_service.py").read_text(encoding="utf-8")

    assert "os.chdir" not in source
