from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services import knowledge_base_service as kb_module
from backend.services.knowledge_base_service import (
    MAX_UPLOAD_BYTES,
    KnowledgeBaseService,
    sanitize_filename,
)


def make_service(tmp_path, monkeypatch) -> KnowledgeBaseService:
    storage_dir = tmp_path / "storage"
    monkeypatch.setattr(kb_module, "STORAGE_DIR", storage_dir)
    monkeypatch.setattr(kb_module, "DOCUMENTS_DIR", storage_dir / "documents")
    monkeypatch.setattr(kb_module, "INDEXES_DIR", storage_dir / "indexes")
    return KnowledgeBaseService(storage_dir / "mathrag_backend.db")


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


def test_recover_interrupted_jobs_marks_pending_and_running_as_failed(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    document = service.save_document(make_upload(), b"%PDF-1.7\ncontent")
    pending_job = service.create_index_job(document["document_id"])
    running_job = service.create_index_job(document["document_id"])
    service.update_job(running_job["job_id"], "running", 50, "building")

    recovered = service.recover_interrupted_jobs()

    assert recovered == 2
    assert service.get_job(pending_job["job_id"])["status"] == "failed"
    assert service.get_job(running_job["job_id"])["error"] == "interrupted_by_restart"
    assert service.get_knowledge_base(pending_job["knowledge_base_id"])["status"] == "failed"


def test_rag_service_does_not_change_process_working_directory():
    source = Path("backend/services/rag_service.py").read_text(encoding="utf-8")

    assert "os.chdir" not in source
