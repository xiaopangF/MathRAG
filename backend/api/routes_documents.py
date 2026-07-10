from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from backend.schemas.documents import (
    BuildIndexRequest,
    BuildIndexResponse,
    DeleteKnowledgeBaseResponse,
    DocumentResponse,
    JobStatusResponse,
    KnowledgeBaseResponse,
)
from backend.services.knowledge_base_service import MAX_UPLOAD_BYTES, knowledge_base_service


router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise ValueError("上传文件为空")
        record = knowledge_base_service.save_document(file, content)
        return DocumentResponse(**record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/index/build", response_model=BuildIndexResponse)
def build_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    try:
        job = knowledge_base_service.create_index_job(request.document_id)
        background_tasks.add_task(knowledge_base_service.build_index, job["job_id"])
        return BuildIndexResponse(**job)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = knowledge_base_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    return JobStatusResponse(**job)


@router.get("/jobs", response_model=list[JobStatusResponse])
def list_jobs(limit: int = 20):
    safe_limit = max(1, min(limit, 100))
    return [
        JobStatusResponse(**item)
        for item in knowledge_base_service.list_jobs(limit=safe_limit)
    ]


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseResponse])
def list_knowledge_bases():
    return [
        KnowledgeBaseResponse(**item)
        for item in knowledge_base_service.list_knowledge_bases()
    ]


@router.delete("/knowledge-bases/{knowledge_base_id}", response_model=DeleteKnowledgeBaseResponse)
def delete_knowledge_base(knowledge_base_id: str):
    try:
        result = knowledge_base_service.delete_knowledge_base(knowledge_base_id)
        return DeleteKnowledgeBaseResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
