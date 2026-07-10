from pydantic import BaseModel


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str = ""
    size_bytes: int
    status: str = "uploaded"


class BuildIndexRequest(BaseModel):
    document_id: str


class BuildIndexResponse(BaseModel):
    job_id: str
    document_id: str
    knowledge_base_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    knowledge_base_id: str
    status: str
    progress: int
    message: str = ""
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    filename: str | None = None
    knowledge_base_name: str | None = None
    knowledge_base_status: str | None = None


class KnowledgeBaseResponse(BaseModel):
    knowledge_base_id: str
    document_id: str
    name: str
    status: str


class DeleteKnowledgeBaseResponse(BaseModel):
    knowledge_base_id: str
    deleted: bool = True
