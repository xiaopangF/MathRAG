from types import SimpleNamespace

from backend.services.eval_service import eval_service
from backend.services.knowledge_base_service import ResourceConflictError
from backend.services.rag_service import RAGBusyError
from backend.api import routes_chat
from backend.api import routes_documents
from backend.api import routes_settings
from backend.main import app
from fastapi.testclient import TestClient
from src.generation.llm_generator import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMQuotaError,
    LLMRateLimitError,
)


def test_eval_service_loads_latest_hybrid_metrics():
    result = eval_service.latest("hybrid")

    assert result["method"] == "hybrid"
    assert result["report_path"] == "reports/retrieval_metrics_100_hybrid.json"
    assert result["metrics"]["question_count"] == 100
    assert result["metrics"]["recall_at_5"] >= 0.95
    assert result["metrics"]["mrr"] >= 0.9

    grounded = eval_service.latest("grounded_sample")
    assert grounded["metrics"]["page_metrics"]["recall_at_5"] == 1.0
    assert grounded["metrics"]["section_metrics"]["recall_at_5"] == 0.8


def test_backend_exposes_core_routes():
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/api/eval/latest?method=hybrid").status_code == 200
    assert client.get("/api/eval/latest?method=grounded_sample").status_code == 200
    assert client.post("/api/chat", json={}).status_code == 422
    assert client.post("/api/documents/upload").status_code == 422
    assert client.post("/api/index/build", json={}).status_code == 422
    assert client.get("/api/settings").status_code == 200


def test_request_id_is_preserved_in_response():
    response = TestClient(app).get(
        "/health",
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "test-request-123"


def test_knowledge_base_list_route(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        routes_documents.knowledge_base_service,
        "list_knowledge_bases",
        lambda: [
            {
                "knowledge_base_id": "kb_test",
                "document_id": "doc_test",
                "name": "test.pdf",
                "status": "ready",
            }
        ],
    )

    response = client.get("/api/knowledge-bases")

    assert response.status_code == 200
    assert response.json()[0]["knowledge_base_id"] == "kb_test"


def test_job_history_route(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        routes_documents.knowledge_base_service,
        "list_jobs",
        lambda limit=20: [
            {
                "job_id": "job_test",
                "document_id": "doc_test",
                "knowledge_base_id": "kb_test",
                "status": "success",
                "progress": 100,
                "message": "知识库构建完成",
                "error": "",
                "created_at": "2026-07-10T00:00:00+00:00",
                "updated_at": "2026-07-10T00:01:00+00:00",
                "filename": "test.pdf",
                "knowledge_base_name": "test.pdf",
                "knowledge_base_status": "ready",
            }
        ],
    )

    response = client.get("/api/jobs")

    assert response.status_code == 200
    assert response.json()[0]["job_id"] == "job_test"
    assert response.json()[0]["filename"] == "test.pdf"


def test_delete_knowledge_base_route(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        routes_documents.knowledge_base_service,
        "delete_knowledge_base",
        lambda knowledge_base_id: {
            "knowledge_base_id": knowledge_base_id,
            "deleted": True,
        },
    )

    response = client.delete("/api/knowledge-bases/kb_test")

    assert response.status_code == 200
    assert response.json() == {
        "knowledge_base_id": "kb_test",
        "deleted": True,
    }


def test_delete_building_knowledge_base_returns_conflict(monkeypatch):
    client = TestClient(app)

    def raise_conflict(knowledge_base_id):
        raise ResourceConflictError("知识库正在构建，不能删除")

    monkeypatch.setattr(
        routes_documents.knowledge_base_service,
        "delete_knowledge_base",
        raise_conflict,
    )

    response = client.delete("/api/knowledge-bases/kb_building")

    assert response.status_code == 409
    assert "正在构建" in response.json()["detail"]


def test_deepseek_key_can_be_configured(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        routes_settings,
        "runtime_settings",
        SimpleNamespace(allow_runtime_api_key=True),
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    response = client.post(
        "/api/settings/deepseek-key",
        json={
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )

    assert response.status_code == 200
    assert response.json()["deepseek_api_key_configured"] is True


def test_deepseek_key_update_is_disabled_in_production(monkeypatch):
    monkeypatch.setattr(
        routes_settings,
        "runtime_settings",
        SimpleNamespace(allow_runtime_api_key=False),
    )

    response = TestClient(app).post(
        "/api/settings/deepseek-key",
        json={"api_key": "test-key"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "permission_denied"


def test_chat_returns_401_when_deepseek_key_is_invalid(monkeypatch):
    client = TestClient(app)

    def raise_auth_error(*args, **kwargs):
        raise LLMAuthenticationError("DeepSeek API Key 无效或已失效，请在设置中重新填写有效 Key。")

    monkeypatch.setattr(routes_chat.rag_service, "ask", raise_auth_error)

    response = client.post("/api/chat", json={"question": "什么是导数？"})

    assert response.status_code == 401
    assert "API Key" in response.json()["detail"]


def test_chat_returns_402_when_deepseek_quota_is_insufficient(monkeypatch):
    client = TestClient(app)

    def raise_quota_error(*args, **kwargs):
        raise LLMQuotaError("DeepSeek 账户余额或额度不足，请检查控制台余额、套餐或充值状态。")

    monkeypatch.setattr(routes_chat.rag_service, "ask", raise_quota_error)

    response = client.post("/api/chat", json={"question": "什么是导数？"})

    assert response.status_code == 402
    assert "余额" in response.json()["detail"]


def test_chat_returns_429_when_deepseek_is_rate_limited(monkeypatch):
    client = TestClient(app)

    def raise_rate_limit_error(*args, **kwargs):
        raise LLMRateLimitError("DeepSeek 请求过于频繁，已触发限流，请稍后再试。")

    monkeypatch.setattr(routes_chat.rag_service, "ask", raise_rate_limit_error)

    response = client.post("/api/chat", json={"question": "什么是导数？"})

    assert response.status_code == 429
    assert "限流" in response.json()["detail"]


def test_chat_returns_503_when_deepseek_connection_fails(monkeypatch):
    client = TestClient(app)

    def raise_connection_error(*args, **kwargs):
        raise LLMConnectionError("无法连接 DeepSeek 服务，请检查网络、代理或 DEEPSEEK_BASE_URL。")

    monkeypatch.setattr(routes_chat.rag_service, "ask", raise_connection_error)

    response = client.post("/api/chat", json={"question": "什么是导数？"})

    assert response.status_code == 503
    assert "无法连接" in response.json()["detail"]


def test_chat_returns_503_when_rag_capacity_is_exhausted(monkeypatch):
    client = TestClient(app)

    def raise_busy_error(*args, **kwargs):
        raise RAGBusyError("问答服务当前繁忙，请稍后重试")

    monkeypatch.setattr(routes_chat.rag_service, "ask", raise_busy_error)

    response = client.post("/api/chat", json={"question": "什么是导数？"})

    assert response.status_code == 503
    assert "繁忙" in response.json()["detail"]


def test_failed_job_retry_route_schedules_background_task(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(
        routes_documents.knowledge_base_service,
        "get_job",
        lambda job_id: {"job_id": job_id, "status": "failed"},
    )
    monkeypatch.setattr(
        routes_documents.knowledge_base_service,
        "retry_index_job",
        lambda job_id: {
            "job_id": job_id,
            "document_id": "doc_test",
            "knowledge_base_id": "kb_test",
            "status": "pending",
            "reused": False,
        },
    )
    monkeypatch.setattr(
        routes_documents.knowledge_base_service,
        "build_index",
        lambda job_id: None,
    )

    response = client.post("/api/jobs/job_test/retry")

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
