import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.core.errors import install_exception_handlers
from backend.core.logging import RequestContextMiddleware
from backend.core.request_limits import RequestBodyLimitMiddleware
from backend.main import app


def test_validation_error_has_stable_code_and_request_id():
    response = TestClient(app).post(
        "/api/chat",
        headers={"X-Request-ID": "validation-test-001"},
        json={"question": "   "},
    )

    assert response.status_code == 422
    assert response.headers["x-request-id"] == "validation-test-001"
    assert response.json()["error"] == {
        "code": "validation_error",
        "request_id": "validation-test-001",
    }


def test_http_error_preserves_detail_and_adds_error_metadata():
    response = TestClient(app).get(
        "/api/jobs/job_missing",
        headers={"X-Request-ID": "not-found-test-001"},
    )

    assert response.status_code == 404
    assert "任务不存在" in response.json()["detail"]
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["request_id"] == "not-found-test-001"


def test_unhandled_error_is_safe_and_traceable():
    test_app = FastAPI()
    install_exception_handlers(test_app)
    test_app.add_middleware(RequestContextMiddleware)

    @test_app.get("/explode")
    def explode():
        raise RuntimeError("private database password")

    response = TestClient(test_app, raise_server_exceptions=False).get(
        "/explode",
        headers={"X-Request-ID": "failure-test-001"},
    )

    assert response.status_code == 500
    assert response.headers["x-request-id"] == "failure-test-001"
    assert response.json()["error"] == {
        "code": "internal_error",
        "request_id": "failure-test-001",
    }
    assert "private database password" not in response.text


def test_chat_rejects_oversized_question_and_invalid_knowledge_base():
    client = TestClient(app)

    oversized = client.post("/api/chat", json={"question": "x" * 4001})
    invalid_kb = client.post(
        "/api/chat",
        json={"question": "导数", "knowledge_base_id": "../../storage"},
    )

    assert oversized.status_code == 422
    assert invalid_kb.status_code == 422


def test_settings_reject_credentials_in_base_url():
    response = TestClient(app).post(
        "/api/settings/deepseek-key",
        json={
            "api_key": "test-key",
            "base_url": "https://user:password@example.com/v1",
        },
    )

    assert response.status_code == 422


def test_feedback_rejects_oversized_context_payload():
    response = TestClient(app).post(
        "/api/feedback",
        json={
            "question": "什么是导数？",
            "rating": "up",
            "contexts": [{"content": "x" * (256 * 1024)}],
        },
    )

    assert response.status_code == 422


def test_request_body_limit_returns_traceable_413():
    response = TestClient(app).post(
        "/api/chat",
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": "payload-test-001",
        },
        content=b"x" * (1024 * 1024 + 1),
    )

    assert response.status_code == 413
    assert response.headers["x-request-id"] == "payload-test-001"
    assert response.json()["error"] == {
        "code": "payload_too_large",
        "request_id": "payload-test-001",
    }


def test_request_body_limit_counts_streamed_chunks_without_content_length():
    sent_messages = []
    incoming_messages = iter(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"789012", "more_body": False},
        ]
    )

    async def receive():
        return next(incoming_messages)

    async def send(message):
        sent_messages.append(message)

    async def downstream(scope, receive, send):
        await receive()
        await receive()

    middleware = RequestBodyLimitMiddleware(
        downstream,
        default_limit_bytes=10,
        upload_limit_bytes=20,
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/chat",
        "headers": [],
    }

    asyncio.run(middleware(scope, receive, send))

    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413
