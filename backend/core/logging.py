import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from starlette.responses import JSONResponse

from backend.core.settings import BackendSettings


request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


class StructuredFormatter(logging.Formatter):
    _extra_fields = (
        "environment",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "client_ip",
        "job_id",
        "document_id",
        "knowledge_base_id",
        "attempt_count",
        "recovered_jobs",
        "restored_directories",
        "removed_staged_directories",
        "pdf_total_pages",
        "pdf_text_pages",
        "ocr_recommended_pages",
        "ocr_applied_pages",
        "ocr_failed_pages",
        "ocr_skipped_pages",
        "removed_margin_blocks",
    )

    def __init__(self, *, json_output: bool):
        super().__init__()
        self.json_output = json_output

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for field in self._extra_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if self.json_output:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        details = " ".join(
            f"{key}={value}"
            for key, value in payload.items()
            if key not in {"timestamp", "level", "logger", "message", "exception"}
        )
        line = (
            f"{payload['timestamp']} {payload['level']} {payload['logger']} "
            f"{payload['message']} {details}"
        ).rstrip()
        if "exception" in payload:
            line = f"{line}\n{payload['exception']}"
        return line


def configure_logging(settings: BackendSettings) -> None:
    app_logger = logging.getLogger("backend")
    app_logger.setLevel(settings.log_level)
    app_logger.propagate = False
    formatter = StructuredFormatter(json_output=settings.log_json)

    handler = next(
        (
            item
            for item in app_logger.handlers
            if getattr(item, "_mathrag_handler", False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler._mathrag_handler = True  # type: ignore[attr-defined]
        app_logger.addHandler(handler)
    handler.setLevel(settings.log_level)
    handler.setFormatter(formatter)
    logging.getLogger("uvicorn.access").disabled = True


def _request_id_from_scope(scope: dict[str, Any], header_name: bytes) -> str:
    for name, value in scope.get("headers", []):
        if name.lower() == header_name:
            candidate = value.decode("ascii", errors="ignore").strip()
            if candidate and len(candidate) <= 128 and all(
                char.isalnum() or char in "-_." for char in candidate
            ):
                return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app, *, header_name: str = "X-Request-ID"):
        self.app = app
        self.header_name = header_name
        self.header_name_bytes = header_name.lower().encode("ascii")
        self.logger = logging.getLogger("backend.request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope, self.header_name_bytes)
        token = request_id_context.set(request_id)
        started_at = time.perf_counter()
        status_code = 500
        response_started = False
        failed = False

        async def send_with_request_id(message):
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != self.header_name_bytes
                ]
                headers.append((self.header_name_bytes, request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        client = scope.get("client")
        extra = {
            "method": scope.get("method", ""),
            "path": scope.get("path", ""),
            "client_ip": client[0] if client else None,
            "request_id": request_id,
        }
        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            failed = True
            extra["status_code"] = status_code
            extra["duration_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            self.logger.exception("request_failed", extra=extra)
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "服务器内部错误，请使用 request_id 查询日志",
                    "error": {
                        "code": "internal_error",
                        "request_id": request_id,
                    },
                },
            )
            await response(scope, receive, send_with_request_id)
        finally:
            if response_started and not failed:
                extra["status_code"] = status_code
                extra["duration_ms"] = round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                )
                log_method = (
                    self.logger.debug
                    if scope.get("path") == "/health"
                    else self.logger.info
                )
                log_method("request_completed", extra=extra)
            request_id_context.reset(token)
