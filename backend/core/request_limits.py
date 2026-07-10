from typing import Any

from starlette.responses import JSONResponse

from backend.core.logging import request_id_context


class RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    def __init__(
        self,
        app,
        *,
        default_limit_bytes: int,
        upload_limit_bytes: int,
        upload_path: str = "/api/documents/upload",
    ):
        self.app = app
        self.default_limit_bytes = default_limit_bytes
        self.upload_limit_bytes = upload_limit_bytes
        self.upload_path = upload_path

    def _limit_for_scope(self, scope: dict[str, Any]) -> int:
        if scope.get("path") == self.upload_path:
            return self.upload_limit_bytes
        return self.default_limit_bytes

    @staticmethod
    def _content_length(scope: dict[str, Any]) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    @staticmethod
    async def _reject(scope, receive, send, limit_bytes: int) -> None:
        request_id = request_id_context.get()
        response = JSONResponse(
            status_code=413,
            content={
                "detail": f"请求体过大，上限为 {limit_bytes // 1024 // 1024} MB",
                "error": {
                    "code": "payload_too_large",
                    "request_id": request_id,
                },
            },
        )
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return

        limit_bytes = self._limit_for_scope(scope)
        content_length = self._content_length(scope)
        if content_length is not None and content_length > limit_bytes:
            await self._reject(scope, receive, send, limit_bytes)
            return

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._reject(scope, receive, send, limit_bytes)
