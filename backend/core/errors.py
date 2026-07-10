import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.core.logging import request_id_context


logger = logging.getLogger(__name__)

ERROR_CODE_BY_STATUS = {
    400: "bad_request",
    401: "authentication_failed",
    402: "quota_exceeded",
    403: "permission_denied",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "service_unavailable",
}


def _error_content(detail: Any, code: str) -> dict[str, Any]:
    return {
        "detail": detail,
        "error": {
            "code": code,
            "request_id": request_id_context.get(),
        },
    }


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code = ERROR_CODE_BY_STATUS.get(exc.status_code, "http_error")
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(_error_content(exc.detail, code)),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            _error_content(exc.errors(), "validation_error")
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request_id_context.get()
    logger.error(
        "unhandled_exception",
        exc_info=(type(exc), exc, exc.__traceback__),
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": 500,
            "request_id": request_id,
        },
    )
    return JSONResponse(
        status_code=500,
        content=_error_content(
            "服务器内部错误，请使用 request_id 查询日志",
            "internal_error",
        ),
    )


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
