from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.errors import install_exception_handlers
from backend.core.logging import RequestContextMiddleware, configure_logging
from backend.core.request_limits import RequestBodyLimitMiddleware
from backend.core.settings import get_settings
from backend.services.knowledge_base_service import knowledge_base_service


settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

from backend.api.routes_chat import router as chat_router
from backend.api.routes_documents import router as documents_router
from backend.api.routes_eval import router as eval_router
from backend.api.routes_feedback import router as feedback_router
from backend.api.routes_readiness import router as readiness_router
from backend.api.routes_settings import router as settings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)
    recovered_jobs = knowledge_base_service.recover_interrupted_jobs()
    recovered_deletions = knowledge_base_service.recover_staged_deletions()
    logger.info(
        "application_started",
        extra={
            "environment": settings.environment,
            "recovered_jobs": recovered_jobs,
            "restored_directories": recovered_deletions["restored"],
            "removed_staged_directories": recovered_deletions["removed"],
        },
    )
    yield
    logger.info("application_stopped", extra={"environment": settings.environment})


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API backend for MathRAG textbook question answering.",
    lifespan=lifespan,
)
install_exception_handlers(app)

app.add_middleware(
    RequestBodyLimitMiddleware,
    default_limit_bytes=settings.max_json_body_bytes,
    upload_limit_bytes=settings.max_upload_bytes + 1024 * 1024,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RequestContextMiddleware,
    header_name=settings.request_id_header,
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(eval_router)
app.include_router(feedback_router)
app.include_router(readiness_router)
app.include_router(settings_router)
