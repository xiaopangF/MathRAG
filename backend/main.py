from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.core.paths import PROJECT_ROOT
from backend.services.knowledge_base_service import knowledge_base_service


load_dotenv(PROJECT_ROOT / ".env", override=True)

from backend.api.routes_chat import router as chat_router
from backend.api.routes_documents import router as documents_router
from backend.api.routes_eval import router as eval_router
from backend.api.routes_feedback import router as feedback_router
from backend.api.routes_readiness import router as readiness_router
from backend.api.routes_settings import router as settings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    knowledge_base_service.recover_interrupted_jobs()
    yield


app = FastAPI(
    title="MathRAG API",
    version="0.1.0",
    description="API backend for MathRAG textbook question answering.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
