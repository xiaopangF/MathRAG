import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from backend.core.database import apply_sqlite_migrations
from backend.core.paths import FEEDBACK_DB_PATH
from backend.core.settings import get_settings


runtime_settings = get_settings()
FEEDBACK_SCHEMA_VERSION = 1


class FeedbackService:
    def __init__(
        self,
        db_path: Path = FEEDBACK_DB_PATH,
        *,
        sqlite_timeout_seconds: float | None = None,
    ):
        self.db_path = Path(db_path)
        self.sqlite_timeout_seconds = (
            sqlite_timeout_seconds
            if sqlite_timeout_seconds is not None
            else runtime_settings.sqlite_timeout_seconds
        )
        self._db_lock = RLock()
        self._db_initialized = False

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.sqlite_timeout_seconds,
        )
        conn.execute(
            f"PRAGMA busy_timeout = {int(self.sqlite_timeout_seconds * 1000)}"
        )
        return conn

    @staticmethod
    def _migrate_schema_v1(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_base_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating TEXT NOT NULL,
                reason TEXT NOT NULL,
                comment TEXT NOT NULL,
                top_rerank_score REAL,
                contexts_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    def _ensure_db(self) -> None:
        if self._db_initialized:
            return
        with self._db_lock:
            if self._db_initialized:
                return
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                apply_sqlite_migrations(
                    conn,
                    database_name="feedback database",
                    migrations={1: self._migrate_schema_v1},
                )
            self._db_initialized = True

    def save(self, payload: dict[str, Any]) -> int:
        self._ensure_db()
        created_at = datetime.now(timezone.utc).isoformat()
        contexts_json = json.dumps(
            payload.get("contexts", []),
            ensure_ascii=False,
        )

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback (
                    knowledge_base_id,
                    question,
                    answer,
                    rating,
                    reason,
                    comment,
                    top_rerank_score,
                    contexts_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("knowledge_base_id", "default"),
                    payload["question"],
                    payload.get("answer", ""),
                    payload["rating"],
                    payload.get("reason", ""),
                    payload.get("comment", ""),
                    payload.get("top_rerank_score"),
                    contexts_json,
                    created_at,
                ),
            )
            return int(cursor.lastrowid)


feedback_service = FeedbackService()
