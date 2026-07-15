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

    def list_feedback(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        rating: str | None = None,
        knowledge_base_id: str | None = None,
    ) -> dict[str, Any]:
        """Return recent feedback records plus total count."""
        self._ensure_db()
        safe_limit = max(1, min(int(limit), 100))
        safe_offset = max(0, int(offset))

        filters = []
        params: list[Any] = []
        if rating:
            filters.append("rating = ?")
            params.append(rating)
        if knowledge_base_id:
            filters.append("knowledge_base_id = ?")
            params.append(knowledge_base_id)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute(
                f"SELECT COUNT(*) FROM feedback {where_clause}",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT
                    id,
                    knowledge_base_id,
                    question,
                    answer,
                    rating,
                    reason,
                    comment,
                    top_rerank_score,
                    contexts_json,
                    created_at
                FROM feedback
                {where_clause}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, safe_limit, safe_offset],
            ).fetchall()

        items = []
        for row in rows:
            contexts = []
            try:
                contexts = json.loads(row["contexts_json"] or "[]")
            except json.JSONDecodeError:
                contexts = []
            items.append(
                {
                    "id": row["id"],
                    "knowledge_base_id": row["knowledge_base_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "rating": row["rating"],
                    "reason": row["reason"],
                    "comment": row["comment"],
                    "top_rerank_score": row["top_rerank_score"],
                    "contexts": contexts,
                    "created_at": row["created_at"],
                }
            )

        return {
            "items": items,
            "total": total,
            "limit": safe_limit,
            "offset": safe_offset,
        }

    def summarize_feedback(
        self,
        *,
        knowledge_base_id: str | None = None,
    ) -> dict[str, Any]:
        """Return aggregate feedback counts for the dashboard."""
        self._ensure_db()
        filters = []
        params: list[Any] = []
        if knowledge_base_id:
            filters.append("knowledge_base_id = ?")
            params.append(knowledge_base_id)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            summary = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS up_count,
                    SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS down_count,
                    SUM(CASE WHEN TRIM(comment) != '' THEN 1 ELSE 0 END) AS commented_count,
                    AVG(top_rerank_score) AS average_top_rerank_score,
                    MAX(created_at) AS latest_created_at
                FROM feedback
                {where_clause}
                """,
                params,
            ).fetchone()

        total = int(summary["total"] or 0)
        up_count = int(summary["up_count"] or 0)
        down_count = int(summary["down_count"] or 0)
        return {
            "total": total,
            "up_count": up_count,
            "down_count": down_count,
            "commented_count": int(summary["commented_count"] or 0),
            "average_top_rerank_score": summary["average_top_rerank_score"],
            "latest_created_at": summary["latest_created_at"],
            "positive_rate": up_count / total if total else 0.0,
        }


feedback_service = FeedbackService()
