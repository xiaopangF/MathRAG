import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.core.paths import FEEDBACK_DATA_DIR, FEEDBACK_DB_PATH


class FeedbackService:
    def __init__(self, db_path=FEEDBACK_DB_PATH):
        self.db_path = db_path

    def _connect(self):
        FEEDBACK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _ensure_db(self) -> None:
        with self._connect() as conn:
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
