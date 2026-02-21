"""SQLite-backed human review queue + audit trail."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class ReviewItem:
    id: str
    item_type: str
    confidence: float
    payload: dict[str, Any]
    status: str
    source: str
    needs_review: bool
    created_at: str
    updated_at: str


class ReviewStore:
    """Manage pending review queue and decision audit trail."""

    def __init__(self, db_path: str | Path = "data/review_queue.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    source TEXT NOT NULL DEFAULT 'pipeline',
                    needs_review INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_decisions (
                    id TEXT PRIMARY KEY,
                    reviewer TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    before_json TEXT,
                    after_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_items_type_status ON review_items(item_type, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_items_status ON review_items(status)")

    def add_item(
        self,
        item_type: str,
        confidence: float,
        payload: dict[str, Any],
        item_id: Optional[str] = None,
        source: str = "pipeline",
        needs_review: bool = True,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        rid = item_id or f"{item_type}_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO review_items
                (id, item_type, confidence, payload_json, status, source, needs_review, created_at, updated_at)
                VALUES (?, ?, ?, ?,
                        COALESCE((SELECT status FROM review_items WHERE id=?), 'pending'),
                        ?, ?,
                        COALESCE((SELECT created_at FROM review_items WHERE id=?), ?),
                        ?)
                """,
                (
                    rid,
                    item_type,
                    float(confidence),
                    json.dumps(payload, ensure_ascii=False),
                    rid,
                    source,
                    1 if needs_review else 0,
                    rid,
                    now,
                    now,
                ),
            )
        return rid

    def pending_counts(self) -> dict[str, int]:
        counts = {"entity": 0, "conflict": 0, "rule": 0, "relationship": 0, "total": 0}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT item_type, COUNT(*) c FROM review_items WHERE status='pending' GROUP BY item_type"
            ).fetchall()
        for row in rows:
            c = int(row["c"])
            counts[row["item_type"]] = c
            counts["total"] += c
        return counts

    def get_pending(self, item_type: str, limit: int = 100) -> list[ReviewItem]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM review_items WHERE item_type=? AND status='pending' ORDER BY created_at LIMIT ?",
                (item_type, limit),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM review_items WHERE id=?", (item_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def decide(
        self,
        item_id: str,
        decision: str,
        notes: str = "",
        reviewer: str = "human",
        edited_payload: Optional[dict[str, Any]] = None,
        log_to_neo4j: bool = True,
    ) -> bool:
        item = self.get_item(item_id)
        if not item:
            return False

        now = datetime.now(timezone.utc).isoformat()
        payload_after = edited_payload if edited_payload is not None else item.payload

        status_map = {
            "accept": "accepted",
            "accepted": "accepted",
            "reject": "rejected",
            "rejected": "rejected",
            "edit": "edited",
            "edited": "edited",
            "defer": "deferred",
            "deferred": "deferred",
        }
        status = status_map.get(decision.lower(), decision.lower())

        # Acceptance updates confidence per issue requirement
        new_conf = item.confidence
        if status in ("accepted", "edited"):
            new_conf = max(new_conf, 0.95)
        elif status == "rejected":
            new_conf = min(new_conf, 0.10)

        with self._conn() as conn:
            conn.execute(
                "UPDATE review_items SET status=?, confidence=?, payload_json=?, updated_at=? WHERE id=?",
                (status, float(new_conf), json.dumps(payload_after, ensure_ascii=False), now, item_id),
            )
            conn.execute(
                """
                INSERT INTO review_decisions
                (id, reviewer, item_type, item_id, decision, timestamp, notes, before_json, after_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"decision_{uuid.uuid4().hex[:12]}",
                    reviewer,
                    item.item_type,
                    item_id,
                    status,
                    now,
                    notes,
                    json.dumps(item.payload, ensure_ascii=False),
                    json.dumps(payload_after, ensure_ascii=False),
                ),
            )

        if log_to_neo4j:
            self._log_review_decision_to_neo4j(
                reviewer=reviewer,
                item_type=item.item_type,
                item_id=item_id,
                decision=status,
                timestamp=now,
                notes=notes,
            )

        return True

    def recent_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM review_decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _row_to_item(self, row: sqlite3.Row) -> ReviewItem:
        return ReviewItem(
            id=row["id"],
            item_type=row["item_type"],
            confidence=float(row["confidence"]),
            payload=json.loads(row["payload_json"]),
            status=row["status"],
            source=row["source"],
            needs_review=bool(row["needs_review"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _log_review_decision_to_neo4j(
        self,
        reviewer: str,
        item_type: str,
        item_id: str,
        decision: str,
        timestamp: str,
        notes: str,
    ) -> None:
        try:
            from ..graph.connection import get_driver

            driver = get_driver()
            if not driver:
                return
            with driver.session() as session:
                session.run(
                    """
                    MERGE (r:ReviewDecision {id: $id})
                    SET r.reviewer = $reviewer,
                        r.item_type = $item_type,
                        r.item_id = $item_id,
                        r.decision = $decision,
                        r.timestamp = $timestamp,
                        r.notes = $notes
                    """,
                    id=f"review_{uuid.uuid4().hex[:16]}",
                    reviewer=reviewer,
                    item_type=item_type,
                    item_id=item_id,
                    decision=decision,
                    timestamp=timestamp,
                    notes=notes,
                )
            driver.close()
        except Exception:
            return
