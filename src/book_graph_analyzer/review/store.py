"""Human review queue storage (SQLite) + audit trail + Neo4j decision logging."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


@dataclass
class ReviewItem:
    id: str
    item_type: str  # entity|conflict|rule|relationship
    confidence: float
    payload: dict[str, Any]
    status: str = "pending"  # pending|accepted|rejected|deferred|edited
    source: str = "pipeline"
    needs_review: bool = True
    created_at: str = ""
    updated_at: str = ""


class ReviewStore:
    def __init__(self, db_path: str | Path = "data/review_queue.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_decisions_item ON review_decisions(item_type, item_id)")

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
        iid = item_id or f"{item_type}_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
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
                    iid,
                    item_type,
                    float(confidence),
                    json.dumps(payload, ensure_ascii=False),
                    iid,
                    source,
                    1 if needs_review else 0,
                    iid,
                    now,
                    now,
                ),
            )
        return iid

    def add_items(self, items: list[tuple[str, float, dict[str, Any], Optional[str], str, bool]]) -> int:
        count = 0
        for item_type, confidence, payload, item_id, source, needs_review in items:
            self.add_item(item_type, confidence, payload, item_id=item_id, source=source, needs_review=needs_review)
            count += 1
        return count

    def pending_counts(self) -> dict[str, int]:
        out = {"entity": 0, "conflict": 0, "rule": 0, "relationship": 0, "total": 0}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT item_type, COUNT(*) c FROM review_items WHERE status='pending' GROUP BY item_type"
            ).fetchall()
            for r in rows:
                out[r["item_type"]] = int(r["c"])
                out["total"] += int(r["c"])
        return out

    def get_pending(self, item_type: Optional[str] = None, limit: int = 100) -> list[ReviewItem]:
        with self._connect() as conn:
            if item_type:
                rows = conn.execute(
                    "SELECT * FROM review_items WHERE status='pending' AND item_type=? ORDER BY created_at LIMIT ?",
                    (item_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM review_items WHERE status='pending' ORDER BY created_at LIMIT ?",
                    (limit,),
                ).fetchall()

        items: list[ReviewItem] = []
        for r in rows:
            items.append(
                ReviewItem(
                    id=r["id"],
                    item_type=r["item_type"],
                    confidence=float(r["confidence"]),
                    payload=json.loads(r["payload_json"]),
                    status=r["status"],
                    source=r["source"],
                    needs_review=bool(r["needs_review"]),
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            )
        return items

    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM review_items WHERE id=?", (item_id,)).fetchone()
        if not r:
            return None
        return ReviewItem(
            id=r["id"],
            item_type=r["item_type"],
            confidence=float(r["confidence"]),
            payload=json.loads(r["payload_json"]),
            status=r["status"],
            source=r["source"],
            needs_review=bool(r["needs_review"]),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    def decide(
        self,
        item_id: str,
        decision: str,  # accepted|rejected|edited|deferred
        notes: str = "",
        reviewer: str = "human",
        edited_payload: Optional[dict[str, Any]] = None,
        log_to_neo4j: bool = True,
    ) -> bool:
        item = self.get_item(item_id)
        if not item:
            return False

        now = datetime.now(timezone.utc).isoformat()
        new_payload = edited_payload if edited_payload is not None else item.payload
        status = {
            "accepted": "accepted",
            "rejected": "rejected",
            "edited": "edited",
            "deferred": "deferred",
        }.get(decision, decision)

        with self._connect() as conn:
            conn.execute(
                "UPDATE review_items SET status=?, payload_json=?, updated_at=? WHERE id=?",
                (status, json.dumps(new_payload, ensure_ascii=False), now, item_id),
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
                    decision,
                    now,
                    notes,
                    json.dumps(item.payload, ensure_ascii=False),
                    json.dumps(new_payload, ensure_ascii=False),
                ),
            )

        if log_to_neo4j:
            self._log_decision_to_neo4j(
                reviewer=reviewer,
                item_type=item.item_type,
                item_id=item_id,
                decision=decision,
                timestamp=now,
                notes=notes,
            )

        return True

    def _log_decision_to_neo4j(
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
            # Non-blocking: local audit in sqlite is still authoritative
            return

    def recent_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
