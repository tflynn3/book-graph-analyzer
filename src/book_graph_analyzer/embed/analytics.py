"""
DuckDB Analytics Sidecar

Fast tabular analytics on the BGA corpus: passage metrics,
style distributions, sentence length histograms, era breakdowns.

Schema:
  passages_metrics — one row per passage with all numeric/style features
  embedding_log    — when each item was embedded (for incremental tracking)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PassageAnalytics:
    """
    DuckDB-backed analytics store for corpus metrics.

    Usage (persistent):
        db = PassageAnalytics("data/bga_analytics.duckdb")

    Usage (in-memory, for tests):
        db = PassageAnalytics(":memory:")
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS passage_metrics (
        id              VARCHAR PRIMARY KEY,
        book            VARCHAR,
        chapter         VARCHAR,
        chapter_num     INTEGER,
        paragraph_num   INTEGER,
        sentence_num    INTEGER,
        word_count      INTEGER,
        sentence_count  INTEGER,
        avg_sentence_length DOUBLE,
        passive_ratio   DOUBLE,
        dialogue_density DOUBLE,
        archaic_word_count INTEGER,
        story_era       VARCHAR,
        story_year      INTEGER,
        tolkien_register VARCHAR,
        is_dialogue     BOOLEAN,
        embedded_at     TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS embedding_log (
        id          VARCHAR,
        collection  VARCHAR,
        embedded_at TIMESTAMP,
        model       VARCHAR,
        PRIMARY KEY (id, collection)
    );

    CREATE TABLE IF NOT EXISTS entity_metrics (
        id              VARCHAR PRIMARY KEY,
        canonical_name  VARCHAR,
        alias_count     INTEGER,
        entity_type     VARCHAR,
        embedded_at     TIMESTAMP
    );
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        import duckdb
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self._db_path)
        self._init_schema()
        logger.debug("PassageAnalytics connected: %s", self._db_path)

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        for stmt in self.SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    def upsert_passage(self, passage_data: dict[str, Any]) -> None:
        """
        Upsert a single passage row.

        Expected keys match the passage_metrics schema.
        """
        now = datetime.now(timezone.utc)
        self._conn.execute("""
            INSERT OR REPLACE INTO passage_metrics VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            passage_data.get("id", ""),
            passage_data.get("book", ""),
            passage_data.get("chapter", ""),
            passage_data.get("chapter_num", 0),
            passage_data.get("paragraph_num", 0),
            passage_data.get("sentence_num", 0),
            passage_data.get("word_count", 0),
            passage_data.get("sentence_count", 0),
            passage_data.get("avg_sentence_length", 0.0),
            passage_data.get("passive_ratio", 0.0),
            passage_data.get("dialogue_density", 0.0),
            passage_data.get("archaic_word_count", 0),
            passage_data.get("story_era"),
            passage_data.get("story_year"),
            passage_data.get("tolkien_register"),
            passage_data.get("is_dialogue", False),
            now,
        ])

    def upsert_passages_bulk(self, passages: list[dict[str, Any]]) -> int:
        """Upsert multiple passages. Returns count inserted."""
        now = datetime.now(timezone.utc)
        rows = [
            (
                p.get("id", ""),
                p.get("book", ""),
                p.get("chapter", ""),
                p.get("chapter_num", 0),
                p.get("paragraph_num", 0),
                p.get("sentence_num", 0),
                p.get("word_count", 0),
                p.get("sentence_count", 0),
                p.get("avg_sentence_length", 0.0),
                p.get("passive_ratio", 0.0),
                p.get("dialogue_density", 0.0),
                p.get("archaic_word_count", 0),
                p.get("story_era"),
                p.get("story_year"),
                p.get("tolkien_register"),
                p.get("is_dialogue", False),
                now,
            )
            for p in passages
        ]
        self._conn.executemany("""
            INSERT OR REPLACE INTO passage_metrics VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, rows)
        return len(rows)

    def log_embedding(self, item_id: str, collection: str, model: str) -> None:
        """Record that an item was embedded."""
        now = datetime.now(timezone.utc)
        self._conn.execute("""
            INSERT OR REPLACE INTO embedding_log VALUES (?, ?, ?, ?)
        """, [item_id, collection, now, model])

    def log_embeddings_bulk(self, ids: list[str], collection: str, model: str) -> None:
        """Record that many items were embedded."""
        now = datetime.now(timezone.utc)
        rows = [(i, collection, now, model) for i in ids]
        self._conn.executemany("""
            INSERT OR REPLACE INTO embedding_log VALUES (?, ?, ?, ?)
        """, rows)

    def upsert_entity(self, entity_id: str, canonical_name: str,
                      alias_count: int, entity_type: str) -> None:
        now = datetime.now(timezone.utc)
        self._conn.execute("""
            INSERT OR REPLACE INTO entity_metrics VALUES (?, ?, ?, ?, ?)
        """, [entity_id, canonical_name, alias_count, entity_type, now])

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------

    def get_embedded_ids(self, collection: str) -> set[str]:
        """Return IDs already logged as embedded (for incremental build)."""
        result = self._conn.execute(
            "SELECT id FROM embedding_log WHERE collection = ?", [collection]
        ).fetchall()
        return {row[0] for row in result}

    def style_distribution(self, book: Optional[str] = None) -> list[dict]:
        """
        Return style metric distribution for a book (or all books).

        Returns: list of dicts with book, avg/min/max sentence length,
                 avg passive ratio, avg dialogue density.
        """
        where = "WHERE book = ?" if book else ""
        params = [book] if book else []
        rows = self._conn.execute(f"""
            SELECT
                book,
                COUNT(*) AS passage_count,
                ROUND(AVG(avg_sentence_length), 2) AS avg_sentence_len,
                ROUND(MIN(avg_sentence_length), 2) AS min_sentence_len,
                ROUND(MAX(avg_sentence_length), 2) AS max_sentence_len,
                ROUND(AVG(passive_ratio), 3) AS avg_passive_ratio,
                ROUND(AVG(dialogue_density), 3) AS avg_dialogue_density,
                ROUND(AVG(archaic_word_count), 2) AS avg_archaic_words
            FROM passage_metrics
            {where}
            GROUP BY book
            ORDER BY passage_count DESC
        """, params).fetchall()

        cols = ["book", "passage_count", "avg_sentence_len", "min_sentence_len",
                "max_sentence_len", "avg_passive_ratio", "avg_dialogue_density",
                "avg_archaic_words"]
        return [dict(zip(cols, row)) for row in rows]

    def era_breakdown(self) -> list[dict]:
        """Return passage count by story era."""
        rows = self._conn.execute("""
            SELECT
                COALESCE(story_era, 'unknown') AS era,
                COUNT(*) AS count,
                ROUND(AVG(avg_sentence_length), 2) AS avg_sent_len
            FROM passage_metrics
            GROUP BY era
            ORDER BY count DESC
        """).fetchall()
        return [{"era": r[0], "count": r[1], "avg_sent_len": r[2]} for r in rows]

    def sentence_length_histogram(self, bucket_size: int = 5) -> list[dict]:
        """Return sentence length histogram."""
        rows = self._conn.execute(f"""
            SELECT
                FLOOR(avg_sentence_length / {bucket_size}) * {bucket_size} AS bucket,
                COUNT(*) AS count
            FROM passage_metrics
            GROUP BY bucket
            ORDER BY bucket
        """).fetchall()
        return [{"bucket": r[0], "count": r[1]} for r in rows]

    def register_distribution(self) -> list[dict]:
        """Return passage count by Tolkien register."""
        rows = self._conn.execute("""
            SELECT
                COALESCE(tolkien_register, 'unclassified') AS register,
                COUNT(*) AS count
            FROM passage_metrics
            GROUP BY register
            ORDER BY count DESC
        """).fetchall()
        return [{"register": r[0], "count": r[1]} for r in rows]

    def total_counts(self) -> dict[str, int]:
        """Return high-level corpus counts."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) AS passages,
                COUNT(DISTINCT book) AS books,
                COUNT(DISTINCT story_era) AS eras,
                SUM(CASE WHEN is_dialogue THEN 1 ELSE 0 END) AS dialogue_passages,
                SUM(word_count) AS total_words
            FROM passage_metrics
        """).fetchone()
        if row:
            return {
                "passages": row[0],
                "books": row[1],
                "eras": row[2],
                "dialogue_passages": row[3],
                "total_words": row[4] or 0,
            }
        return {"passages": 0, "books": 0, "eras": 0, "dialogue_passages": 0, "total_words": 0}

    def get_passage_metrics(self, passage_id: str) -> Optional[dict]:
        """Retrieve metrics for a specific passage."""
        row = self._conn.execute(
            "SELECT * FROM passage_metrics WHERE id = ?", [passage_id]
        ).fetchone()
        if not row:
            return None
        desc = self._conn.description
        return dict(zip([d[0] for d in desc], row))

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            self._conn.close()
        except Exception:
            pass
