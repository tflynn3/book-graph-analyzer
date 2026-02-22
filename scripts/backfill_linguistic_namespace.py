"""Backfill linguistic lineage IDs to canonical namespace.

Canonical namespace contract:
  LanguageForm.id = lf:<entity_id>:<form-slug>

Updates:
- JSON lineage artifacts (in-place or dry-run)
- Neo4j LanguageForm node ids + DERIVED_FROM references
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from book_graph_analyzer.worldbible.lineage import (
    canonical_language_form_id,
    lineages_to_json,
    load_lineages_from_file,
)


def migrate_json(path: Path, write: bool) -> tuple[int, int]:
    lineages = load_lineages_from_file(path)
    total = sum(len(l.forms) for l in lineages)
    changed = 0

    original = json.loads(path.read_text(encoding="utf-8"))
    # parse_lineage already canonicalizes ids; compare against original ids if present
    old_ids: set[str] = set()
    for lin in original.get("lineages", []):
        for f in lin.get("forms", []):
            if isinstance(f, dict) and f.get("id"):
                old_ids.add(str(f["id"]))

    new_payload = lineages_to_json(lineages)
    for lin in new_payload.get("lineages", []):
        for f in lin.get("forms", []):
            fid = str(f.get("id", ""))
            if fid and fid not in old_ids:
                changed += 1

    if write:
        path.write_text(json.dumps(new_payload, indent=2), encoding="utf-8")

    return total, changed


def migrate_neo4j(write: bool) -> tuple[int, int]:
    from book_graph_analyzer.graph.writer import GraphWriter

    writer = GraphWriter()
    try:
        rows = writer.driver.session().run(
            """
            MATCH (lf:LanguageForm)
            RETURN lf.id AS id, lf.form AS form, coalesce(lf.entity_id, '') AS entity_id
            """
        )
        pairs: list[tuple[str, str]] = []
        for r in rows:
            old_id = str(r.get("id") or "")
            form = str(r.get("form") or "")
            entity_id = str(r.get("entity_id") or "")
            if not old_id or not form or not entity_id:
                continue
            new_id = canonical_language_form_id(entity_id, form, legacy_id=old_id)
            if new_id != old_id:
                pairs.append((old_id, new_id))

        if write and pairs:
            with writer.driver.session() as s:
                for old_id, new_id in pairs:
                    s.run(
                        """
                        MATCH (lf:LanguageForm {id:$old_id})
                        SET lf.id = $new_id
                        """,
                        old_id=old_id,
                        new_id=new_id,
                    )
        return len(pairs), len(pairs)
    finally:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="*", default=[], help="Lineage JSON files to backfill")
    ap.add_argument("--neo4j", action="store_true", help="Backfill Neo4j LanguageForm ids")
    ap.add_argument("--write", action="store_true", help="Apply changes (default is dry-run)")
    args = ap.parse_args()

    for p in args.json:
        total, changed = migrate_json(Path(p), write=args.write)
        print(f"{p}: {changed}/{total} forms need namespace migration")

    if args.neo4j:
        total, changed = migrate_neo4j(write=args.write)
        print(f"neo4j: {changed}/{total} LanguageForm nodes need namespace migration")


if __name__ == "__main__":
    main()
