from __future__ import annotations

import json
from pathlib import Path

from book_graph_analyzer.graph.writer import GraphWriter
from book_graph_analyzer.worldbible.lineage import lineages_to_json
from book_graph_analyzer.worldbible.lineage_density import (
    BOOK_THRESHOLDS,
    build_lineages_from_corpus,
    compute_lineage_metrics,
    load_book_surface_text,
    load_seed_alias_catalog,
    parse_lineages_payload,
    threshold_pass,
)

BOOKS = {
    "hobbit": {
        "title": "The Hobbit",
        "events": "data/output/hobbit_events.json",
        "lineages": "data/output/layer_load/hobbit_lineages.json",
        "lore": "data/output/layer_load/hobbit_lore_depth.json",
    },
    "fellowship_of_ring": {
        "title": "Fellowship of the Ring",
        "events": "data/output/fellowship_events.json",
        "lineages": "data/output/layer_load/fellowship_of_ring_lineages.json",
        "lore": "data/output/layer_load/fellowship_of_ring_lore_depth.json",
    },
    "two_towers": {
        "title": "The Two Towers",
        "events": "data/output/twotowers_events.json",
        "lineages": "data/output/layer_load/two_towers_lineages.json",
        "lore": "data/output/layer_load/two_towers_lore_depth.json",
    },
    "return_of_king": {
        "title": "The Return of the King",
        "events": "data/output/return_events.json",
        "lineages": "data/output/layer_load/return_of_king_lineages.json",
        "lore": "data/output/layer_load/return_of_king_lore_depth.json",
    },
    "silmarillion": {
        "title": "The Silmarillion",
        "events": "data/output/silmarillion_events.json",
        "lineages": "data/output/layer_load/silmarillion_lineages.json",
        "lore": "data/output/layer_load/silmarillion_lore_depth.json",
    },
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = load_seed_alias_catalog(root)
    canonical_ids = set(catalog.keys())

    before_after: dict[str, dict] = {}

    writer = GraphWriter()
    try:
        for book_slug, cfg in BOOKS.items():
            lineage_path = root / cfg["lineages"]
            old_payload = json.loads(lineage_path.read_text(encoding="utf-8")) if lineage_path.exists() else {"lineages": []}
            before = compute_lineage_metrics(old_payload.get("lineages", []), canonical_ids)

            corpus = load_book_surface_text(root / cfg["events"], root / cfg["lore"])
            generated = build_lineages_from_corpus(corpus, catalog, source_passage_id=book_slug)
            parsed = parse_lineages_payload(generated)
            new_payload = lineages_to_json(parsed)

            lineage_path.parent.mkdir(parents=True, exist_ok=True)
            lineage_path.write_text(json.dumps(new_payload, indent=2), encoding="utf-8")

            # Idempotent per-book rewrite in graph for lineage forms and derivations.
            with writer.driver.session() as s:
                s.run(
                    """
                    MATCH (lf:LanguageForm {source_passage_id: $book_slug})
                    OPTIONAL MATCH (lf)-[r:DERIVED_FROM]-()
                    DELETE r
                    WITH lf
                    DETACH DELETE lf
                    """,
                    book_slug=book_slug,
                )
            writer.write_linguistic_lineage_batch(parsed)

            after = compute_lineage_metrics(new_payload.get("lineages", []), canonical_ids)
            threshold = BOOK_THRESHOLDS[book_slug]
            before_after[book_slug] = {
                "title": cfg["title"],
                "before": before,
                "after": after,
                "threshold": {
                    "min_lineages": threshold.min_lineages,
                    "min_forms": threshold.min_forms,
                    "min_derivations": threshold.min_derivations,
                    "min_join_rate": threshold.min_join_rate,
                },
                "pass": threshold_pass(after, threshold),
            }
    finally:
        writer.close()

    out_json = root / "data" / "output" / "layer_load" / "lineage_density_report.json"
    out_json.write_text(json.dumps(before_after, indent=2), encoding="utf-8")

    print(json.dumps(before_after, indent=2))


if __name__ == "__main__":
    main()
