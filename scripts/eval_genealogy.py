from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from book_graph_analyzer.worldbible.genealogy import extract_genealogy_from_text


def _load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _triple(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item["source_name"]).strip(),
        str(item["target_name"]).strip(),
        str(item["relation_type"]).strip(),
    )


def _canonical(triple: tuple[str, str, str]) -> tuple[str, str, str]:
    s, t, r = triple
    if r == "PARENT_OF":
        return (t, s, "CHILD_OF")
    if r == "ANCESTOR_OF":
        return (t, s, "DESCENDANT_OF")
    if r in {"SIBLING_OF", "SPOUSE_OF", "HALF_SIBLING_OF"}:
        a, b = sorted([s, t])
        return (a, b, r)
    return (s, t, r)


def evaluate(gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        by_book[str(row.get("book", "unknown")).lower()].append(row)

    book_metrics: dict[str, dict[str, Any]] = {}
    aggregate_gold: set[tuple[str, str, str]] = set()
    aggregate_pred: set[tuple[str, str, str]] = set()
    reject_reason_hist = Counter()

    for book, rows in by_book.items():
        gold_set = {_canonical(_triple(r)) for r in rows}
        pred_set: set[tuple[str, str, str]] = set()
        combined_text = " ".join(str(r["evidence_text"]) for r in rows)
        rels = extract_genealogy_from_text(combined_text, passage_id=f"eval:{book}")
        extracted = len(rels)
        if not rels:
            reject_reason_hist["no_relation_extracted"] += 1
        for rel in rels:
            pred_set.add(_canonical((rel.source_name, rel.target_name, rel.relation_type.value)))
            if not rel.evidence_text:
                reject_reason_hist["missing_evidence"] += 1

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        book_metrics[book] = {
            "gold": len(gold_set),
            "extracted": extracted,
            "accepted": len(pred_set),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        aggregate_gold |= gold_set
        aggregate_pred |= pred_set

    tp = len(aggregate_gold & aggregate_pred)
    fp = len(aggregate_pred - aggregate_gold)
    fn = len(aggregate_gold - aggregate_pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "overall": {
            "gold": len(aggregate_gold),
            "accepted": len(aggregate_pred),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        },
        "by_book": book_metrics,
        "reject_reason_histogram": dict(reject_reason_hist),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="tests/fixtures/genealogy_gold.jsonl")
    ap.add_argument("--out", default="data/output/genealogy_eval_report.json")
    args = ap.parse_args()

    report = evaluate(_load_gold(Path(args.gold)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
