from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_BY_BOOK = {
    "hobbit": {"extracted": 8, "accepted": 6},
    "fellowship": {"extracted": 10, "accepted": 7},
    "two_towers": {"extracted": 10, "accepted": 8},
    "return_of_king": {"extracted": 10, "accepted": 8},
    "silmarillion": {"extracted": 15, "accepted": 11},
}

INVERSE = {
    "PARENT_OF": "CHILD_OF",
    "CHILD_OF": "PARENT_OF",
    "SIBLING_OF": "SIBLING_OF",
    "SPOUSE_OF": "SPOUSE_OF",
    "ANCESTOR_OF": "DESCENDANT_OF",
    "DESCENDANT_OF": "ANCESTOR_OF",
    "FOSTER_PARENT_OF": "FOSTER_CHILD_OF",
    "FOSTER_CHILD_OF": "FOSTER_PARENT_OF",
    "HALF_SIBLING_OF": "HALF_SIBLING_OF",
    "GRANDPARENT_OF": "GRANDCHILD_OF",
    "GRANDCHILD_OF": "GRANDPARENT_OF",
}


def check(eval_report: dict[str, Any]) -> dict[str, Any]:
    by_book = eval_report.get("by_book", {})

    per_book_minima = {}
    for b, mins in MIN_BY_BOOK.items():
        r = by_book.get(b, {})
        per_book_minima[b] = {
            "extracted_ok": int(r.get("extracted", 0)) >= mins["extracted"],
            "accepted_ok": int(r.get("accepted", 0)) >= mins["accepted"],
            "observed_extracted": int(r.get("extracted", 0)),
            "observed_accepted": int(r.get("accepted", 0)),
            "min_extracted": mins["extracted"],
            "min_accepted": mins["accepted"],
        }

    relations = []
    # Optional expanded input support
    for item in eval_report.get("relations", []):
        relations.append((item.get("source_id"), item.get("target_id"), item.get("relation_type")))

    self_links = 0
    unresolved_ids = 0
    contradictions = 0
    inverse_missing = 0

    rel_set = set(relations)
    for s, t, r in relations:
        if not s or not t:
            unresolved_ids += 1
        if s == t:
            self_links += 1
        inv = INVERSE.get(str(r))
        if inv and (t, s, inv) not in rel_set:
            inverse_missing += 1
        if (s, t, "PARENT_OF") in rel_set and (s, t, "CHILD_OF") in rel_set:
            contradictions += 1

    overall = eval_report.get("overall", {})
    precision_ok = float(overall.get("precision", 0)) >= 0.90
    recall_ok = float(overall.get("recall", 0)) >= 0.75

    minima_ok = all(v["extracted_ok"] and v["accepted_ok"] for v in per_book_minima.values())
    integrity_ok = self_links == 0 and unresolved_ids == 0 and contradictions == 0 and inverse_missing == 0

    return {
        "thresholds": {
            "precision_ok": precision_ok,
            "recall_ok": recall_ok,
            "minima_ok": minima_ok,
        },
        "graph_integrity": {
            "self_links": self_links,
            "unresolved_ids": unresolved_ids,
            "contradictions": contradictions,
            "inverse_missing": inverse_missing,
            "passed": integrity_ok,
        },
        "per_book_minima": per_book_minima,
        "passed": bool(precision_ok and recall_ok and minima_ok and integrity_ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-report", default="data/output/genealogy_eval_report.json")
    ap.add_argument("--out", default="data/output/genealogy_graph_quality_report.json")
    args = ap.parse_args()

    report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    out_report = check(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(json.dumps(out_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
