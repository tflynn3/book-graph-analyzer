#!/usr/bin/env python3
"""Write machine-readable Hobbit 7-layer acceptance gate artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Hobbit 7-layer acceptance gate artifact")
    parser.add_argument("--status", choices=["PASS", "FAIL"], required=True)
    parser.add_argument(
        "--artifact",
        default="gates/hobbit_7layer_acceptance_gate.json",
        help="Output path for gate artifact JSON",
    )
    parser.add_argument("--source-sha", default="unknown")
    parser.add_argument(
        "--suite",
        default="tests/test_milestone_47_49_50_51_acceptance.py",
        help="Acceptance test suite identifier",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--schema-version", default="1")
    parser.add_argument("--acceptance-smoke", choices=["PASS", "FAIL"], default="FAIL")
    parser.add_argument("--schema-check", choices=["PASS", "FAIL"], default="FAIL")
    parser.add_argument("--data-structure-check", choices=["PASS", "FAIL"], default="FAIL")
    parser.add_argument("--graph-accuracy-check", choices=["PASS", "FAIL"], default="FAIL")
    parser.add_argument("--layer-47", choices=["PASS", "FAIL"], default="FAIL")
    parser.add_argument("--layer-49", choices=["PASS", "FAIL"], default="FAIL")
    parser.add_argument("--layer-50", choices=["PASS", "FAIL"], default="FAIL")
    parser.add_argument("--layer-51", choices=["PASS", "FAIL"], default="FAIL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.artifact)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gate": "hobbit-7layer",
        "schema_version": args.schema_version,
        "status": args.status,
        "suite": args.suite,
        "source_sha": args.source_sha,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "acceptance_smoke": args.acceptance_smoke,
            "schema": args.schema_check,
            "data_structure": args.data_structure_check,
            "graph_accuracy": args.graph_accuracy_check,
        },
        "layers": {
            "47": args.layer_47,
            "49": args.layer_49,
            "50": args.layer_50,
            "51": args.layer_51,
        },
        "notes": args.notes,
    }

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
