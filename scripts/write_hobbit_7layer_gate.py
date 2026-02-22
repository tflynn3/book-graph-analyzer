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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.artifact)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gate": "hobbit-7layer",
        "status": args.status,
        "suite": args.suite,
        "source_sha": args.source_sha,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
    }

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
