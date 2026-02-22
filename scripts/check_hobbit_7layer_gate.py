#!/usr/bin/env python3
"""Validate Hobbit 7-layer acceptance gate artifact.

Exit codes:
- 0: PASS
- 1: artifact malformed or status != PASS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_KEYS = {
    "gate",
    "schema_version",
    "status",
    "suite",
    "checked_at_utc",
    "checks",
    "layers",
}

REQUIRED_CHECKS = {"acceptance_smoke", "schema", "data_structure", "graph_accuracy"}
REQUIRED_LAYERS = {"47", "49", "50", "51"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Hobbit 7-layer acceptance gate artifact")
    parser.add_argument(
        "--artifact",
        default="gates/hobbit_7layer_acceptance_gate.json",
        help="Path to gate JSON artifact",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_path = Path(args.artifact)

    if not artifact_path.exists():
        print(f"FAIL: gate artifact not found at {artifact_path}")
        return 1

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL: artifact is not valid JSON: {exc}")
        return 1

    missing = sorted(REQUIRED_KEYS - payload.keys())
    if missing:
        print(f"FAIL: artifact missing required keys: {', '.join(missing)}")
        return 1

    if payload.get("gate") != "hobbit-7layer":
        print("FAIL: artifact gate must be 'hobbit-7layer'")
        return 1

    status = str(payload.get("status", "")).upper()
    if status != "PASS":
        print(f"FAIL: Hobbit 7-layer gate status is {status or 'UNKNOWN'} (must be PASS)")
        return 1

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        print("FAIL: artifact checks must be an object")
        return 1
    missing_checks = sorted(REQUIRED_CHECKS - checks.keys())
    if missing_checks:
        print(f"FAIL: artifact checks missing required keys: {', '.join(missing_checks)}")
        return 1
    failed_checks = [k for k in sorted(REQUIRED_CHECKS) if str(checks.get(k, "")).upper() != "PASS"]
    if failed_checks:
        print(f"FAIL: required gate checks failed: {', '.join(failed_checks)}")
        return 1

    layers = payload.get("layers")
    if not isinstance(layers, dict):
        print("FAIL: artifact layers must be an object")
        return 1
    missing_layers = sorted(REQUIRED_LAYERS - layers.keys())
    if missing_layers:
        print(f"FAIL: artifact layers missing required keys: {', '.join(missing_layers)}")
        return 1
    failed_layers = [k for k in sorted(REQUIRED_LAYERS) if str(layers.get(k, "")).upper() != "PASS"]
    if failed_layers:
        print(f"FAIL: required layer checks failed: {', '.join(failed_layers)}")
        return 1

    print("PASS: Hobbit 7-layer gate artifact is valid and status=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
