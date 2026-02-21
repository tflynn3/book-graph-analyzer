"""Utilities for checking required GitHub Actions secrets in local env.

Used to debug agentic workflow failures like:
- "Secret Verification Failed"
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SECRET_PATTERN = re.compile(r"secrets\.([A-Z0-9_]+)")


def extract_required_secrets(workflow_path: str | Path) -> list[str]:
    """Extract unique secret names referenced in a workflow YAML file.

    This is a regex-based parser that looks for `secrets.NAME` usages.
    """
    content = Path(workflow_path).read_text(encoding="utf-8")
    names = sorted(set(_SECRET_PATTERN.findall(content)))
    return names


def parse_env_file(env_file: str | Path) -> dict[str, str]:
    """Parse a .env-style file into key/value mapping."""
    env: dict[str, str] = {}
    path = Path(env_file)
    if not path.exists():
        return env

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def check_secrets_available(
    required: list[str],
    env_overrides: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (present, missing) for required secret names.

    Checks process environment first, then env_overrides.
    """
    present: list[str] = []
    missing: list[str] = []

    for name in required:
        val = os.environ.get(name)
        if not val and env_overrides:
            val = env_overrides.get(name)

        if val:
            present.append(name)
        else:
            missing.append(name)

    return present, missing
