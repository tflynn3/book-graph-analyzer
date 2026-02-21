"""Workflow failure analysis helpers for agentic-workflows issues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .workflow_secrets import extract_required_secrets, parse_env_file, check_secrets_available

_RUN_URL_RE = re.compile(r"https://github\.com/[^\s)]+/actions/runs/(\d+)")


@dataclass
class FailureAnalysis:
    run_id: Optional[str]
    run_url: Optional[str]
    secret_verification_failed: bool
    required_secrets: list[str]
    present_secrets: list[str]
    missing_secrets: list[str]

    @property
    def summary(self) -> str:
        if self.secret_verification_failed and self.missing_secrets:
            return (
                f"Secret verification failure likely caused by missing secrets: "
                f"{', '.join(self.missing_secrets)}"
            )
        if self.secret_verification_failed:
            return "Secret verification failure detected. Verify repository/environment secret scopes."
        return "No explicit secret verification failure phrase detected."


def parse_run_url(issue_markdown: str) -> tuple[Optional[str], Optional[str]]:
    """Extract workflow run URL and run ID from issue markdown."""
    m = _RUN_URL_RE.search(issue_markdown)
    if not m:
        return None, None
    run_id = m.group(1)
    return m.group(0), run_id


def detect_secret_verification_failure(issue_markdown: str) -> bool:
    """Return True if issue text indicates secret verification failure."""
    text = issue_markdown.lower()
    return "secret verification failed" in text or "validate" in text and "secret" in text


def analyze_failure_from_issue_text(
    issue_markdown: str,
    workflow_path: str | Path,
    env_file: str | Path | None = None,
) -> FailureAnalysis:
    """Analyze a failed-workflow issue body and produce a local diagnosis."""
    run_url, run_id = parse_run_url(issue_markdown)
    secret_failed = detect_secret_verification_failure(issue_markdown)

    required = extract_required_secrets(workflow_path)
    overrides = parse_env_file(env_file) if env_file else {}
    present, missing = check_secrets_available(required, env_overrides=overrides)

    return FailureAnalysis(
        run_id=run_id,
        run_url=run_url,
        secret_verification_failed=secret_failed,
        required_secrets=required,
        present_secrets=present,
        missing_secrets=missing,
    )


def analyze_failure_from_issue_file(
    issue_file: str | Path,
    workflow_path: str | Path,
    env_file: str | Path | None = None,
) -> FailureAnalysis:
    """Load issue markdown from file and analyze it."""
    text = Path(issue_file).read_text(encoding="utf-8")
    return analyze_failure_from_issue_text(text, workflow_path=workflow_path, env_file=env_file)
