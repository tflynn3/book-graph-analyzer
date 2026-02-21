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
    def severity(self) -> str:
        """Classify failure severity for triage dashboards.

        Levels:
          - critical: explicit secret-verification failure + missing secrets
          - high: missing secrets (without explicit phrase)
          - medium: explicit secret-verification phrase but no missing secrets
          - low: no secret-related indicators in current analyzer
        """
        if self.secret_verification_failed and self.missing_secrets:
            return "critical"
        if self.missing_secrets:
            return "high"
        if self.secret_verification_failed:
            return "medium"
        return "low"

    def to_row(self, issue_number: int, issue_title: str) -> dict[str, str]:
        """Flatten analysis for CSV/report export."""
        return {
            "issue_number": str(issue_number),
            "issue_title": issue_title,
            "run_id": str(self.run_id or ""),
            "run_url": str(self.run_url or ""),
            "severity": self.severity,
            "secret_verification_failed": "true" if self.secret_verification_failed else "false",
            "required_secrets": ";".join(self.required_secrets),
            "present_secrets": ";".join(self.present_secrets),
            "missing_secrets": ";".join(self.missing_secrets),
            "diagnosis": self.summary,
        }

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


def build_remediation_report(analysis: FailureAnalysis, issue_ref: str = "") -> str:
    """Build a markdown remediation report from failure analysis output."""
    lines: list[str] = []
    lines.append("# Workflow Failure Remediation Report")
    if issue_ref:
        lines.append("")
        lines.append(f"Issue: {issue_ref}")
    lines.append("")
    lines.append(f"Run URL: {analysis.run_url or 'n/a'}")
    lines.append(f"Run ID: {analysis.run_id or 'n/a'}")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append(analysis.summary)
    lines.append("")
    lines.append("## Required secrets")
    if analysis.required_secrets:
        for s in analysis.required_secrets:
            status = "present" if s in analysis.present_secrets else "missing"
            lines.append(f"- {s}: **{status}**")
    else:
        lines.append("- none detected in workflow")

    lines.append("")
    lines.append("## Suggested actions")
    if analysis.missing_secrets:
        lines.append("1. Add missing secrets in repository settings (Settings -> Secrets and variables -> Actions).")
        for s in analysis.missing_secrets:
            lines.append(f"   - {s}")
        lines.append("2. Re-run the failed workflow after adding secrets.")
        lines.append("3. Verify secret scope (repo/org/environment) matches workflow context.")
    else:
        lines.append("1. Secrets appear present. Investigate token scope/permissions and workflow logs.")
        lines.append("2. Use `gh aw logs <run-url>` and `gh aw audit <run-id>` for deeper trace.")

    return "\n".join(lines)
