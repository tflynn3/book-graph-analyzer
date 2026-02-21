"""Helpers to read GitHub issue metadata via gh CLI."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class IssueData:
    number: int
    title: str
    body: str
    url: str


def fetch_issue_via_gh(issue_number: int) -> Optional[IssueData]:
    """Fetch issue body/title/url using `gh issue view --json ...`.

    Returns None if gh is unavailable or call fails.
    """
    try:
        proc = subprocess.run(
            [
                "gh",
                "issue",
                "view",
                str(issue_number),
                "--json",
                "number,title,body,url",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout or "{}")
        return IssueData(
            number=int(data.get("number", issue_number)),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            url=str(data.get("url", "")),
        )
    except Exception:
        return None


def list_open_issues_via_gh(label: str = "", limit: int = 20) -> list[IssueData]:
    """List open issues via gh CLI with optional label filter."""
    args = [
        "gh",
        "issue",
        "list",
        "--state",
        "open",
        "--limit",
        str(limit),
        "--json",
        "number,title,body,url",
    ]
    if label:
        args += ["--label", label]

    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            return []
        arr = json.loads(proc.stdout or "[]")
        out: list[IssueData] = []
        for d in arr:
            out.append(
                IssueData(
                    number=int(d.get("number", 0)),
                    title=str(d.get("title", "")),
                    body=str(d.get("body", "")),
                    url=str(d.get("url", "")),
                )
            )
        return out
    except Exception:
        return []


def post_issue_comment_via_gh(issue_number: int, body: str) -> bool:
    """Post a comment to an issue via gh CLI.

    Returns True on success.
    """
    try:
        proc = subprocess.run(
            ["gh", "issue", "comment", str(issue_number), "--body", body],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False
