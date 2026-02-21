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
