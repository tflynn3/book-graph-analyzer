"""Tests for gh issue helper wrapper."""

from __future__ import annotations

import json
from types import SimpleNamespace

from book_graph_analyzer.ops.gh_issue import fetch_issue_via_gh


def test_fetch_issue_via_gh_success(monkeypatch):
    payload = {
        "number": 41,
        "title": "failed run",
        "body": "body text",
        "url": "https://github.com/x/y/issues/41",
    }

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    issue = fetch_issue_via_gh(41)
    assert issue is not None
    assert issue.number == 41
    assert issue.title == "failed run"


def test_fetch_issue_via_gh_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("subprocess.run", fake_run)
    issue = fetch_issue_via_gh(41)
    assert issue is None
