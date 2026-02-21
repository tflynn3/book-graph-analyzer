"""Tests for gh issue helper wrapper."""

from __future__ import annotations

import json
from types import SimpleNamespace

from book_graph_analyzer.ops.gh_issue import (
    fetch_issue_via_gh,
    list_open_issues_via_gh,
    post_issue_comment_via_gh,
)


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


def test_list_open_issues_via_gh(monkeypatch):
    payload = [
        {"number": 40, "title": "[agentics] Failed runs", "body": "x", "url": "u1"},
        {"number": 41, "title": "[agentics] Architecture failed", "body": "y", "url": "u2"},
    ]

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    issues = list_open_issues_via_gh(label="agentic-workflows", limit=20)
    assert len(issues) == 2
    assert issues[0].number == 40
    assert issues[1].number == 41


def test_post_issue_comment_via_gh_success(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert post_issue_comment_via_gh(41, "report") is True


def test_post_issue_comment_via_gh_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert post_issue_comment_via_gh(41, "report") is False
