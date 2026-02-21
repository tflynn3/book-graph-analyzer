"""Tests for workflow failure analysis helpers and CLI command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.ops.workflow_failure import (
    parse_run_url,
    detect_secret_verification_failure,
    analyze_failure_from_issue_text,
)
from book_graph_analyzer.ops.gh_issue import IssueData


ISSUE_TEXT = """
### Workflow Failure

**Run URL:** https://github.com/tflynn3/book-graph-analyzer/actions/runs/22260153462

**⚠️ Secret Verification Failed**: The workflow's secret validation step failed.
"""


def test_parse_run_url():
    url, run_id = parse_run_url(ISSUE_TEXT)
    assert url is not None
    assert run_id == "22260153462"


def test_detect_secret_verification_failure_true():
    assert detect_secret_verification_failure(ISSUE_TEXT) is True


def test_detect_secret_verification_failure_false():
    text = "workflow failed due to timeout"
    assert detect_secret_verification_failure(text) is False


def test_analyze_failure_from_issue_text(tmp_path: Path):
    wf = tmp_path / "wf.yml"
    wf.write_text(
        """
jobs:
  a:
    steps:
      - run: echo hi
        env:
          COPILOT_GITHUB_TOKEN: ${{ secrets.COPILOT_GITHUB_TOKEN }}
          GH_AW_GITHUB_TOKEN: ${{ secrets.GH_AW_GITHUB_TOKEN }}
""",
        encoding="utf-8",
    )

    envf = tmp_path / ".env"
    envf.write_text("GH_AW_GITHUB_TOKEN=abc\n", encoding="utf-8")

    analysis = analyze_failure_from_issue_text(ISSUE_TEXT, workflow_path=wf, env_file=envf)
    assert analysis.secret_verification_failed is True
    assert analysis.run_id == "22260153462"
    assert "GH_AW_GITHUB_TOKEN" in analysis.present_secrets
    assert "COPILOT_GITHUB_TOKEN" in analysis.missing_secrets


def test_cli_workflow_analyze_failure(tmp_path: Path):
    wf = tmp_path / "wf.yml"
    wf.write_text(
        """
jobs:
  a:
    steps:
      - run: echo hi
        env:
          COPILOT_GITHUB_TOKEN: ${{ secrets.COPILOT_GITHUB_TOKEN }}
""",
        encoding="utf-8",
    )

    issue_file = tmp_path / "issue.md"
    issue_file.write_text(ISSUE_TEXT, encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-analyze-failure",
            "--issue-file",
            str(issue_file),
            "--workflow",
            str(wf),
        ],
    )
    assert res.exit_code != 0
    assert "Workflow Failure Analysis" in res.output
    assert "missing" in res.output.lower()


def test_cli_workflow_analyze_failure_success_with_env_file(tmp_path: Path):
    wf = tmp_path / "wf.yml"
    wf.write_text(
        """
jobs:
  a:
    steps:
      - run: echo hi
        env:
          COPILOT_GITHUB_TOKEN: ${{ secrets.COPILOT_GITHUB_TOKEN }}
""",
        encoding="utf-8",
    )

    issue_file = tmp_path / "issue.md"
    issue_file.write_text(ISSUE_TEXT, encoding="utf-8")

    envf = tmp_path / ".env"
    envf.write_text("COPILOT_GITHUB_TOKEN=abc\n", encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-analyze-failure",
            "--issue-file",
            str(issue_file),
            "--workflow",
            str(wf),
            "--env-file",
            str(envf),
        ],
    )
    assert res.exit_code == 0
    assert "Diagnosis" in res.output


def test_cli_workflow_analyze_failure_issue(monkeypatch, tmp_path: Path):
    wf = tmp_path / "wf.yml"
    wf.write_text(
        """
jobs:
  a:
    steps:
      - run: echo hi
        env:
          COPILOT_GITHUB_TOKEN: ${{ secrets.COPILOT_GITHUB_TOKEN }}
""",
        encoding="utf-8",
    )

    def fake_fetch(issue_number: int):
        assert issue_number == 41
        return IssueData(number=41, title="failed", body=ISSUE_TEXT, url="https://example")

    monkeypatch.setattr("book_graph_analyzer.ops.fetch_issue_via_gh", fake_fetch)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-analyze-failure-issue",
            "--issue",
            "41",
            "--workflow",
            str(wf),
        ],
    )
    # Missing token -> non-zero + analysis printed
    assert res.exit_code != 0
    assert "Issue #41" in res.output
    assert "Workflow Failure Analysis" in res.output
