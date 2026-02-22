"""Tests for workflow failure analysis helpers and CLI command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.ops.workflow_failure import (
    parse_run_url,
    detect_secret_verification_failure,
    analyze_failure_from_issue_text,
    build_remediation_report,
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
    assert analysis.severity == "critical"


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


def test_cli_workflow_analyze_open_failures(monkeypatch, tmp_path: Path):
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

    issues = [
        IssueData(number=40, title="[agentics] Failed runs", body="parent", url="u40"),
        IssueData(number=41, title="[agentics] Architecture failed", body=ISSUE_TEXT, url="u41"),
    ]

    monkeypatch.setattr("book_graph_analyzer.ops.list_open_issues_via_gh", lambda label, limit: issues)

    out_csv = tmp_path / "analysis.csv"
    out_json = tmp_path / "analysis.json"
    out_md = tmp_path / "analysis.md"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-analyze-open-failures",
            "--workflow",
            str(wf),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )
    # Missing secret from issue 41 should fail command
    assert res.exit_code != 0
    assert "Issue #41" in res.output
    assert "Workflow Failure Analysis" in res.output
    assert out_csv.exists()
    csv_text = out_csv.read_text(encoding="utf-8")
    assert "issue_number" in csv_text
    assert "severity" in csv_text
    assert "41" in csv_text

    assert out_json.exists()
    json_text = out_json.read_text(encoding="utf-8")
    assert "summary" in json_text
    assert "severity_counts" in json_text
    assert "unresolved" in json_text
    assert "unresolved_by_severity" in json_text
    assert "issue_number" in json_text
    assert "severity" in json_text
    assert "41" in json_text

    assert out_md.exists()
    md_text = out_md.read_text(encoding="utf-8")
    assert "# Open Workflow Failure Analysis" in md_text
    assert "| Issue | Run ID | Severity |" in md_text
    assert "#41" in md_text


def test_build_remediation_report_contains_actions(tmp_path: Path):
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
    analysis = analyze_failure_from_issue_text(ISSUE_TEXT, workflow_path=wf)
    report = build_remediation_report(analysis, issue_ref="#41 failed")
    assert "Remediation Report" in report
    assert "COPILOT_GITHUB_TOKEN" in report
    assert "Suggested actions" in report


def test_cli_workflow_remediation_report_issue(monkeypatch, tmp_path: Path):
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

    monkeypatch.setattr(
        "book_graph_analyzer.ops.fetch_issue_via_gh",
        lambda issue_number: IssueData(number=41, title="failed", body=ISSUE_TEXT, url="u41"),
    )

    out = tmp_path / "report.md"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-remediation-report",
            "--issue",
            "41",
            "--workflow",
            str(wf),
            "--out",
            str(out),
        ],
    )
    assert res.exit_code != 0  # missing secret -> abort
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Workflow Failure Remediation Report" in text


def test_cli_workflow_post_open_failures_summary(monkeypatch, tmp_path: Path):
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

    envf = tmp_path / ".env"
    envf.write_text("COPILOT_GITHUB_TOKEN=abc\n", encoding="utf-8")

    issues = [
        IssueData(number=40, title="[agentics] Failed runs", body="parent", url="u40"),
        IssueData(number=41, title="[agentics] Architecture failed", body=ISSUE_TEXT, url="u41"),
    ]
    monkeypatch.setattr("book_graph_analyzer.ops.list_open_issues_via_gh", lambda label, limit: issues)

    captured = {"issue": None, "body": ""}

    def fake_post(issue_number: int, body: str):
        captured["issue"] = issue_number
        captured["body"] = body
        return True

    monkeypatch.setattr("book_graph_analyzer.ops.post_issue_comment_via_gh", fake_post)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-post-open-failures-summary",
            "--parent-issue",
            "40",
            "--workflow",
            str(wf),
            "--env-file",
            str(envf),
        ],
    )
    assert res.exit_code == 0
    assert captured["issue"] == 40
    assert "Automated Failure Summary" in captured["body"]
    assert "Severity" in captured["body"]
    assert "Severity breakdown" in captured["body"]
    assert "#41" in captured["body"]


def test_cli_workflow_post_diagnosis(monkeypatch, tmp_path: Path):
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

    monkeypatch.setattr(
        "book_graph_analyzer.ops.fetch_issue_via_gh",
        lambda issue_number: IssueData(number=41, title="failed", body=ISSUE_TEXT, url="u41"),
    )
    monkeypatch.setattr("book_graph_analyzer.ops.post_issue_comment_via_gh", lambda issue_number, body: True)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-post-diagnosis",
            "--issue",
            "41",
            "--workflow",
            str(wf),
        ],
    )
    # Missing secret -> command still aborts non-zero after posting
    assert res.exit_code != 0
    assert "Posted diagnosis report" in res.output


def test_cli_workflow_close_resolved_failures_dry_run(monkeypatch, tmp_path: Path):
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

    envf = tmp_path / ".env"
    envf.write_text("COPILOT_GITHUB_TOKEN=abc\n", encoding="utf-8")

    # Parent tracker + one actionable issue
    issues = [
        IssueData(number=40, title="[agentics] Failed runs", body="parent", url="u40"),
        IssueData(number=41, title="[agentics] Architecture failed", body="Timeout only, no secret errors", url="u41"),
    ]
    monkeypatch.setattr("book_graph_analyzer.ops.list_open_issues_via_gh", lambda label, limit: issues)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-close-resolved-failures",
            "--workflow",
            str(wf),
            "--env-file",
            str(envf),
            "--dry-run",
        ],
    )
    assert res.exit_code == 0
    assert "would close issue #41" in res.output


def test_cli_workflow_close_resolved_failures_real_close(monkeypatch, tmp_path: Path):
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

    envf = tmp_path / ".env"
    envf.write_text("COPILOT_GITHUB_TOKEN=abc\n", encoding="utf-8")

    issues = [IssueData(number=41, title="[agentics] Architecture failed", body="No secret validation failure.", url="u41")]
    monkeypatch.setattr("book_graph_analyzer.ops.list_open_issues_via_gh", lambda label, limit: issues)
    monkeypatch.setattr("book_graph_analyzer.ops.close_issue_via_gh", lambda issue_number, reason="completed": True)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-close-resolved-failures",
            "--workflow",
            str(wf),
            "--env-file",
            str(envf),
        ],
    )
    assert res.exit_code == 0
    assert "Closed issue #41" in res.output
