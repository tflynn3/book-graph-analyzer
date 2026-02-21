"""Tests for workflow secret-check utilities and CLI command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.ops.workflow_secrets import (
    extract_required_secrets,
    parse_env_file,
    check_secrets_available,
)


def test_extract_required_secrets(tmp_path: Path):
    wf = tmp_path / "wf.yml"
    wf.write_text(
        """
name: test
jobs:
  a:
    steps:
      - run: echo hi
        env:
          GH_TOKEN: ${{ secrets.GH_AW_GITHUB_TOKEN }}
          COPILOT_GITHUB_TOKEN: ${{ secrets.COPILOT_GITHUB_TOKEN }}
      - run: echo bye
        env:
          GH_TOKEN2: ${{ secrets.GH_AW_GITHUB_TOKEN }}
""",
        encoding="utf-8",
    )
    names = extract_required_secrets(wf)
    assert "GH_AW_GITHUB_TOKEN" in names
    assert "COPILOT_GITHUB_TOKEN" in names
    assert names.count("GH_AW_GITHUB_TOKEN") == 1


def test_parse_env_file(tmp_path: Path):
    envf = tmp_path / ".env"
    envf.write_text(
        """
# comment
COPILOT_GITHUB_TOKEN=abc
GH_AW_GITHUB_TOKEN="def"
INVALID_LINE
""",
        encoding="utf-8",
    )
    env = parse_env_file(envf)
    assert env["COPILOT_GITHUB_TOKEN"] == "abc"
    assert env["GH_AW_GITHUB_TOKEN"] == "def"
    assert "INVALID_LINE" not in env


def test_check_secrets_available(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_AW_GITHUB_TOKEN", "token")

    present, missing = check_secrets_available([
        "GH_AW_GITHUB_TOKEN",
        "COPILOT_GITHUB_TOKEN",
    ])
    assert "GH_AW_GITHUB_TOKEN" in present
    assert "COPILOT_GITHUB_TOKEN" in missing


def test_cli_workflow_check_secrets_missing(tmp_path: Path, monkeypatch):
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
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)

    runner = CliRunner()
    res = runner.invoke(main, ["workflow-check-secrets", "--workflow", str(wf)])
    assert res.exit_code != 0
    assert "missing" in res.output.lower()


def test_cli_workflow_check_secrets_present_via_env_file(tmp_path: Path, monkeypatch):
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
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workflow-check-secrets",
            "--workflow",
            str(wf),
            "--env-file",
            str(envf),
        ],
    )
    assert res.exit_code == 0
    assert "all required secrets found" in res.output.lower()
