import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main


def test_story_group_registered():
    assert "story" in main.commands
    assert "init" in main.commands["story"].commands
    assert "plan" in main.commands["story"].commands
    assert "validate" in main.commands["story"].commands
    assert "draft" in main.commands["story"].commands
    assert "audit" in main.commands["story"].commands


def test_story_init_non_interactive(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "story",
            "init",
            "--name",
            "Mithrandir in the East",
            "--slug",
            "mithrandir-east",
            "--premise",
            "A covert mission beyond Rhun.",
            "--target-chapters",
            "4",
            "--scenes-per-chapter",
            "2",
            "--projects-dir",
            str(tmp_path),
            "--non-interactive",
        ],
    )

    assert result.exit_code == 0, result.output
    proj_dir = tmp_path / "mithrandir-east"
    assert (proj_dir / "project.json").exists()
    assert (proj_dir / "constraints.json").exists()
    assert (proj_dir / "story_bible.md").exists()
    assert (proj_dir / "plan.json").exists()


def test_story_plan_auto_generates_structure(tmp_path):
    runner = CliRunner()
    init = runner.invoke(
        main,
        [
            "story",
            "init",
            "--name",
            "Test Project",
            "--slug",
            "test-proj",
            "--premise",
            "A test premise",
            "--target-chapters",
            "3",
            "--scenes-per-chapter",
            "2",
            "--projects-dir",
            str(tmp_path),
            "--non-interactive",
        ],
    )
    assert init.exit_code == 0, init.output

    plan = runner.invoke(
        main,
        [
            "story",
            "plan",
            "--project",
            "test-proj",
            "--auto",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert plan.exit_code == 0, plan.output

    payload = json.loads((tmp_path / "test-proj" / "plan.json").read_text(encoding="utf-8"))
    assert payload["project_slug"] == "test-proj"
    assert len(payload["chapters"]) == 3
    assert len(payload["chapters"][0]["scenes"]) == 2


def test_story_validate_generates_reports(tmp_path):
    runner = CliRunner()
    for args in (
        [
            "story",
            "init",
            "--name",
            "Validation Project",
            "--slug",
            "val-proj",
            "--premise",
            "Validation premise",
            "--projects-dir",
            str(tmp_path),
            "--non-interactive",
        ],
        ["story", "plan", "--project", "val-proj", "--auto", "--projects-dir", str(tmp_path)],
    ):
        result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output

    validate = runner.invoke(
        main,
        ["story", "validate", "--project", "val-proj", "--projects-dir", str(tmp_path)],
    )
    assert validate.exit_code == 0, validate.output

    proj_dir = tmp_path / "val-proj"
    assert (proj_dir / "validation_report.json").exists()
    assert (proj_dir / "validation_report.md").exists()
    report = json.loads((proj_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert report["project_slug"] == "val-proj"
    assert report["status"] in {"pass", "fail"}


def test_story_draft_enforces_required_terms(tmp_path):
    runner = CliRunner()
    setup = runner.invoke(
        main,
        [
            "story",
            "init",
            "--name",
            "Required Terms Project",
            "--slug",
            "required-terms-proj",
            "--premise",
            "A constrained grounded draft.",
            "--projects-dir",
            str(tmp_path),
            "--non-interactive",
        ],
    )
    assert setup.exit_code == 0, setup.output

    constraints_path = tmp_path / "required-terms-proj" / "constraints.json"
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    constraints["required_elements"] = ["Thingol", "Tol-in-Gaurhoth confrontation"]
    constraints_path.write_text(json.dumps(constraints, indent=2), encoding="utf-8")

    draft = runner.invoke(
        main,
        [
            "story",
            "draft",
            "--project",
            "required-terms-proj",
            "--chapter",
            "1",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert draft.exit_code == 0, draft.output

    chapter_path = tmp_path / "required-terms-proj" / "chapters" / "chapter-01.md"
    chapter_text = chapter_path.read_text(encoding="utf-8")
    assert "Thingol" in chapter_text
    assert "Tol-in-Gaurhoth confrontation" in chapter_text

    audit = runner.invoke(
        main,
        [
            "story",
            "audit",
            "--project",
            "required-terms-proj",
            "--chapter",
            "1",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert audit.exit_code == 0, audit.output
    audit_report = json.loads(
        (tmp_path / "required-terms-proj" / "chapters" / "chapter-01.audit.json").read_text(encoding="utf-8")
    )
    assert audit_report["status"] == "PASS"
    assert audit_report["summary"]["required_terms_missing"] == 0


def test_story_draft_fails_when_required_terms_never_appear(tmp_path, monkeypatch):
    from book_graph_analyzer import story_cli as story_module

    runner = CliRunner()
    setup = runner.invoke(
        main,
        [
            "story",
            "init",
            "--name",
            "Failing Required Terms",
            "--slug",
            "required-terms-fail",
            "--premise",
            "A constrained grounded draft.",
            "--projects-dir",
            str(tmp_path),
            "--non-interactive",
        ],
    )
    assert setup.exit_code == 0, setup.output

    constraints_path = tmp_path / "required-terms-fail" / "constraints.json"
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    constraints["required_elements"] = ["Melian"]
    constraints["enforcement"] = {"required_terms": True, "max_retries": 2}
    constraints_path.write_text(json.dumps(constraints, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        story_module,
        "_generate_grounded_chapter_text",
        lambda project, constraints, chapter_number: "# Chapter\n\nNo required term appears here.\n",
    )

    draft = runner.invoke(
        main,
        [
            "story",
            "draft",
            "--project",
            "required-terms-fail",
            "--chapter",
            "1",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert draft.exit_code != 0
    assert "failed required-term enforcement" in draft.output
    assert "Missing required terms" in draft.output
