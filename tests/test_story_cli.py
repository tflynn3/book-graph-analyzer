import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main


def test_story_group_registered():
    assert "story" in main.commands
    assert "init" in main.commands["story"].commands
    assert "plan" in main.commands["story"].commands
    assert "validate" in main.commands["story"].commands
    assert "context" in main.commands["story"].commands
    assert "grow-shadow" in main.commands["story"].commands
    assert "solve" in main.commands["story"].commands
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


def test_shadow_graph_workflow_end_to_end(tmp_path):
    events_path = tmp_path / "events.json"
    events_payload = {
        "events": {
            "1": {"id": "1", "description": "Beren enters Doriath in secret.", "agent": "Beren", "action": "enter", "patient": "Doriath", "era": "First Age", "year": 465},
            "2": {"id": "2", "description": "Luthien sings beneath moonlit beeches.", "agent": "Luthien", "action": "sing", "patient": "song", "era": "First Age", "year": 465},
            "3": {"id": "3", "description": "Thingol sets a perilous bride-price.", "agent": "Thingol", "action": "decree", "patient": "quest", "era": "First Age", "year": 466},
            "4": {"id": "4", "description": "Beren swears an oath and departs.", "agent": "Beren", "action": "swear", "patient": "oath", "era": "First Age", "year": 466},
        },
        "relations": [],
    }
    events_path.write_text(json.dumps(events_payload, indent=2), encoding="utf-8")

    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "genre": "fantasy",
                "premise": "A grounded retelling that explores implied beats between canonical events.",
                "target_chapters": 2,
                "scenes_per_chapter": 2,
                "event_files": [str(events_path)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": ["oath"], "forbidden_terms": ["spaceship"], "style": {}}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "beren-luthien-expanded",
                "chapters": [
                    {"chapter_number": 1, "title": "Ch1", "scenes": [{"scene_id": "ch01-sc01", "goal": "setup", "summary": "Beren enters"}, {"scene_id": "ch01-sc02", "goal": "vow", "summary": "Oath set"}]},
                    {"chapter_number": 2, "title": "Ch2", "scenes": [{"scene_id": "ch02-sc01", "goal": "journey", "summary": "Road"}, {"scene_id": "ch02-sc02", "goal": "trial", "summary": "Trial"}]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    commands = [
        ["story", "context", "--project", "beren-luthien-expanded", "--graph-stats", "--projects-dir", str(tmp_path)],
        ["story", "grow-shadow", "--project", "beren-luthien-expanded", "--auto", "--projects-dir", str(tmp_path)],
        ["story", "solve", "--project", "beren-luthien-expanded", "--projects-dir", str(tmp_path)],
        ["story", "draft", "--project", "beren-luthien-expanded", "--chapter", "1", "--grounded", "--projects-dir", str(tmp_path)],
        ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)],
    ]
    for cmd in commands:
        result = runner.invoke(main, cmd)
        assert result.exit_code == 0, result.output

    assert (proj_dir / "context_stats.json").exists()
    assert (proj_dir / "shadow_graph.json").exists()
    assert (proj_dir / "shadow_candidates.json").exists()
    assert (proj_dir / "shadow_solution.json").exists()
    assert (proj_dir / "chapter_01.md").exists()
    assert (proj_dir / "chapter_01_trace.json").exists()
    assert (proj_dir / "chapter_01_audit.json").exists()

    trace = json.loads((proj_dir / "chapter_01_trace.json").read_text(encoding="utf-8"))
    assert trace["schema_version"] == "chapter-trace-v1"
    assert trace["chapter"] == 1
    assert len(trace["sections"]) == 2

    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == "chapter-audit-v1"
    assert audit["status"] in {"pass", "warn", "fail"}
