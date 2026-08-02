import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main


class _FakeStoryShadowGraph:
    def __init__(self, story_id: str):
        self.story_id = story_id

    def extract_delta_from_scene(self, scene_text: str, characters: list[str], scene_id: str, chapter_num: int = 0, scene_num: int = 0):
        from book_graph_analyzer.generate.shadow.models import StateDelta

        return StateDelta(
            story_id=self.story_id,
            scene_id=scene_id,
            character_updates={name: {} for name in characters},
            scene_summary=f"chapter {chapter_num} scene {scene_num}",
            chapter_num=chapter_num,
            scene_num=scene_num,
        )

    def commit_state_delta(self, _delta):
        return None


class _FakeStoryGenerationWriter:
    def __init__(self):
        self.driver = None
        self.story_writes = []
        self.scene_writes = []

    def write_story(self, story):
        self.story_writes.append(story)
        return {"stories": 1}

    def write_scene(self, scene, chapter_id: str):
        self.scene_writes.append((scene, chapter_id))
        return {"scenes": 1}


class _EchoStorySceneGenerator:
    def __init__(self, shadow_graph=None):
        self.shadow_graph = shadow_graph
        self.driver = None

    def load_world_bible(self, _path):
        return None

    def generate_scene(self, **kwargs):
        from book_graph_analyzer.generate.models import GenerationStatus, Scene, SceneScores

        text = (
            f"{kwargs['scene_goal']} "
            f"At {kwargs['place']}, {', '.join(kwargs['characters'])} move within the tale's shadow."
        )
        return Scene(
            id="temp",
            number=int(kwargs.get("scene_num", 0) or 0),
            text=text,
            summary=str(kwargs["scene_goal"]),
            characters=list(kwargs["characters"]),
            places=[kwargs["place"]],
            objects=list(kwargs.get("objects") or []),
            scores=SceneScores(lore_score=0.9, style_score=0.8, narrative_score=0.85, consistency_score=0.9, overall=0.86),
            status=GenerationStatus.DRAFT,
            model_used="fake-scene-generator",
            pipeline_stages_run=["drafter"],
        )


class _NoAnchorStorySceneGenerator(_EchoStorySceneGenerator):
    def generate_scene(self, **kwargs):
        from book_graph_analyzer.generate.models import GenerationStatus, Scene, SceneScores

        return Scene(
            id="temp",
            number=int(kwargs.get("scene_num", 0) or 0),
            text="No required canon anchors appear here.",
            summary=str(kwargs["scene_goal"]),
            characters=list(kwargs["characters"]),
            places=[kwargs["place"]],
            scores=SceneScores(lore_score=0.9, style_score=0.8, narrative_score=0.85, consistency_score=0.9, overall=0.86),
            status=GenerationStatus.DRAFT,
            model_used="fake-scene-generator",
            pipeline_stages_run=["drafter"],
        )


class _FutureLeakStorySceneGenerator(_EchoStorySceneGenerator):
    def generate_scene(self, **kwargs):
        from book_graph_analyzer.generate.models import GenerationStatus, Scene, SceneScores

        return Scene(
            id="temp",
            number=int(kwargs.get("scene_num", 0) or 0),
            text="Beren stood in Doriath, and Bilbo was beside him in counsel.",
            summary=str(kwargs["scene_goal"]),
            characters=list(kwargs["characters"]),
            places=[kwargs["place"]],
            scores=SceneScores(lore_score=0.9, style_score=0.8, narrative_score=0.85, consistency_score=0.9, overall=0.86),
            status=GenerationStatus.DRAFT,
            model_used="future-leak-generator",
            pipeline_stages_run=["drafter"],
        )


def test_story_group_registered():
    assert "story" in main.commands
    assert "init" in main.commands["story"].commands
    assert "plan" in main.commands["story"].commands
    assert "validate" in main.commands["story"].commands
    assert "context" in main.commands["story"].commands
    assert "grow-shadow" in main.commands["story"].commands
    assert "sample-shadow" in main.commands["story"].commands
    assert "score-shadow" in main.commands["story"].commands
    assert "select-shadow" in main.commands["story"].commands
    assert "solve" in main.commands["story"].commands
    assert "draft" in main.commands["story"].commands
    assert "audit" in main.commands["story"].commands
    assert "beats" in main.commands["story"].commands
    assert "expand" in main.commands["story"].commands["beats"].commands
    assert "validate" in main.commands["story"].commands["beats"].commands
    assert "show" in main.commands["story"].commands["beats"].commands
    assert "clean" in main.commands["story"].commands["beats"].commands


def test_story_beats_validate_show_clean_flow_and_scoping(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beats-proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "Beats", "slug": "beats-proj", "target_chapters": 2, "scenes_per_chapter": 2}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_beats.json").write_text(
        json.dumps(
            {
                "schema_version": "shadow-beats-v1",
                "project_slug": "beats-proj",
                "beats": [
                    {"beat_id": "ch01-sc01-b01-aa", "position": 1, "beat_type": "setup", "cause_refs": [], "failed_constraints": []},
                    {"beat_id": "ch01-sc02-b02-bb", "position": 2, "beat_type": "pivot", "cause_refs": ["missing-ref"], "failed_constraints": []},
                    {"beat_id": "ch02-sc01-b03-cc", "position": 3, "beat_type": "confrontation", "cause_refs": ["ch01-sc02-b02-bb"], "failed_constraints": ["forbidden:spaceship"]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    v = runner.invoke(
        main,
        ["story", "beats", "validate", "--project", "beats-proj", "--chapter", "1", "--projects-dir", str(tmp_path)],
    )
    assert v.exit_code == 0, v.output
    report = json.loads((proj_dir / "shadow_beats_validation.json").read_text(encoding="utf-8"))
    assert report["summary"]["beats"] == 2
    assert report["summary"]["errors"] == 1
    assert report["summary"]["warnings"] == 0

    show = runner.invoke(
        main,
        ["story", "beats", "show", "--project", "beats-proj", "--scene", "ch02-sc01", "--projects-dir", str(tmp_path)],
    )
    assert show.exit_code == 0, show.output
    assert "count=1" in show.output
    assert "Per-scene counts: ch02-sc01:1" in show.output
    assert "warnings=1" in runner.invoke(
        main,
        ["story", "beats", "validate", "--project", "beats-proj", "--scene", "ch02-sc01", "--projects-dir", str(tmp_path)],
    ).output

    dry = runner.invoke(
        main,
        ["story", "beats", "clean", "--project", "beats-proj", "--chapter", "1", "--dry-run", "--projects-dir", str(tmp_path)],
    )
    assert dry.exit_code == 0, dry.output
    unchanged = json.loads((proj_dir / "shadow_beats.json").read_text(encoding="utf-8"))
    assert len(unchanged["beats"]) == 3

    clean = runner.invoke(
        main,
        ["story", "beats", "clean", "--project", "beats-proj", "--chapter", "1", "--projects-dir", str(tmp_path)],
    )
    assert clean.exit_code == 0, clean.output
    changed = json.loads((proj_dir / "shadow_beats.json").read_text(encoding="utf-8"))
    assert len(changed["beats"]) == 1
    assert changed["beats"][0]["beat_id"].startswith("ch02-")


def test_story_beats_validate_strict_exit_codes(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beats-strict"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "Strict", "slug": "beats-strict", "target_chapters": 1, "scenes_per_chapter": 1}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_beats.json").write_text(
        json.dumps(
            {
                "beats": [
                    {"beat_id": "ch01-sc01-b01-a", "position": 1, "beat_type": "setup", "cause_refs": [], "failed_constraints": ["forbidden:x"]}
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    non_strict = runner.invoke(main, ["story", "beats", "validate", "--project", "beats-strict", "--projects-dir", str(tmp_path)])
    assert non_strict.exit_code == 0, non_strict.output

    strict_errors_only = runner.invoke(
        main,
        ["story", "beats", "validate", "--project", "beats-strict", "--strict", "--projects-dir", str(tmp_path)],
    )
    assert strict_errors_only.exit_code == 0, strict_errors_only.output

    strict_warn = runner.invoke(
        main,
        [
            "story",
            "beats",
            "validate",
            "--project",
            "beats-strict",
            "--strict",
            "--strict-warnings",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert strict_warn.exit_code != 0
    assert "Strict validation failed" in strict_warn.output


def test_story_beats_validate_backward_compatible_with_legacy_artifact(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "legacy-beats"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(json.dumps({"name": "Legacy", "slug": "legacy-beats"}, indent=2), encoding="utf-8")
    (proj_dir / "shadow_beats.json").write_text(
        json.dumps(
            {
                "schema_version": "shadow-beats-v1",
                "project_slug": "legacy-beats",
                "beats": [
                    {"beat_id": "ch01-sc01-b01-aa", "position": 1, "beat_type": "setup", "cause_refs": [], "failed_constraints": []},
                    {"beat_id": "ch01-sc01-b02-bb", "position": 2, "beat_type": "pivot", "cause_refs": ["ch01-sc01-b01-aa"], "failed_constraints": []},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    res = runner.invoke(main, ["story", "beats", "validate", "--project", "legacy-beats", "--projects-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    report = json.loads((proj_dir / "shadow_beats_validation.json").read_text(encoding="utf-8"))
    codes = {i["code"] for i in report["issues"]}
    assert "CANON_GROUNDING_WEAK" not in codes


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
    project = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    assert project["timeline"]["story_era"] == "Third Age"
    assert project["timeline"]["story_year"] == 3018


def test_story_context_and_grow_shadow_hard_gate_future_characters(tmp_path):
    runner = CliRunner()
    events_path = tmp_path / "mixed_events.json"
    events_path.write_text(
        json.dumps(
            {
                "events": {
                    "1": {"id": "1", "description": "Beren enters Doriath", "agent": "Beren", "action": "enter", "era": "First Age", "year": 465},
                    "2": {"id": "2", "description": "Luthien sings in the woods", "agent": "Luthien", "action": "sing", "era": "First Age", "year": 465},
                    "3": {"id": "3", "description": "Thingol sets the quest", "agent": "Thingol", "action": "decree", "era": "First Age", "year": 466},
                    "4": {"id": "4", "description": "Bilbo finds the Ring", "agent": "Bilbo", "action": "find", "era": "Third Age", "year": 2941},
                    "5": {"id": "5", "description": "Frodo leaves the Shire", "agent": "Frodo", "action": "leave", "era": "Third Age", "year": 3018},
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "premise": "A First Age retelling.",
                "target_chapters": 1,
                "scenes_per_chapter": 2,
                "event_files": [str(events_path)],
                "timeline": {"story_era": "First Age", "story_year": 466},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": [], "style": {}}, indent=2), encoding="utf-8")
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "beren-luthien-expanded",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "title": "Ch1",
                        "scenes": [
                            {"scene_id": "ch01-sc01", "goal": "setup", "summary": "Beren enters Doriath"},
                            {"scene_id": "ch01-sc02", "goal": "response", "summary": "Thingol answers"},
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for cmd in (
        ["story", "context", "--project", "beren-luthien-expanded", "--graph-stats", "--projects-dir", str(tmp_path)],
        ["story", "grow-shadow", "--project", "beren-luthien-expanded", "--auto", "--projects-dir", str(tmp_path)],
    ):
        result = runner.invoke(main, cmd)
        assert result.exit_code == 0, result.output

    context = json.loads((proj_dir / "context_stats.json").read_text(encoding="utf-8"))
    assert "Bilbo" in context["timeline"]["future_guardrail_entities"]

    candidates = json.loads((proj_dir / "shadow_candidates.json").read_text(encoding="utf-8"))
    all_chars = {
        char
        for row in candidates["candidates"]
        for char in row["shadow_event"]["characters"]
    }
    assert "Bilbo" not in all_chars
    assert "Frodo" not in all_chars
    assert all_chars


def test_story_context_prefers_local_seed_neighborhood_over_same_era_noise(tmp_path):
    runner = CliRunner()
    events_path = tmp_path / "first_age_events.json"
    events_path.write_text(
        json.dumps(
            {
                "events": {
                    "1": {"id": "1", "description": "Beren enters Doriath", "agent": "Beren", "action": "enter", "era": "First Age", "year": 465},
                    "2": {"id": "2", "description": "Luthien sings before Thingol", "agent": "Luthien", "patient": "Thingol", "action": "sing", "era": "First Age", "year": 465},
                    "3": {"id": "3", "description": "Thingol sends Beren on the quest", "agent": "Thingol", "patient": "Beren", "action": "decree", "era": "First Age", "year": 466},
                    "4": {"id": "4", "description": "Turin slays Glaurung", "agent": "Turin", "patient": "Glaurung", "action": "slay", "era": "First Age", "year": 499},
                    "5": {"id": "5", "description": "Nienor mourns Turin", "agent": "Nienor", "patient": "Turin", "action": "mourn", "era": "First Age", "year": 499},
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "premise": "A focused retelling of Beren and Luthien.",
                "event_files": [str(events_path)],
                "timeline": {"story_era": "First Age", "story_year": 466},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": []}, indent=2), encoding="utf-8")

    result = runner.invoke(
        main,
        ["story", "context", "--project", "beren-luthien-expanded", "--graph-stats", "--projects-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output

    context = json.loads((proj_dir / "context_stats.json").read_text(encoding="utf-8"))
    local_chars = context["local_story_neighborhood"]["character_priors"]
    assert "Beren" in local_chars
    assert "Luthien" in local_chars
    assert "Thingol" in local_chars
    assert "Turin" not in local_chars
    assert "Nienor" not in local_chars


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


def test_shadow_graph_workflow_end_to_end(tmp_path, monkeypatch):
    from book_graph_analyzer import story_cli as story_module

    monkeypatch.setattr(story_module, "_new_story_shadow_graph", lambda story_id: _FakeStoryShadowGraph(story_id))
    monkeypatch.setattr(story_module, "_new_story_scene_generator", lambda shadow_graph: _EchoStorySceneGenerator(shadow_graph))
    monkeypatch.setattr(story_module, "_new_story_generation_writer", lambda: _FakeStoryGenerationWriter())

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
    assert trace["sections"][0]["generated_scene_id"] == "beren-luthien-expanded-ch01-sc01"
    assert trace["sections"][0]["model_used"] == "fake-scene-generator"

    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == "chapter-audit-v1"
    assert audit["status"] in {"pass", "warn", "fail"}


def test_story_draft_enforces_required_terms(tmp_path, monkeypatch):
    from book_graph_analyzer import story_cli as story_module

    monkeypatch.setattr(story_module, "_new_story_shadow_graph", lambda story_id: _FakeStoryShadowGraph(story_id))
    monkeypatch.setattr(story_module, "_new_story_scene_generator", lambda shadow_graph: _EchoStorySceneGenerator(shadow_graph))
    monkeypatch.setattr(story_module, "_new_story_generation_writer", lambda: _FakeStoryGenerationWriter())

    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps({"events": [], "relations": []}), encoding="utf-8")

    proj_dir = tmp_path / "required-terms-proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Required Terms Project",
                "slug": "required-terms-proj",
                "genre": "fantasy",
                "premise": "A constrained grounded draft.",
                "target_chapters": 1,
                "scenes_per_chapter": 1,
                "event_files": [str(events_path)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": ["Thingol", "Tol-in-Gaurhoth confrontation"],
                "forbidden_terms": [],
                "enforcement": {"required_terms": True, "max_retries": 2},
                "style": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps(
            {
                "trajectory": [
                    {
                        "scene_id": "ch01-sc01",
                        "shadow_event_id": "shadow-event-1",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps(
            {
                "nodes": [
                        {
                            "id": "shadow-event-1",
                            "characters": ["Beren", "Luthien"],
                            "motifs": ["song"],
                            "action": "journey",
                            "source_canon_node_ids": ["canon-event-1"],
                        },
                        {"id": "shadow-ch01-sc01"},
                        {"id": "canon-event-1", "type": "CanonEvidence"},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    draft = runner.invoke(
        main,
        ["story", "draft", "--project", "required-terms-proj", "--chapter", "1", "--grounded", "--projects-dir", str(tmp_path)],
    )
    assert draft.exit_code == 0, draft.output

    chapter_text = (proj_dir / "chapter_01.md").read_text(encoding="utf-8")
    assert "Thingol" in chapter_text
    assert "Tol-in-Gaurhoth confrontation" in chapter_text

    audit = runner.invoke(
        main,
        ["story", "audit", "--project", "required-terms-proj", "--chapter", "1", "--projects-dir", str(tmp_path), "--enforce-required-terms"],
    )
    assert audit.exit_code == 0, audit.output
    audit_report = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit_report["status"] == "pass"
    assert audit_report["constraints"]["required_missing"] == []


def test_story_draft_fails_when_required_terms_never_appear(tmp_path, monkeypatch):
    from book_graph_analyzer import story_cli as story_module

    proj_dir = tmp_path / "required-terms-fail"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "x", "slug": "required-terms-fail", "target_chapters": 1, "scenes_per_chapter": 1}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": ["Melian"], "forbidden_terms": [], "enforcement": {"required_terms": True, "max_retries": 2}}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(json.dumps({"nodes": [{"id": "shadow-event-1"}]}, indent=2), encoding="utf-8")

    monkeypatch.setattr(story_module, "_new_story_shadow_graph", lambda story_id: _FakeStoryShadowGraph(story_id))
    monkeypatch.setattr(story_module, "_new_story_scene_generator", lambda shadow_graph: _NoAnchorStorySceneGenerator(shadow_graph))
    monkeypatch.setattr(story_module, "_new_story_generation_writer", lambda: _FakeStoryGenerationWriter())

    runner = CliRunner()
    draft = runner.invoke(
        main,
        ["story", "draft", "--project", "required-terms-fail", "--chapter", "1", "--grounded", "--projects-dir", str(tmp_path)],
    )
    assert draft.exit_code != 0
    assert "failed required-term enforcement" in draft.output
    assert "Missing required terms" in draft.output


def test_shadow_sampler_determinism_and_selector(tmp_path):
    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "B&L", "slug": "beren-luthien-expanded", "target_chapters": 2, "scenes_per_chapter": 2}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "chapters": [
                    {"chapter_number": 1, "scenes": [{"scene_id": "ch01-sc01", "goal": "setup", "summary": "setup"}, {"scene_id": "ch01-sc02", "goal": "vow", "summary": "vow"}]},
                    {"chapter_number": 2, "scenes": [{"scene_id": "ch02-sc01", "goal": "journey", "summary": "journey"}, {"scene_id": "ch02-sc02", "goal": "trial", "summary": "trial"}]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "context_stats.json").write_text(
        json.dumps(
            {
                "event_transition_probabilities": {
                    "unknown": {"journey": 0.7, "reveal": 0.3},
                    "journey": {"journey": 0.2, "conflict": 0.8},
                    "conflict": {"reveal": 0.6, "journey": 0.4},
                    "reveal": {"journey": 1.0},
                },
                "character_participation_priors": {"Beren": 0.5, "Luthien": 0.4, "Thingol": 0.1},
                "motif_reference_density_priors": {"oath": 0.5, "song": 0.3, "fate": 0.2},
                "register_style_budgets": {"target_words_per_scene": 3},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": ["oath"], "forbidden_terms": ["spaceship"]}, indent=2),
        encoding="utf-8",
    )

    runner = CliRunner()
    sample_cmd = [
        "story",
        "sample-shadow",
        "--project",
        "beren-luthien-expanded",
        "--n",
        "12",
        "--method",
        "anneal",
        "--seed",
        "42",
        "--steps",
        "20",
        "--projects-dir",
        str(tmp_path),
    ]
    r1 = runner.invoke(main, sample_cmd)
    assert r1.exit_code == 0, r1.output
    first = (proj_dir / "shadow_samples.jsonl").read_text(encoding="utf-8")
    r2 = runner.invoke(main, sample_cmd)
    assert r2.exit_code == 0, r2.output
    second = (proj_dir / "shadow_samples.jsonl").read_text(encoding="utf-8")
    assert first == second
    first_row = json.loads(first.splitlines()[0])
    assert first_row["prior_sources"]["characters"] == "project_canon_global_fallback"
    assert first_row["prior_sources"]["local_story_neighborhood_available"] is False

    score = runner.invoke(main, ["story", "score-shadow", "--project", "beren-luthien-expanded", "--pareto", "--projects-dir", str(tmp_path)])
    assert score.exit_code == 0, score.output
    score_payload = json.loads((proj_dir / "shadow_scores.json").read_text(encoding="utf-8"))
    assert score_payload["scores"]
    one = score_payload["scores"][0]
    comps = one["components"]
    for key in (
        "canon_consistency_penalty",
        "canon_consistency",
        "transition_likelihood",
        "arc_coherence",
        "style_register",
        "novelty_diversity",
    ):
        assert key in comps
        assert 0.0 <= float(comps[key]) <= 1.0
    assert (proj_dir / "shadow_pareto_front.json").exists()

    sel = runner.invoke(main, ["story", "select-shadow", "--project", "beren-luthien-expanded", "--top", "5", "--projects-dir", str(tmp_path)])
    assert sel.exit_code == 0, sel.output
    selected = json.loads((proj_dir / "shadow_selected.json").read_text(encoding="utf-8"))
    assert len(selected["selected"]) == 5
    # stable ordering rule
    pairs = [(float(r["weighted_score"]), str(r["candidate_id"])) for r in selected["selected"]]
    assert pairs == sorted(pairs, key=lambda x: (-x[0], x[1]))


def test_shadow_sampler_prefers_local_story_time_filtered_priors(tmp_path):
    proj_dir = tmp_path / "hunt-for-gollum"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Hunt for Gollum",
                "slug": "hunt-for-gollum",
                "timeline": {"story_era": "Third Age", "story_year": 3017},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "chapters": [
                    {
                        "chapter_number": 1,
                        "scenes": [
                            {
                                "scene_id": "ch01-sc01",
                                "goal": "Follow the trail",
                                "summary": "Aragorn searches the wild.",
                            }
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "context_stats.json").write_text(
        json.dumps(
            {
                "timeline": {"story_era": "Third Age", "story_year": 3017},
                "event_transition_probabilities": {"unknown": {"feast": 1.0}},
                "character_participation_priors": {"Fingon": 0.9, "Aragorn": 0.1},
                "motif_reference_density_priors": {"doom": 1.0},
                "entity_temporal_presence": {
                    "Aragorn": {
                        "eras": ["Third Age"],
                        "years_by_era": {
                            "Third Age": {"year_start": 2950, "year_end": 3019}
                        },
                    },
                    "Bilbo": {
                        "eras": ["Third Age"],
                        "years_by_era": {
                            "Third Age": {"year_start": 2941, "year_end": 3001}
                        },
                    },
                    "Fingon": {"eras": ["First Age"], "years_by_era": {}},
                },
                "local_story_neighborhood": {
                    "character_priors": {"Aragorn": 0.7, "Bilbo": 0.3},
                    "action_priors": {"track": 1.0},
                    "motif_priors": {"patience": 1.0},
                },
                "register_style_budgets": {"target_words_per_scene": 4},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": [], "forbidden_terms": []}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "story",
            "sample-shadow",
            "--project",
            "hunt-for-gollum",
            "--n",
            "1",
            "--seed",
            "7",
            "--steps",
            "1",
            "--projects-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    row = json.loads(
        (proj_dir / "shadow_samples.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["prior_sources"]["characters"] == "local_story_neighborhood"
    assert row["prior_sources"]["motifs"] == "local_story_neighborhood"
    assert row["prior_sources"]["actions"] == "local_story_neighborhood_blended"
    assert row["prior_sources"]["timeline_filter"]["excluded_characters"]["Bilbo"] == "past_only"
    assert row["state"][0]["characters"] == ["Aragorn"]
    assert row["state"][0]["motifs"] == ["patience"]
    assert row["state"][0]["action"] == "track"


def test_stable_seed_uses_canonical_json_materialization():
    from book_graph_analyzer.story_cli import _canonical_json, _stable_seed

    plan_a = {"b": 1, "a": 2}
    plan_b = {"a": 2, "b": 1}
    constraints_a = {"required_elements": ["oath"], "forbidden_terms": []}
    constraints_b = {"forbidden_terms": [], "required_elements": ["oath"]}

    seed_a = _stable_seed("proj", _canonical_json(plan_a), _canonical_json(constraints_a))
    seed_b = _stable_seed("proj", _canonical_json(plan_b), _canonical_json(constraints_b))

    assert seed_a == seed_b


def test_story_solve_fails_hard_when_required_elements_missing(tmp_path):
    proj_dir = tmp_path / "solve-hard-gate"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "x", "slug": "solve-hard-gate", "target_chapters": 1, "scenes_per_chapter": 1}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": ["Melian"], "forbidden_terms": []}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "scene_id": "ch01-sc01",
                        "shadow_event": {"id": "e1", "action": "journey", "description": "Beren travels.", "characters": ["Beren"]},
                        "plausibility_score": 0.9,
                        "transition_probability": 0.9,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out = runner.invoke(main, ["story", "solve", "--project", "solve-hard-gate", "--projects-dir", str(tmp_path)])
    assert out.exit_code != 0
    assert "failed hard required-element gating" in out.output


def test_selector_early_stops_on_goal_completion(tmp_path):
    proj_dir = tmp_path / "solve-early-stop"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(json.dumps({"name": "x", "slug": "solve-early-stop"}, indent=2), encoding="utf-8")
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "selection": {"goal_completion_threshold": 1.0, "min_beats_per_scene": 2, "anti_padding_penalty": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "scene_id": "ch01-sc01",
                        "scene_goal_progress": 0.5,
                        "shadow_event": {"id": "e1", "action": "setup", "description": "Setup scene.", "characters": ["A"]},
                        "plausibility_score": 0.95,
                        "transition_probability": 0.95,
                    },
                    {
                        "candidate_id": "c2",
                        "scene_id": "ch01-sc02",
                        "scene_goal_progress": 1.0,
                        "shadow_event": {"id": "e2", "action": "pivot", "description": "Goal is achieved.", "characters": ["A"]},
                        "plausibility_score": 0.95,
                        "transition_probability": 0.95,
                    },
                    {
                        "candidate_id": "c3",
                        "scene_id": "ch01-sc03",
                        "scene_goal_progress": 1.0,
                        "shadow_event": {"id": "e3", "action": "linger", "description": "Padding after completion.", "characters": ["A"]},
                        "plausibility_score": 0.95,
                        "transition_probability": 0.95,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out = runner.invoke(main, ["story", "solve", "--project", "solve-early-stop", "--projects-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    solved = json.loads((proj_dir / "shadow_solution.json").read_text(encoding="utf-8"))
    assert solved["k_max"] == 3
    assert [row["candidate_id"] for row in solved["trajectory"]] == ["c1", "c2"]


def test_selector_rejects_padded_nonprogress_beats(tmp_path):
    proj_dir = tmp_path / "solve-nonprogress-padding"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(json.dumps({"name": "x", "slug": "solve-nonprogress-padding"}, indent=2), encoding="utf-8")
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "selection": {
                    "goal_completion_threshold": 1.0,
                    "min_beats_per_scene": 2,
                    "anti_padding_penalty": 1.2,
                    "unresolved_thread_penalty": 0.7,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "scene_id": "ch01-sc01",
                        "scene_goal_progress": 0.6,
                        "shadow_event": {"id": "e1", "action": "setup", "description": "Setup scene.", "characters": ["A"]},
                        "plausibility_score": 0.99,
                        "transition_probability": 0.99,
                    },
                    {
                        "candidate_id": "c2",
                        "scene_id": "ch01-sc02",
                        "scene_goal_progress": 1.0,
                        "shadow_event": {"id": "e2", "action": "resolve", "description": "Resolution.", "characters": ["A"]},
                        "plausibility_score": 0.99,
                        "transition_probability": 0.99,
                    },
                    {
                        "candidate_id": "c3_pad",
                        "scene_id": "ch01-sc03",
                        "scene_goal_progress": 1.0,
                        "unresolved_causal_threads": 2,
                        "shadow_event": {"id": "e3", "action": "linger", "description": "Extra non-progress beat.", "characters": ["A"]},
                        "plausibility_score": 0.99,
                        "transition_probability": 0.99,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out = runner.invoke(main, ["story", "solve", "--project", "solve-nonprogress-padding", "--projects-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    solved = json.loads((proj_dir / "shadow_solution.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in solved["trajectory"]] == ["c1", "c2"]


def test_causal_precondition_fail_with_state_mismatch(tmp_path):
    proj_dir = tmp_path / "solve-causal-state"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(json.dumps({"name": "x", "slug": "solve-causal-state"}, indent=2), encoding="utf-8")
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": [], "forbidden_terms": []}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "scene_id": "ch01-sc01",
                        "shadow_event": {"id": "e1", "action": "approach", "description": "Approach the gate.", "characters": ["A"]},
                        "plausibility_score": 0.95,
                        "transition_probability": 0.95,
                    },
                    {
                        "candidate_id": "c2_invalid",
                        "scene_id": "ch01-sc02",
                        "preconditions": ["gate_open"],
                        "shadow_event": {"id": "e2", "action": "enter", "description": "Enter through the gate.", "characters": ["A"]},
                        "plausibility_score": 0.99,
                        "transition_probability": 0.99,
                    },
                    {
                        "candidate_id": "c2_valid",
                        "scene_id": "ch01-sc02",
                        "shadow_event": {"id": "e3", "action": "wait", "description": "Wait outside the closed gate.", "characters": ["A"]},
                        "plausibility_score": 0.7,
                        "transition_probability": 0.7,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out = runner.invoke(main, ["story", "solve", "--project", "solve-causal-state", "--projects-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    solved = json.loads((proj_dir / "shadow_solution.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in solved["trajectory"]] == ["c1", "c2_valid"]


def test_causal_precondition_tristate_unknown_strict_reject_by_default(tmp_path):
    proj_dir = tmp_path / "solve-causal-state-unknown-reject"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(json.dumps({"name": "x", "slug": "solve-causal-state-unknown-reject"}, indent=2), encoding="utf-8")
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": [], "forbidden_terms": []}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "scene_id": "ch01-sc01",
                        "shadow_event": {"id": "e1", "action": "approach", "description": "Approach the gate.", "characters": ["A"]},
                        "plausibility_score": 0.95,
                        "transition_probability": 0.95,
                    },
                    {
                        "candidate_id": "c2_unknown_rejected",
                        "scene_id": "ch01-sc02",
                        "preconditions": {"gate_open": True},
                        "shadow_event": {"id": "e2", "action": "enter", "description": "Enter through the gate.", "characters": ["A"]},
                        "plausibility_score": 0.99,
                        "transition_probability": 0.99,
                    },
                    {
                        "candidate_id": "c2_valid",
                        "scene_id": "ch01-sc02",
                        "shadow_event": {"id": "e3", "action": "wait", "description": "Wait outside the gate.", "characters": ["A"]},
                        "plausibility_score": 0.7,
                        "transition_probability": 0.7,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out = runner.invoke(main, ["story", "solve", "--project", "solve-causal-state-unknown-reject", "--projects-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    solved = json.loads((proj_dir / "shadow_solution.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in solved["trajectory"]] == ["c1", "c2_valid"]


def test_causal_precondition_unknown_policy_soft_penalty_allows_candidate(tmp_path):
    proj_dir = tmp_path / "solve-causal-state-unknown-soft"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(json.dumps({"name": "x", "slug": "solve-causal-state-unknown-soft"}, indent=2), encoding="utf-8")
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "enforcement": {
                    "precondition_unknown_policy": "penalize",
                    "precondition_unknown_penalty": 0.1,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "scene_id": "ch01-sc01",
                        "shadow_event": {"id": "e1", "action": "approach", "description": "Approach the gate.", "characters": ["A"]},
                        "plausibility_score": 0.95,
                        "transition_probability": 0.95,
                    },
                    {
                        "candidate_id": "c2_unknown_soft",
                        "scene_id": "ch01-sc02",
                        "preconditions": {"gate_open": "true"},
                        "shadow_event": {"id": "e2", "action": "enter", "description": "Enter through the gate.", "characters": ["A"]},
                        "plausibility_score": 0.99,
                        "transition_probability": 0.99,
                    },
                    {
                        "candidate_id": "c2_valid_low",
                        "scene_id": "ch01-sc02",
                        "shadow_event": {"id": "e3", "action": "wait", "description": "Wait outside the gate.", "characters": ["A"]},
                        "plausibility_score": 0.6,
                        "transition_probability": 0.6,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out = runner.invoke(main, ["story", "solve", "--project", "solve-causal-state-unknown-soft", "--projects-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    solved = json.loads((proj_dir / "shadow_solution.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in solved["trajectory"]] == ["c1", "c2_unknown_soft"]


def test_causal_precondition_false_satisfies_from_explicit_effect_state(tmp_path):
    proj_dir = tmp_path / "solve-causal-state-false"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(json.dumps({"name": "x", "slug": "solve-causal-state-false"}, indent=2), encoding="utf-8")
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": [], "forbidden_terms": []}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "c1_set_false",
                        "scene_id": "ch01-sc01",
                        "effects": {"alarm_on": False},
                        "shadow_event": {"id": "e1", "action": "silence", "description": "Silence the alarm.", "characters": ["A"]},
                        "plausibility_score": 0.95,
                        "transition_probability": 0.95,
                    },
                    {
                        "candidate_id": "c2_requires_false",
                        "scene_id": "ch01-sc02",
                        "preconditions": {"alarm_on": False},
                        "shadow_event": {"id": "e2", "action": "sneak", "description": "Sneak past the gate.", "characters": ["A"]},
                        "plausibility_score": 0.99,
                        "transition_probability": 0.99,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out = runner.invoke(main, ["story", "solve", "--project", "solve-causal-state-false", "--projects-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    solved = json.loads((proj_dir / "shadow_solution.json").read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in solved["trajectory"]] == ["c1_set_false", "c2_requires_false"]


def test_story_grow_shadow_applies_project_priors_and_audit_reports_domain_metrics(tmp_path, monkeypatch):
    from book_graph_analyzer import story_cli as story_module

    monkeypatch.setattr(story_module, "_new_story_shadow_graph", lambda story_id: _FakeStoryShadowGraph(story_id))
    monkeypatch.setattr(story_module, "_new_story_scene_generator", lambda shadow_graph: _EchoStorySceneGenerator(shadow_graph))
    monkeypatch.setattr(story_module, "_new_story_generation_writer", lambda: _FakeStoryGenerationWriter())

    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps({"events": [], "relations": []}), encoding="utf-8")

    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "target_chapters": 1,
                "scenes_per_chapter": 1,
                "event_files": [str(events_path)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": ["Thingol"], "forbidden_terms": []}, indent=2), encoding="utf-8"
    )
    (proj_dir / "plan.json").write_text(
        json.dumps({"project_slug": "beren-luthien-expanded", "chapters": [{"chapter_number": 1, "scenes": [{"scene_id": "ch01-sc01", "goal": "setup", "summary": "x"}]}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "context_stats.json").write_text(
        json.dumps(
            {
                "event_transition_probabilities": {"unknown": {"journey": 0.8, "unknown": 0.2}},
                "character_participation_priors": {"Frodo": 0.95, "Beren": 0.2, "Thingol": 0.2},
                "motif_reference_density_priors": {"oath": 0.7, "song": 0.6},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    assert runner.invoke(main, ["story", "grow-shadow", "--project", "beren-luthien-expanded", "--auto", "--projects-dir", str(tmp_path)]).exit_code == 0
    assert runner.invoke(main, ["story", "solve", "--project", "beren-luthien-expanded", "--projects-dir", str(tmp_path)]).exit_code == 0
    assert runner.invoke(main, ["story", "draft", "--project", "beren-luthien-expanded", "--chapter", "1", "--grounded", "--projects-dir", str(tmp_path)]).exit_code == 0
    assert runner.invoke(main, ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)]).exit_code == 0

    cand = json.loads((proj_dir / "shadow_candidates.json").read_text(encoding="utf-8"))["candidates"][0]
    assert cand["project_prior"]["canon_hits"] >= 1

    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert "quality_proxies" in audit
    assert "domain_alignment" in audit


def test_story_template_renderer_full_chapter_flow_without_external_services(tmp_path):
    events_path = tmp_path / "events.json"
    events_path.write_text(
        json.dumps(
            {
                "events": {
                    "1": {
                        "id": "1",
                        "description": "Beren reaches Doriath under the shadow of his oath.",
                        "agent": "Beren",
                        "action": "journey",
                        "patient": "Doriath",
                        "era": "First Age",
                        "year": 466,
                    },
                    "2": {
                        "id": "2",
                        "description": "Luthien speaks before Thingol and Melian.",
                        "agent": "Luthien",
                        "action": "counsel",
                        "patient": "Thingol",
                        "era": "First Age",
                        "year": 466,
                    },
                    "3": {
                        "id": "3",
                        "description": "Thingol names the Silmaril as the bride-price.",
                        "agent": "Thingol",
                        "action": "decree",
                        "patient": "Silmaril",
                        "era": "First Age",
                        "year": 466,
                    },
                },
                "relations": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "premise": "A First Age retelling focused on oath, love, and the Doom of Beleriand.",
                "target_chapters": 1,
                "scenes_per_chapter": 2,
                "event_files": [str(events_path)],
                "timeline": {"story_era": "First Age", "story_year": 466},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": ["Thingol", "Silmaril"],
                "forbidden_terms": ["spaceship"],
                "enforcement": {"required_terms": True, "max_retries": 2},
                "quality": {
                    "min_scene_words": 80,
                    "target_scene_words": 130,
                    "min_dialogue_ratio": 0.08,
                    "target_dialogue_ratio": 0.10,
                    "forbid_placeholder_terms": ["Unknown", "Someone", "placeholder"],
                    "forbid_out_of_domain_entities": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "beren-luthien-expanded",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "title": "The Price Named",
                        "scenes": [
                            {"scene_id": "ch01-sc01", "goal": "Beren reaches Doriath.", "summary": "Beren enters Thingol's guarded realm."},
                            {"scene_id": "ch01-sc02", "goal": "Thingol names the Silmaril.", "summary": "The oath is given shape before Melian."},
                        ],
                    }
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
        ["story", "beats", "expand", "--project", "beren-luthien-expanded", "--projects-dir", str(tmp_path)],
        [
            "story",
            "draft",
            "--project",
            "beren-luthien-expanded",
            "--chapter",
            "1",
            "--grounded",
            "--renderer",
            "template",
            "--projects-dir",
            str(tmp_path),
        ],
        ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)],
    ]
    for cmd in commands:
        result = runner.invoke(main, cmd)
        assert result.exit_code == 0, result.output

    chapter_text = (proj_dir / "chapter_01.md").read_text(encoding="utf-8")
    assert "Unknown" not in chapter_text
    for wrong_era_name in ["Bilbo", "Frodo", "Gandalf"]:
        assert wrong_era_name not in chapter_text

    trace = json.loads((proj_dir / "chapter_01_trace.json").read_text(encoding="utf-8"))
    assert trace["sections"][0]["model_used"] == "template-renderer"
    assert trace["sections"][0]["word_count"] >= 80

    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["quality_proxies"]["placeholder_hits"] == []
    assert audit["quality_proxies"]["min_scene_word_violations"] == []
    assert audit["quality_proxies"]["dialogue_ratio"] >= 0.08
    assert audit["domain_alignment"]["out_of_domain_text_hits"] == []


def test_story_template_renderer_shire_gap_flow_without_external_services(tmp_path):
    events_path = tmp_path / "shire_gap_events.json"
    events_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "id": "shire-gap-001",
                        "era": "Third Age",
                        "year": 3001,
                        "agent": "Frodo",
                        "patient": "Bag End",
                        "action": "inherit",
                        "description": "Frodo keeps Bag End after Bilbo leaves the Shire.",
                    },
                    {
                        "id": "shire-gap-002",
                        "era": "Third Age",
                        "year": 3008,
                        "agent": "Gandalf",
                        "patient": "Frodo",
                        "action": "question",
                        "description": "Gandalf visits Frodo in autumn and asks careful questions about Bilbo's ring.",
                    },
                    {
                        "id": "shire-gap-003",
                        "era": "Third Age",
                        "year": 3008,
                        "agent": "Sam",
                        "patient": "Frodo",
                        "action": "witness",
                        "description": "Sam tends the garden while Gandalf leaves Frodo with a private warning.",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    proj_dir = tmp_path / "shire-last-autumn-visit"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "The Last Autumn Visit",
                "slug": "shire-last-autumn-visit",
                "premise": "A Third Age Shire chapter set in autumn 3008, before Frodo learns what Bilbo's ring truly is.",
                "target_chapters": 1,
                "scenes_per_chapter": 3,
                "event_files": [str(events_path)],
                "timeline": {
                    "story_era": "Third Age",
                    "story_year": 3008,
                    "allow_past_references": True,
                    "forbid_future_entities": True,
                    "forbidden_entities": ["Boromir", "Legolas", "Gimli", "Aragorn", "Ringwraith", "Nazgul", "Sauron", "Saruman", "Gollum"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": ["Frodo", "Gandalf", "Sam", "Bag End", "Ring"],
                "forbidden_terms": ["One Ring", "Sauron", "Saruman", "Gollum", "Ringwraith", "Nazgul", "Aragorn"],
                "enforcement": {"required_terms": True, "max_retries": 2},
                "quality": {
                    "min_scene_words": 120,
                    "target_scene_words": 180,
                    "min_dialogue_ratio": 0.05,
                    "target_dialogue_ratio": 0.08,
                    "forbid_placeholder_terms": ["Unknown", "Someone", "placeholder"],
                    "forbid_out_of_domain_entities": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "shire-last-autumn-visit",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "title": "The Last Autumn Visit",
                        "intent": "Show Frodo's quiet inheritance and Gandalf's unresolved fear without revealing the ring.",
                        "scenes": [
                            {
                                "scene_id": "ch01-sc01",
                                "goal": "Frodo keeps Bag End in the years after Bilbo's departure.",
                                "summary": "Bag End carries Bilbo's absence while Sam tends the autumn garden.",
                                "characters": ["Frodo", "Sam"],
                                "setting": "Bag End",
                                "objects": ["Ring", "Bilbo's maps"],
                            },
                            {
                                "scene_id": "ch01-sc02",
                                "goal": "Gandalf visits Frodo in autumn 3008 and asks careful questions.",
                                "summary": "Gandalf returns without fireworks and asks about Bilbo's ring.",
                                "characters": ["Frodo", "Gandalf", "Sam"],
                                "setting": "Bag End",
                                "objects": ["Ring", "pipe ash", "garden shears"],
                            },
                            {
                                "scene_id": "ch01-sc03",
                                "goal": "Gandalf leaves before dawn, asking Frodo for caution.",
                                "summary": "Frodo is left with warnings he cannot interpret while Sam watches the Road.",
                                "characters": ["Frodo", "Gandalf", "Sam"],
                                "setting": "Hobbiton",
                                "objects": ["Ring", "walking staff", "Road"],
                            },
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    commands = [
        ["story", "context", "--project", "shire-last-autumn-visit", "--graph-stats", "--projects-dir", str(tmp_path)],
        ["story", "grow-shadow", "--project", "shire-last-autumn-visit", "--auto", "--projects-dir", str(tmp_path)],
        ["story", "solve", "--project", "shire-last-autumn-visit", "--projects-dir", str(tmp_path)],
        ["story", "beats", "expand", "--project", "shire-last-autumn-visit", "--projects-dir", str(tmp_path)],
        ["story", "draft", "--project", "shire-last-autumn-visit", "--chapter", "1", "--grounded", "--renderer", "template", "--projects-dir", str(tmp_path)],
        ["story", "audit", "--project", "shire-last-autumn-visit", "--chapter", "1", "--projects-dir", str(tmp_path)],
    ]
    for cmd in commands:
        result = runner.invoke(main, cmd)
        assert result.exit_code == 0, result.output

    chapter_text = (proj_dir / "chapter_01.md").read_text(encoding="utf-8")
    assert "Unknown" not in chapter_text
    assert "One Ring" not in chapter_text
    for wrong_era_name in ["Boromir", "Legolas", "Gimli", "Aragorn", "Ringwraith", "Nazgul", "Sauron", "Saruman", "Gollum"]:
        assert wrong_era_name not in chapter_text

    trace = json.loads((proj_dir / "chapter_01_trace.json").read_text(encoding="utf-8"))
    assert len(trace["sections"]) == 3
    assert {sec["model_used"] for sec in trace["sections"]} == {"template-renderer"}

    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["constraints"]["required_missing"] == []
    assert audit["quality_proxies"]["placeholder_hits"] == []
    assert audit["domain_alignment"]["out_of_domain_text_hits"] == []


def test_story_template_renderer_hunt_for_gollum_flow_without_external_services(tmp_path):
    events_path = tmp_path / "hunt_events.json"
    events_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "id": "hunt-001",
                        "era": "Third Age",
                        "year": 3009,
                        "agent": "Gandalf",
                        "patient": "Aragorn",
                        "action": "charge",
                        "description": "Gandalf asks Aragorn to hunt Gollum before Baggins and the Shire are exposed.",
                        "motifs": ["hunt", "Baggins", "Shire"],
                    },
                    {
                        "id": "hunt-002",
                        "era": "Third Age",
                        "year": 3009,
                        "agent": "Aragorn",
                        "patient": "Gollum",
                        "action": "track",
                        "description": "Aragorn follows Gollum along the Anduin and into Wilderland.",
                        "motifs": ["trail", "wilderness", "Gollum"],
                    },
                    {
                        "id": "hunt-003",
                        "era": "Third Age",
                        "year": 3017,
                        "agent": "Gollum",
                        "patient": "Baggins",
                        "action": "mutter",
                        "description": "Gollum mutters about Baggins, the Shire, and the lost ring.",
                        "motifs": ["fear", "Baggins", "Shire"],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    proj_dir = tmp_path / "hunt-for-gollum"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "The Hunt for Gollum",
                "slug": "hunt-for-gollum",
                "premise": "Gandalf sends Aragorn to hunt Gollum before the names Shire and Baggins become dangerous.",
                "target_chapters": 1,
                "scenes_per_chapter": 3,
                "variable_scenes_per_chapter": True,
                "event_files": [str(events_path)],
                "default_setting": "Wilderland",
                "timeline": {
                    "story_era": "Third Age",
                    "story_year": 3017,
                    "forbid_future_entities": True,
                    "forbidden_entities": ["Frodo", "Sam", "Merry", "Pippin", "Fellowship", "Council of Elrond"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": ["Gandalf", "Aragorn", "Gollum", "Shire", "Baggins"],
                "forbidden_terms": ["Frodo", "Merry", "Pippin", "Fellowship", "Council of Elrond"],
                "enforcement": {"required_terms": True, "max_retries": 2},
                "quality": {
                    "min_scene_words": 120,
                    "min_chapter_words": 380,
                    "target_scene_words": 180,
                    "min_dialogue_ratio": 0.05,
                    "target_dialogue_ratio": 0.08,
                    "forbid_placeholder_terms": ["Unknown", "Someone", "placeholder"],
                    "forbid_out_of_domain_entities": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "hunt-for-gollum",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "title": "A Name in the Dark",
                        "intent": "Turn suspicion into a hidden charge, then prove the quarry is carrying dangerous names.",
                        "structure_role": "inciting charge; three movements because suspicion, pursuit, and quarry response must each turn the chapter",
                        "movement_count": 3,
                        "movement_count_basis": "inciting charge; three movements because suspicion, pursuit, and quarry response must each turn the chapter",
                        "scenes": [
                            {
                                "scene_id": "ch01-sc01",
                                "goal": "Gandalf asks Aragorn near Bree to hunt Gollum before the names Shire and Baggins travel farther east.",
                                "summary": "Aragorn accepts the hidden labour of the hunt.",
                                "characters": ["Gandalf", "Aragorn"],
                                "setting": "Bree",
                                "objects": ["maps", "walking staff"],
                            },
                            {
                                "scene_id": "ch01-sc02",
                                "goal": "Aragorn follows the first thin trail along the Anduin.",
                                "summary": "The hunt becomes practical tracking work.",
                                "characters": ["Aragorn"],
                                "setting": "Anduin",
                                "objects": ["tracks", "maps"],
                            },
                            {
                                "scene_id": "ch01-sc03",
                                "goal": "Gollum slips through reeds and stone, muttering about Baggins and the Shire.",
                                "summary": "The quarry carries the dangerous words.",
                                "characters": ["Gollum"],
                                "setting": "Wilderland",
                                "objects": ["fish bones", "ring"],
                            },
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    commands = [
        ["story", "context", "--project", "hunt-for-gollum", "--graph-stats", "--projects-dir", str(tmp_path)],
        ["story", "grow-shadow", "--project", "hunt-for-gollum", "--auto", "--projects-dir", str(tmp_path)],
        ["story", "solve", "--project", "hunt-for-gollum", "--projects-dir", str(tmp_path)],
        ["story", "beats", "expand", "--project", "hunt-for-gollum", "--projects-dir", str(tmp_path)],
        ["story", "draft", "--project", "hunt-for-gollum", "--chapter", "1", "--grounded", "--renderer", "template", "--projects-dir", str(tmp_path)],
        ["story", "audit", "--project", "hunt-for-gollum", "--chapter", "1", "--projects-dir", str(tmp_path)],
    ]
    for cmd in commands:
        result = runner.invoke(main, cmd)
        assert result.exit_code == 0, result.output

    chapter_text = (proj_dir / "chapter_01.md").read_text(encoding="utf-8")
    assert "Gollum" in chapter_text
    assert "Aragorn" in chapter_text
    assert "Baggins" in chapter_text
    assert "That purpose governed" not in chapter_text
    assert "The labour was not to multiply happenings" not in chapter_text
    for out_of_scope in ["Frodo", "Sam", "Fellowship", "Council of Elrond"]:
        assert out_of_scope not in chapter_text

    draft = json.loads((proj_dir / "chapter_01_draft.json").read_text(encoding="utf-8"))
    assert draft["chapter_structure"]["purpose_driven"] is True
    assert draft["chapter_structure"]["movement_count_matches_plan"] is True

    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["chapter_structure"]["purpose_driven"] is True
    assert audit["chapter_structure"]["actual_movement_count"] == 3
    assert audit["constraints"]["required_missing"] == []
    assert audit["domain_alignment"]["out_of_domain_text_hits"] == []


def test_story_audit_fails_hunt_for_gollum_out_of_scope_leakage(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "hunt-for-gollum"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "The Hunt for Gollum",
                "slug": "hunt-for-gollum",
                "timeline": {
                    "story_era": "Third Age",
                    "story_year": 3017,
                    "forbidden_entities": ["Frodo", "Sam", "Fellowship"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": ["Gandalf", "Aragorn", "Gollum", "Shire", "Baggins"],
                "forbidden_terms": [],
                "quality": {
                    "min_scene_words": 1,
                    "forbid_placeholder_terms": ["Unknown"],
                    "forbid_out_of_domain_entities": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "chapter_01.md").write_text(
        "Gandalf and Aragorn hunted Gollum for the sake of Baggins and the Shire, but Frodo and Sam appeared in the Fellowship.\n",
        encoding="utf-8",
    )
    (proj_dir / "chapter_01_trace.json").write_text(
        json.dumps(
            {"sections": [{"section": 1, "scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "shadow_scene_id": "shadow-ch01-sc01", "word_count": 20}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "action": "hunt"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Gandalf", "Aragorn"]}, {"id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["story", "audit", "--project", "hunt-for-gollum", "--chapter", "1", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert "frodo" in audit["domain_alignment"]["out_of_domain_text_hits"]
    assert "sam" in audit["domain_alignment"]["out_of_domain_text_hits"]


def test_story_audit_fails_on_placeholder_prose(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "Beren and Luthien Expanded", "slug": "beren-luthien-expanded"}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "quality": {"forbid_placeholder_terms": ["Unknown"], "min_scene_words": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "chapter_01.md").write_text("Unknown stood in Doriath beside Beren.\n", encoding="utf-8")
    (proj_dir / "chapter_01_trace.json").write_text(
        json.dumps(
            {"sections": [{"section": 1, "scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "shadow_scene_id": "shadow-ch01-sc01", "word_count": 6}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "action": "counsel"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Beren"]}, {"id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert audit["quality_proxies"]["placeholder_hits"] == ["Unknown"]


def test_story_audit_fails_on_template_artifacts(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "hunt-for-gollum"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "The Hunt for Gollum", "slug": "hunt-for-gollum"}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "quality": {"min_scene_words": 1, "forbid_template_artifacts": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "chapter_01.md").write_text(
        "Gandalf and Aragorn watched Gollum. In Road, the road {ranger} chose held the old warning.\n",
        encoding="utf-8",
    )
    (proj_dir / "chapter_01_trace.json").write_text(
        json.dumps(
            {"sections": [{"section": 1, "scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "shadow_scene_id": "shadow-ch01-sc01", "word_count": 14}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "action": "counsel"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Gandalf", "Aragorn"]}, {"id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["story", "audit", "--project", "hunt-for-gollum", "--chapter", "1", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert "brace_placeholder" in audit["quality_proxies"]["template_artifact_hits"]
    assert "in_road" in audit["quality_proxies"]["template_artifact_hits"]


def test_story_audit_fails_when_dialogue_ratio_is_too_low(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "Beren and Luthien Expanded", "slug": "beren-luthien-expanded"}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "quality": {"min_dialogue_ratio": 0.10, "min_scene_words": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "chapter_01.md").write_text(
        "Beren stood in Doriath beside Luthien. The court waited in silence.\n",
        encoding="utf-8",
    )
    (proj_dir / "chapter_01_trace.json").write_text(
        json.dumps(
            {"sections": [{"section": 1, "scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "shadow_scene_id": "shadow-ch01-sc01", "word_count": 10}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "action": "counsel"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Beren"]}, {"id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert audit["quality_proxies"]["dialogue_ratio"] == 0.0
    assert audit["quality_proxies"]["min_dialogue_ratio_violation"] is True


def test_story_audit_fails_when_event_density_is_too_low(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "Beren and Luthien Expanded", "slug": "beren-luthien-expanded"}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "quality": {"min_event_sentence_ratio": 0.50, "min_scene_words": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "chapter_01.md").write_text(
        "The sorrow of the hour was deep. The matter had moral weight. Hope seemed distant. The silence was meaningful.\n",
        encoding="utf-8",
    )
    (proj_dir / "chapter_01_trace.json").write_text(
        json.dumps(
            {"sections": [{"section": 1, "scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "shadow_scene_id": "shadow-ch01-sc01", "word_count": 17}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "action": "counsel"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Beren"]}, {"id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert audit["quality_proxies"]["event_density"]["event_sentence_ratio"] < 0.50
    assert audit["quality_proxies"]["min_event_sentence_ratio_violation"] is True


def test_story_audit_fails_when_average_sentence_length_is_too_high(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps({"name": "Beren and Luthien Expanded", "slug": "beren-luthien-expanded"}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "quality": {"max_avg_sentence_words": 8.0, "min_scene_words": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "chapter_01.md").write_text(
        "Beren stood in Doriath beside Luthien while the court waited in deep silence beneath the hidden trees of the guarded realm.\n",
        encoding="utf-8",
    )
    (proj_dir / "chapter_01_trace.json").write_text(
        json.dumps(
            {"sections": [{"section": 1, "scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "shadow_scene_id": "shadow-ch01-sc01", "word_count": 20}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "action": "counsel"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Beren"]}, {"id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert audit["quality_proxies"]["max_avg_sentence_words_violation"] is True


def test_story_draft_fails_construction_quality_gate(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "premise": "A First Age retelling.",
                "target_chapters": 1,
                "scenes_per_chapter": 1,
                "timeline": {"story_era": "First Age", "story_year": 466},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps(
            {
                "required_elements": [],
                "forbidden_terms": [],
                "enforcement": {"max_retries": 0},
                "quality": {"target_scene_words": 90, "max_avg_sentence_words": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "beren-luthien-expanded",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "title": "Ch1",
                        "scenes": [
                            {
                                "scene_id": "ch01-sc01",
                                "goal": "Beren enters Doriath.",
                                "summary": "Beren enters the guarded realm.",
                                "characters": ["Beren", "Luthien"],
                                "setting": "Doriath",
                            }
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "shadow-event-1", "characters": ["Beren", "Luthien"], "motifs": ["oath"], "action": "counsel"},
                    {"id": "shadow-ch01-sc01"},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "story",
            "draft",
            "--project",
            "beren-luthien-expanded",
            "--chapter",
            "1",
            "--grounded",
            "--renderer",
            "template",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "Quality gate failures" in result.output
    assert "average sentence length" in result.output


def test_story_draft_fails_on_future_era_contamination(tmp_path, monkeypatch):
    from book_graph_analyzer import story_cli as story_module

    monkeypatch.setattr(story_module, "_new_story_shadow_graph", lambda story_id: _FakeStoryShadowGraph(story_id))
    monkeypatch.setattr(story_module, "_new_story_scene_generator", lambda shadow_graph: _FutureLeakStorySceneGenerator(shadow_graph))
    monkeypatch.setattr(story_module, "_new_story_generation_writer", lambda: _FakeStoryGenerationWriter())

    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "premise": "A First Age retelling.",
                "target_chapters": 1,
                "scenes_per_chapter": 1,
                "timeline": {"story_era": "First Age", "story_year": 466},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": [], "forbidden_terms": [], "enforcement": {"max_retries": 0}, "style": {}}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "beren-luthien-expanded",
                "chapters": [{"chapter_number": 1, "title": "Ch1", "scenes": [{"scene_id": "ch01-sc01", "goal": "setup", "summary": "Beren meets Thingol"}]}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Beren", "Thingol"], "motifs": ["oath"], "action": "counsel"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "context_stats.json").write_text(
        json.dumps(
            {
                "timeline": {"story_era": "First Age", "story_year": 466, "future_guardrail_entities": ["Bilbo"]},
                "entity_temporal_presence": {
                    "Bilbo": {"count": 10, "eras": ["Third Age"], "era_counts": {"Third Age": 10}, "year_start": 2941, "year_end": 3001, "years_by_era": {"Third Age": {"year_start": 2941, "year_end": 3001, "count": 10}}, "source_files": ["hobbit_events.json"]},
                    "Beren": {"count": 10, "eras": ["First Age"], "era_counts": {"First Age": 10}, "year_start": 465, "year_end": 466, "years_by_era": {"First Age": {"year_start": 465, "year_end": 466, "count": 10}}, "source_files": ["silmarillion_events.json"]},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["story", "draft", "--project", "beren-luthien-expanded", "--chapter", "1", "--grounded", "--projects-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "Future-era contamination" in result.output


def test_story_audit_allows_past_references_but_flags_future_mentions(tmp_path):
    runner = CliRunner()
    proj_dir = tmp_path / "beren-luthien-expanded"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Beren and Luthien Expanded",
                "slug": "beren-luthien-expanded",
                "timeline": {"story_era": "First Age", "story_year": 466},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": []}, indent=2), encoding="utf-8")
    (proj_dir / "context_stats.json").write_text(
        json.dumps(
            {
                "timeline": {"story_era": "First Age", "story_year": 466, "allow_past_references": True},
                "entity_temporal_presence": {
                    "Feanor": {"count": 4, "eras": ["First Age"], "era_counts": {"First Age": 4}, "year_start": 100, "year_end": 455, "years_by_era": {"First Age": {"year_start": 100, "year_end": 455, "count": 4}}, "source_files": ["silmarillion_events.json"]},
                    "Bilbo": {"count": 6, "eras": ["Third Age"], "era_counts": {"Third Age": 6}, "year_start": 2941, "year_end": 3001, "years_by_era": {"Third Age": {"year_start": 2941, "year_end": 3001, "count": 6}}, "source_files": ["hobbit_events.json"]},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (proj_dir / "chapter_01.md").write_text("Beren remembered Feanor, but Bilbo stood beside him in the hall.\n", encoding="utf-8")
    (proj_dir / "chapter_01_trace.json").write_text(
        json.dumps({"sections": [{"section": 1, "shadow_event_id": "shadow-event-1", "shadow_scene_id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_solution.json").write_text(
        json.dumps({"trajectory": [{"scene_id": "ch01-sc01", "shadow_event_id": "shadow-event-1", "action": "remember"}]}, indent=2),
        encoding="utf-8",
    )
    (proj_dir / "shadow_graph.json").write_text(
        json.dumps({"nodes": [{"id": "shadow-event-1", "characters": ["Beren"], "motifs": ["memory"]}, {"id": "shadow-ch01-sc01"}]}, indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        ["story", "audit", "--project", "beren-luthien-expanded", "--chapter", "1", "--projects-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    audit = json.loads((proj_dir / "chapter_01_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert audit["temporal_alignment"]["future_mention_count"] == 1
    assert audit["temporal_alignment"]["past_reference_count"] == 1
    assert audit["temporal_alignment"]["future_mentions"][0]["name"] == "Bilbo"
    assert audit["temporal_alignment"]["past_references"][0]["name"] == "Feanor"
    assert "bilbo" in audit["domain_alignment"]["out_of_domain_text_hits"]


def test_story_context_filters_future_events_and_emits_resolvable_evidence(tmp_path):
    events_path = tmp_path / "third_age_events.json"
    events_path.write_text(
        json.dumps(
            {
                "events": {
                    "past": {
                        "id": "past",
                        "description": "Gandalf asks Aragorn to seek Gollum.",
                        "agent": "Gandalf",
                        "patient": "Aragorn",
                        "action": "charge",
                        "era": "Third Age",
                        "year": 3017,
                        "source_book": "The Lord of the Rings",
                        "source_location": "Appendix B",
                    },
                    "future": {
                        "id": "future",
                        "description": "Aragorn is crowned after the War of the Ring.",
                        "agent": "Aragorn",
                        "action": "crowned",
                        "era": "Third Age",
                        "year": 3019,
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    proj_dir = tmp_path / "hunt-for-gollum"
    proj_dir.mkdir()
    (proj_dir / "project.json").write_text(
        json.dumps(
            {
                "name": "Hunt for Gollum",
                "slug": "hunt-for-gollum",
                "premise": "Gandalf sends Aragorn to seek Gollum.",
                "event_files": [str(events_path)],
                "timeline": {"story_era": "Third Age", "story_year": 3017},
            }
        ),
        encoding="utf-8",
    )
    (proj_dir / "constraints.json").write_text(
        json.dumps({"required_elements": [], "forbidden_terms": []}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["story", "context", "--project", "hunt-for-gollum", "--graph-stats", "--projects-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    context = json.loads((proj_dir / "context_stats.json").read_text(encoding="utf-8"))
    assert context["totals"]["events"] == 1
    evidence = context["canon_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["source_event_id"] == "past"
    assert evidence[0]["source_book"] == "The Lord of the Rings"
    assert evidence[0]["source_location"] == "Appendix B"
    assert evidence[0]["evidence_id"].startswith("canon-event-")
    assert "crowned" not in json.dumps(context).lower()


def test_arc_progression_rewards_development_not_action_stasis():
    from book_graph_analyzer.story_cli import _arc_progression_score

    static = _arc_progression_score(["travel", "travel", "travel", "travel"])
    developing = _arc_progression_score(["travel", "discover", "discover", "confront"])

    assert developing > static


def test_selected_shadow_samples_are_wired_into_solver_priors(tmp_path):
    from book_graph_analyzer.story_cli import _selected_sample_prior_score, _selected_sample_scene_priors

    (tmp_path / "shadow_selected.json").write_text(
        json.dumps({"selected": [{"candidate_id": "sample-1"}]}),
        encoding="utf-8",
    )
    (tmp_path / "shadow_samples.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "sample-1",
                "state": [
                    {
                        "scene_id": "ch01-sc01",
                        "action": "track",
                        "characters": ["Aragorn"],
                        "motifs": ["patience"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    priors = _selected_sample_scene_priors(tmp_path)
    aligned = {
        "scene_id": "ch01-sc01",
        "shadow_event": {"action": "track", "characters": ["Aragorn"], "motifs": ["patience"]},
    }
    unrelated = {
        "scene_id": "ch01-sc01",
        "shadow_event": {"action": "feast", "characters": ["Bilbo"], "motifs": ["comfort"]},
    }

    assert _selected_sample_prior_score(aligned, priors) > _selected_sample_prior_score(unrelated, priors)
