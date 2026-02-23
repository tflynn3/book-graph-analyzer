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
    assert "sample-shadow" in main.commands["story"].commands
    assert "score-shadow" in main.commands["story"].commands
    assert "select-shadow" in main.commands["story"].commands
    assert "solve" in main.commands["story"].commands
    assert "draft" in main.commands["story"].commands
    assert "audit" in main.commands["story"].commands
    assert "beats" in main.commands["story"].commands
    assert "expand" in main.commands["story"].commands["beats"].commands


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


def test_story_draft_enforces_required_terms(tmp_path):
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
                    },
                    {"id": "shadow-ch01-sc01"},
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

    monkeypatch.setattr(story_module, "_render_grounded_chapter_text", lambda chapter, chapter_rows, graph_node_by_id, required_terms: ("# Chapter\n\nNo required term appears here.\n", []))

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


def test_story_grow_shadow_applies_project_priors_and_audit_reports_domain_metrics(tmp_path):
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
