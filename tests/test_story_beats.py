import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.story_cli import _compute_dynamic_beat_budget


def test_dynamic_beat_budget_floor_before_clamp():
    # 250/2=125; shaped for ordinal 2 => 135.0 then floor should keep integer behavior
    assert _compute_dynamic_beat_budget(250, ordinal=2, beat_count=2) == 128
    assert _compute_dynamic_beat_budget(89.9, ordinal=1, beat_count=3) == 45


def test_shadow_beats_id_and_seed_are_deterministic(tmp_path):
    proj = tmp_path / "det-proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "project.json").write_text(json.dumps({"name": "Det", "slug": "det-proj"}, indent=2), encoding="utf-8")
    (proj / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": [], "style": {"target_words_per_scene": 333.7}}, indent=2), encoding="utf-8")
    (proj / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "det-proj",
                "chapters": [{"chapter_number": 1, "scenes": [{"scene_id": "ch01-sc01", "goal": "g1", "summary": "s1"}, {"scene_id": "ch01-sc02", "goal": "g2", "summary": "s2"}]}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    cmd = [
        "story",
        "beats",
        "expand",
        "--project",
        "det-proj",
        "--beats-per-scene",
        "2",
        "--projects-dir",
        str(tmp_path),
    ]
    r1 = runner.invoke(main, cmd)
    assert r1.exit_code == 0, r1.output
    p1 = json.loads((proj / "shadow_beats.json").read_text(encoding="utf-8"))

    r2 = runner.invoke(main, cmd)
    assert r2.exit_code == 0, r2.output
    p2 = json.loads((proj / "shadow_beats.json").read_text(encoding="utf-8"))

    assert p1["seed"] == p2["seed"]
    assert [b["beat_id"] for b in p1["beats"]] == [b["beat_id"] for b in p2["beats"]]
    assert [b["position"] for b in p1["beats"]] == [1, 2, 3, 4]
    for beat in p1["beats"]:
        assert beat["action"]
        assert isinstance(beat["participants"], list)
        assert isinstance(beat["motifs"], list)
        assert isinstance(beat["preconditions"], list)
        assert isinstance(beat["effects"], list)
        assert isinstance(beat["source_canon_node_ids"], list)
        assert isinstance(beat["style_register_hints"], dict)
        assert set(beat["scoring_breakdown"].keys()) == {"lore", "style", "coherence"}
    assert p1["validation"]["cause_ref_issues"] == []


def test_story_beats_validate_lore_and_style_signals(tmp_path):
    proj = tmp_path / "beren-luthien-lore"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "project.json").write_text(json.dumps({"name": "BL", "slug": "beren-luthien-lore"}, indent=2), encoding="utf-8")
    (proj / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": [], "style": {"target_words_per_scene": 900}}, indent=2), encoding="utf-8")
    (proj / "shadow_beats.json").write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "beat_id": "ch01-sc01-b01-a",
                        "position": 1,
                        "beat_type": "setup",
                        "cause_refs": [],
                        "failed_constraints": [],
                        "action": "establish",
                        "participants": ["Frodo"],
                        "motifs": ["shadow"],
                        "preconditions": ["scene-goal:x"],
                        "effects": ["effect:x"],
                        "source_canon_node_ids": [],
                        "style_register_hints": {"beat_share": 0.1},
                        "prose_budget_words": 220,
                        "scoring_breakdown": {"lore": 0.4, "style": 0.7, "coherence": 0.8},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["story", "beats", "validate", "--project", "beren-luthien-lore", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    report = json.loads((proj / "shadow_beats_validation.json").read_text(encoding="utf-8"))
    codes = {i["code"] for i in report["issues"]}
    assert "OUT_OF_DOMAIN_PARTICIPANT" in codes
    assert "CANON_GROUNDING_WEAK" in codes
    assert "STYLE_BUDGET_MISMATCH" in codes


def test_shadow_beats_cause_refs_are_existing_and_prior(tmp_path):
    proj = tmp_path / "cause-proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "project.json").write_text(json.dumps({"name": "Cause", "slug": "cause-proj"}, indent=2), encoding="utf-8")
    (proj / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": [], "style": {}}, indent=2), encoding="utf-8")
    (proj / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "cause-proj",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "scenes": [
                            {"scene_id": "ch01-sc01", "goal": "g1", "summary": "s1"},
                            {"scene_id": "ch01-sc02", "goal": "g2", "summary": "s2"},
                            {"scene_id": "ch01-sc03", "goal": "g3", "summary": "s3"},
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["story", "beats", "expand", "--project", "cause-proj", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output

    payload = json.loads((proj / "shadow_beats.json").read_text(encoding="utf-8"))
    beats = payload["beats"]
    ids = [b["beat_id"] for b in beats]
    pos_by_id = {b["beat_id"]: b["position"] for b in beats}

    assert beats[0]["cause_refs"] == []
    for beat in beats[1:]:
        assert len(beat["cause_refs"]) == 1
        ref = beat["cause_refs"][0]
        assert ref in ids
        assert pos_by_id[ref] < beat["position"]

    assert payload["validation"]["cause_ref_issues"] == []


def test_story_beats_expand_writes_artifact_and_sidecar(tmp_path):
    proj = tmp_path / "beats-proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "project.json").write_text(json.dumps({"name": "Beats", "slug": "beats-proj"}, indent=2), encoding="utf-8")
    (proj / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": ["spaceship"], "style": {}}, indent=2), encoding="utf-8")
    (proj / "shadow_selected.json").write_text(json.dumps({"selected": [{"candidate_id": "c1"}]}, indent=2), encoding="utf-8")
    (proj / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "beats-proj",
                "chapters": [{"chapter_number": 1, "scenes": [{"scene_id": "ch01-sc01", "goal": "setup", "summary": "no spaceship here"}]}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["story", "beats", "expand", "--project", "beats-proj", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output

    out = proj / "shadow_beats.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "shadow-beats-v1"
    assert payload["beats"]
    assert "failed_constraints" in payload["validation"]
    assert (proj / "shadow_beats_selected_sidecar.json").exists()


def test_story_beats_expand_emits_more_than_one_beat_per_scene_when_budget_gt_one(tmp_path):
    proj = tmp_path / "multi-beat-proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "project.json").write_text(json.dumps({"name": "MB", "slug": "multi-beat-proj"}, indent=2), encoding="utf-8")
    (proj / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": [], "style": {}}, indent=2), encoding="utf-8")
    (proj / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "multi-beat-proj",
                "chapters": [{"chapter_number": 1, "scenes": [{"scene_id": "ch01-sc01", "goal": "g1", "summary": "s1"}]}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "story",
            "beats",
            "expand",
            "--project",
            "multi-beat-proj",
            "--beats-per-scene",
            "3",
            "--projects-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads((proj / "shadow_beats.json").read_text(encoding="utf-8"))
    assert len(payload["beats"]) == 3
    ids = [b["beat_id"] for b in payload["beats"]]
    assert ids[0].startswith("ch01-sc01-b01-")
    assert ids[1].startswith("ch01-sc01-b02-")
    assert ids[2].startswith("ch01-sc01-b03-")


def test_story_beats_strict_validate_passes_for_generated_multi_beat_scenes(tmp_path):
    proj = tmp_path / "strict-multi"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "project.json").write_text(json.dumps({"name": "Strict Multi", "slug": "strict-multi"}, indent=2), encoding="utf-8")
    (proj / "constraints.json").write_text(json.dumps({"required_elements": [], "forbidden_terms": [], "style": {}}, indent=2), encoding="utf-8")
    (proj / "plan.json").write_text(
        json.dumps(
            {
                "project_slug": "strict-multi",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "scenes": [
                            {"scene_id": "ch01-sc01", "goal": "g1", "summary": "s1"},
                            {"scene_id": "ch01-sc02", "goal": "g2", "summary": "s2"},
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    expand = runner.invoke(
        main,
        ["story", "beats", "expand", "--project", "strict-multi", "--beats-per-scene", "2", "--projects-dir", str(tmp_path)],
    )
    assert expand.exit_code == 0, expand.output

    strict = runner.invoke(
        main,
        ["story", "beats", "validate", "--project", "strict-multi", "--strict", "--projects-dir", str(tmp_path)],
    )
    assert strict.exit_code == 0, strict.output
