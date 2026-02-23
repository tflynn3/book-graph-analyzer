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
    cmd = ["story", "beats", "expand", "--project", "det-proj", "--projects-dir", str(tmp_path)]
    r1 = runner.invoke(main, cmd)
    assert r1.exit_code == 0, r1.output
    p1 = json.loads((proj / "shadow_beats.json").read_text(encoding="utf-8"))

    r2 = runner.invoke(main, cmd)
    assert r2.exit_code == 0, r2.output
    p2 = json.loads((proj / "shadow_beats.json").read_text(encoding="utf-8"))

    assert p1["seed"] == p2["seed"]
    assert [b["beat_id"] for b in p1["beats"]] == [b["beat_id"] for b in p2["beats"]]
    assert p1["validation"]["cause_ref_issues"] == []


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
