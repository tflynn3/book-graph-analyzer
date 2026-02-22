from __future__ import annotations

import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main


def test_pipeline_worldbuilding_runs_genealogy_stage_and_writes_artifact(tmp_path):
    text_file = tmp_path / "silmarillion_lineage.txt"
    text_file.write_text("Elendil father of Isildur. Isildur father of Valandil.", encoding="utf-8")

    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "worldbuilding",
            str(text_file),
            "--pillars",
            "genealogy",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    output_path = out_dir / "silmarillion_lineage_genealogy.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload.get("relations", [])) > 0


def test_pipeline_worldbuilding_hobbit_gate_produces_non_zero_genealogy(tmp_path):
    hobbit_text = tmp_path / "the_hobbit.txt"
    hobbit_text.write_text(
        "Bilbo, son of Bungo Baggins, lived in the Shire.",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "worldbuilding",
            str(hobbit_text),
            "--title",
            "The Hobbit",
            "--pillars",
            "genealogy",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    output_path = out_dir / "the_hobbit_genealogy.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["relation_count"] > 0


def test_pipeline_worldbuilding_genealogy_includes_threshold_status(tmp_path):
    text_file = tmp_path / "twotowers.txt"
    text_file.write_text("Aragorn son of Arathorn.", encoding="utf-8")

    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "worldbuilding",
            str(text_file),
            "--title",
            "The Two Towers",
            "--pillars",
            "genealogy",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "threshold" in result.output.lower()
    assert "FAIL" in result.output


def test_pipeline_worldbuilding_cultural_pillar_outputs_metrics(tmp_path):
    hobbit_text = tmp_path / "the_hobbit.txt"
    hobbit_text.write_text(
        "A hobbit lived in a hole in the ground, and among hobbits this was comfort.",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "pipeline",
            "worldbuilding",
            str(hobbit_text),
            "--title",
            "The Hobbit",
            "--pillars",
            "cultural",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((out_dir / "the_hobbit_cultures.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["culture_count"] >= 1
