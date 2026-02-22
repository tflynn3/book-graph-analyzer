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


def test_pipeline_worldbuilding_hobbit_gate_requires_non_population_genealogy(tmp_path):
    hobbit_text = tmp_path / "the_hobbit.txt"
    hobbit_text.write_text(
        "Hobbits are a quiet folk and live in the Shire.",
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

    assert result.exit_code != 0
    assert "Hobbit acceptance gate failed" in result.output


def test_pipeline_worldbuilding_hobbit_gate_accepts_non_population_genealogy(tmp_path):
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
