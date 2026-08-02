import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.draft_doctor import analyze_draft


BROKEN_PARAGRAPH = (
    "The old powers did not need to enter the road, for hidden labour and common weather "
    "had already taught mercy, hope, warning, shadow, burden, service, and truth to walk unseen."
)


def _write_broken_draft(tmp_path):
    ch1 = tmp_path / "chapter_01.md"
    ch1.write_text(
        f"""# A Name in the Dark

In Road, {{ranger}} found maps and Road before dawn.

{BROKEN_PARAGRAPH}

'The road must bear truth before hope can speak,' said Gandalf.

'The road must bear truth before hope can speak,' answered Aragorn.

rope, lamps, maps, folded letters, fish bones, and muddy water lay beside the road.

It had to.

For now, it was enough.

## A Weight of Mercy

Mercy and hope and warning and truth became a burden of service in the shadow. The burden of service became warning and truth and hope, yet no one moved and nothing changed.

## Marsh Capture

In the Dead Marshes Aragorn found Gollum crawling under reeds, cut the rope from his pack, caught the creature by the wrist, bound him, and dragged him away from the black water before the moon went down.

'Baggins bites. Shire bites,' Gollum said.
""",
        encoding="utf-8",
    )
    ch2 = tmp_path / "chapter_02.md"
    ch2.write_text(
        f"""# Questions Under the Trees

{BROKEN_PARAGRAPH}

'Baggins bites. Shire bites,' Gollum said.

## Guarded Questioning

In Mirkwood Gandalf questioned Gollum beneath guarded lamps. Gollum answered crookedly, named Baggins, and revealed that the Shire had become a road for danger. Aragorn heard the answer, changed the watch, and sent warning west.

'Kind hands tie knots,' Gollum said.

'Kind hands tie knots,' Gollum said.
""",
        encoding="utf-8",
    )
    return tmp_path


def test_draft_doctor_cli_reports_repair_plan_and_markdown(tmp_path):
    draft_dir = _write_broken_draft(tmp_path)
    output = tmp_path / "report.json"

    result = CliRunner().invoke(
        main,
        ["draft", "doctor", str(draft_dir), "--profile", "tolkien", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    markdown = output.with_suffix(".md")
    assert markdown.exists()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["strict_validation"]["pass"] is False
    assert report["strict_validation"]["blocking_issue_count"] > 0
    categories = {issue["category"] for issue in report["issues"]}
    assert "placeholder_continuity_lint" in categories
    assert "repetition_graph" in categories
    assert "scene_causality" in categories
    assert "tolkien_register_balance" in categories
    assert "voice_differentiation" in categories
    assert "object_causality" in categories
    assert "ending_cadence_overload" in categories

    plan_categories = {item["category"] for item in report["ranked_repair_plan"]}
    assert "placeholder_continuity_lint" in plan_categories
    assert "scene_causality" in plan_categories
    assert "salvage_passages" in plan_categories

    issue_text = json.dumps(report["issues"])
    assert "brace_placeholder" in issue_text
    assert "duplicate_paragraph" in issue_text
    assert "recycled_dialogue" in issue_text
    assert "gollum_formula_repeat" in issue_text

    salvage_labels = {item["label"] for item in report["salvage_passages"]}
    assert "Marsh capture" in salvage_labels
    assert "Guarded questioning" in salvage_labels
    md_text = markdown.read_text(encoding="utf-8")
    assert "Draft Quality Repair Report" in md_text
    assert "Ranked Repair Plan" in md_text
    assert "Marsh capture" in md_text


def test_draft_doctor_strict_mode_fails_after_writing_report(tmp_path):
    draft_dir = _write_broken_draft(tmp_path)
    output = tmp_path / "strict-report.json"

    result = CliRunner().invoke(
        main,
        ["draft", "doctor", str(draft_dir), "--profile", "tolkien", "--strict", "--output", str(output)],
    )

    assert result.exit_code != 0
    assert "Strict draft validation failed" in result.output
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["strict_validation"]["pass"] is False


def test_draft_doctor_does_not_treat_named_road_conjunction_as_placeholder(tmp_path):
    chapter = tmp_path / "chapter_01.md"
    chapter.write_text(
        "# Watch\n\nThe strangers watched the East Road and withdrew before dawn.\n",
        encoding="utf-8",
    )

    report = analyze_draft(
        tmp_path,
        min_chapter_words=0,
        min_scene_words=0,
        min_total_words=0,
    )

    placeholder_messages = [
        issue["message"]
        for issue in report["issues"]
        if issue["category"] == "placeholder_continuity_lint"
    ]
    assert all("road_and_object" not in message for message in placeholder_messages)


def test_draft_doctor_blocks_editorial_leaks_and_modern_process_diction(tmp_path):
    chapter = tmp_path / "chapter_01.md"
    chapter.write_text(
        "# Watch\n\n"
        "By the end of the chapter the warning had to stand.\n\n"
        "* * *\n\n"
        "The investigation needed a stopping rule, and mercy entered as a controlled risk.\n",
        encoding="utf-8",
    )

    report = analyze_draft(
        tmp_path,
        min_chapter_words=0,
        min_scene_words=0,
        min_total_words=0,
    )

    assert report["strict_validation"]["pass"] is False
    assert any(
        issue["category"] == "placeholder_continuity_lint"
        and issue["evidence"].get("pattern") == "meta_end_of_chapter"
        for issue in report["issues"]
    )
    assert any(
        issue["category"] == "tolkien_register_balance"
        and issue["evidence"].get("kind") == "modern_analytical_diction"
        and set(issue["evidence"].get("patterns", [])) == {"stopping_rule", "controlled_risk"}
        for issue in report["issues"]
    )


def test_draft_doctor_strict_mode_passes_clean_draft(tmp_path):
    draft_dir = tmp_path / "clean"
    draft_dir.mkdir()
    (draft_dir / "chapter_01.md").write_text(
        """# Clean Trail

Aragorn found the print below the willow, crossed the ford before noon, and warned the northern watch before rain reached the bank.

'The heel turns east; I will follow before the mud closes,' answered Aragorn.

* * *

Gandalf questioned the ferryman at dusk, learned which boat had gone missing, and sent the answer west by a rider who knew the old road.

'Keep Baggins out of the tale until I stand at his door,' said Gandalf.
""",
        encoding="utf-8",
    )
    output = tmp_path / "clean-report.json"

    result = CliRunner().invoke(
        main,
        [
            "draft",
            "doctor",
            str(draft_dir),
            "--profile",
            "tolkien",
            "--strict",
            "--min-chapter-words",
            "1",
            "--min-scene-words",
            "1",
            "--min-total-words",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["strict_validation"]["pass"] is True
    assert report["strict_validation"]["blocking_issue_count"] == 0


def test_draft_doctor_strict_mode_fails_clean_but_short_draft_by_default(tmp_path):
    draft_dir = tmp_path / "short-clean"
    draft_dir.mkdir()
    (draft_dir / "chapter_01.md").write_text(
        """# Clean But Short

Aragorn found the track beside the ford, crossed before rain, and warned the watch by dusk.

## Second Turn

Gandalf questioned the keeper, learned which boat had vanished, and sent the answer west.
""",
        encoding="utf-8",
    )
    output = tmp_path / "short-clean-report.json"

    result = CliRunner().invoke(
        main,
        ["draft", "doctor", str(draft_dir), "--profile", "tolkien", "--strict", "--output", str(output)],
    )

    assert result.exit_code != 0
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["strict_validation"]["pass"] is False
    assert report["strict_validation"]["blocking_counts_by_category"]["draft_fullness"] >= 1
    assert report["summary"]["fullness"]["min_chapter_words"] == 3000
    assert report["summary"]["fullness"]["min_total_words"] == 36000


def test_strict_validation_blocks_medium_material_findings():
    from book_graph_analyzer.draft_doctor import _strict_validation

    report = _strict_validation(
        [
            {
                "id": "voice-medium",
                "category": "voice_differentiation",
                "severity": "medium",
            },
            {
                "id": "polish-low",
                "category": "voice_differentiation",
                "severity": "low",
            },
        ]
    )

    assert report["pass"] is False
    assert report["blocking_issue_count"] == 1
    assert report["blocking_issue_ids"] == ["voice-medium"]
    assert report["rules"]["blocking_severities"] == ["high", "medium"]


def test_draft_doctor_groups_issues_by_chapter(tmp_path):
    draft_dir = _write_broken_draft(tmp_path)
    report = analyze_draft(
        draft_dir,
        profile="tolkien",
        min_chapter_words=1,
        min_scene_words=1,
        min_total_words=1,
    )

    assert report["issues_by_chapter"]["1"]
    assert report["issues_by_scene"]["chapter_01/scene_01"]
    assert report["issues_by_severity"]["high"]
    assert report["issues_by_chapter"]["2"]
    first_issue = report["issues"][0]
    assert {"chapter", "scene", "paragraph"}.issubset(first_issue["location"])
