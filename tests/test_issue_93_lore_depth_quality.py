from click.testing import CliRunner


def test_candidate_linking_audit_buckets_classify_unresolved_refs():
    from book_graph_analyzer.lore.depth import (
        build_candidate_linking_audit_buckets,
        extract_lore_depth,
        link_broken_reference_candidates,
    )

    text = "[[the Enemy]] met [[zzz unknown token]] near the pass."
    result = extract_lore_depth(text, source_book="The Silmarillion", passage_id="p93-1")
    link_broken_reference_candidates(result.broken_references, book="The Silmarillion")

    buckets = build_candidate_linking_audit_buckets(result.broken_references)

    assert "no_candidates" in buckets
    assert "high_confidence_top_candidate" in buckets
    assert sum(len(v) for v in buckets.values()) == len(result.unresolved_queue)


def test_unresolved_quality_gate_reports_failures_when_coverage_low():
    from book_graph_analyzer.lore.depth import evaluate_unresolved_quality_gates, extract_lore_depth

    # Deliberately no candidate linking pass to keep candidate coverage low.
    result = extract_lore_depth("[[mystery sigil]]", source_book="LOTR", passage_id="p93-2", context_window=2)

    report = evaluate_unresolved_quality_gates(
        result.broken_references,
        min_context_coverage=1.0,
        min_candidate_coverage=1.0,
    )

    assert report["passed"] is False
    assert report["summary"]["total_unresolved"] == 1
    assert report["failures"]["no_candidates"]


def test_cli_worldbible_artifacts_fails_when_quality_gate_fails(tmp_path):
    from book_graph_analyzer.cli import main

    text_path = tmp_path / "quality_fail.txt"
    text_path.write_text("[[mystery sigil]]", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "worldbible",
            "artifacts",
            str(text_path),
            "--no-link-candidates",
            "--min-context-coverage",
            "1.0",
            "--min-candidate-coverage",
            "1.0",
        ],
    )

    assert result.exit_code == 2
    assert "Quality gate:" in result.output


def test_hobbit_alias_linking_improves_candidate_coverage_for_gate():
    from book_graph_analyzer.lore.depth import (
        evaluate_unresolved_quality_gates,
        extract_lore_depth,
        link_broken_reference_candidates,
    )

    text = "[[Bilbo]] found [[Arkenstone]] while hunting [[Smaug]]."
    out = extract_lore_depth(text, source_book="The Hobbit", passage_id="p93-hobbit")
    linked = link_broken_reference_candidates(out.broken_references, book="The Hobbit")

    report = evaluate_unresolved_quality_gates(
        linked,
        min_context_coverage=1.0,
        min_candidate_coverage=1.0,
    )

    assert report["passed"] is True
    assert report["summary"]["total_unresolved"] == 3
    assert report["summary"]["candidate_coverage"] == 1.0
    assert all(ref.candidates for ref in linked)


def test_linked_candidates_are_sorted_by_confidence_descending():
    from book_graph_analyzer.lore.depth import extract_lore_depth, link_broken_reference_candidates

    out = extract_lore_depth("[[Bilbo Baggins]]", source_book="The Hobbit", passage_id="p93-sort")
    linked = link_broken_reference_candidates(out.broken_references, book="The Hobbit", max_candidates=3)

    assert linked
    confs = [c.confidence for c in linked[0].candidates]
    assert confs == sorted(confs, reverse=True)


def test_hobbit_alias_linking_respects_max_candidates_limit():
    from book_graph_analyzer.lore.depth import extract_lore_depth, link_broken_reference_candidates

    out = extract_lore_depth("[[Bilbo]]", source_book="The Hobbit", passage_id="p93-max")
    linked = link_broken_reference_candidates(out.broken_references, book="The Hobbit", max_candidates=1)

    assert linked
    assert len(linked[0].candidates) <= 1
