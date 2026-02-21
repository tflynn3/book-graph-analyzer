from unittest.mock import MagicMock

from click.testing import CliRunner


class _FakeLLM:
    def generate(self, prompt: str):
        return '[{"mention_text": "the Enemy", "confidence": 0.74, "reason": "ambiguous epithet"}]'


def test_extract_lore_depth_context_windows_and_expected_type():
    from book_graph_analyzer.lore.depth import extract_lore_depth

    text = "In the black vault they sought [[forgotten crown]] but no chronicler knew its owner."
    result = extract_lore_depth(text, source_book="Silmarillion", passage_id="p2", context_window=25)

    assert result.broken_references
    ref = result.broken_references[0]
    assert ref.context_before is not None
    assert ref.context_after is not None
    assert ref.expected_type in {"artifact", "unknown"}


def test_llm_fallback_can_add_unresolved_refs():
    from book_graph_analyzer.lore.depth import extract_lore_depth

    result = extract_lore_depth(
        "No explicit marker here, but the Enemy is invoked.",
        source_book="LOTR",
        passage_id="p3",
        llm_client=_FakeLLM(),
    )

    assert any(r.mention_text.lower() == "the enemy" for r in result.broken_references)
    assert any(any(n.startswith("llm:") for n in r.provenance_notes) for r in result.broken_references)


def test_candidate_linking_uses_resolver_and_weights_queue():
    from book_graph_analyzer.lore.depth import extract_lore_depth, link_broken_reference_candidates

    text = "[[the Enemy]] rose again with unknown relic in hand."
    result = extract_lore_depth(text, source_book="The Silmarillion", passage_id="p4")
    linked = link_broken_reference_candidates(result.broken_references, era="First Age", book="The Silmarillion")

    assert linked
    first = linked[0]
    assert first.candidates
    assert first.candidates[0].canonical_id
    assert result.unresolved_queue[0].conflict_weight >= 0.3


def test_graph_writer_unresolved_queue_query_exists():
    from book_graph_analyzer.graph.writer import GraphWriter

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    writer = GraphWriter(driver=mock_driver)
    writer.query_unresolved_reference_queue(source_book="Sil", limit=10)
    assert mock_session.run.called


def test_cli_artifacts_accepts_slice2_flags(tmp_path):
    from book_graph_analyzer.cli import main

    text_path = tmp_path / "sample.txt"
    text_path.write_text("[[the Enemy]] and unknown relic.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["worldbible", "artifacts", str(text_path), "--context-window", "60", "--no-link-candidates"],
    )

    assert result.exit_code == 0
