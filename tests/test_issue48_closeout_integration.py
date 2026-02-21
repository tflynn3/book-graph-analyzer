from pathlib import Path

from book_graph_analyzer.spatiotemporal import CorpusReconciler


def test_issue48_corpus_reconcile_with_real_fixtures_enriches_source_metadata():
    """End-to-end: real corpus fixtures -> bridge -> cross-book reconcile.

    Uses representative repository fixtures from data/output and validates that
    source/editorial metadata is preserved on normalized events.
    """
    repo_root = Path(__file__).resolve().parents[1]
    hobbit = repo_root / "data" / "output" / "hobbit_events.json"
    ut = repo_root / "data" / "output" / "unfinished_tales_events.json"

    assert hobbit.exists(), f"Missing fixture: {hobbit}"
    assert ut.exists(), f"Missing fixture: {ut}"

    reconciler = CorpusReconciler(extract_causal=False)
    hobbit_count = reconciler.add_book_from_json(hobbit, "hobbit", "The Hobbit")
    ut_count = reconciler.add_book_from_json(ut, "unfinished_tales", "Unfinished Tales")

    assert hobbit_count > 0
    assert ut_count > 0

    result = reconciler.reconcile(check_causal=False)
    assert result.total_events == hobbit_count + ut_count
    assert len(result.books) == 2

    # Event IDs are namespaced per book to avoid collisions during cross-book merge.
    assert all(e.id.startswith("hobbit:") for e in result.books[0].events)
    assert all(e.id.startswith("unfinished_tales:") for e in result.books[1].events)

    # Metadata should be inferred from source titles and carried end-to-end.
    hobbit_event = result.books[0].events[0]
    ut_event = result.books[1].events[0]

    assert hobbit_event.source_book == "The Hobbit"
    assert hobbit_event.editorial_status == "published"
    assert hobbit_event.source_authority_weight is not None

    assert ut_event.source_book == "Unfinished Tales"
    assert ut_event.editorial_status == "unfinished"
    assert ut_event.source_authority_weight is not None
