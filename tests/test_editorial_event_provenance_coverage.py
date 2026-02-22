from book_graph_analyzer.spatiotemporal.models import NormalizedTime, SpatiotemporalEvent
from book_graph_analyzer.worldbible.editorial import validate_event_provenance_coverage


def _ev(event_id: str, *, source_book: str | None, source_passage_id: str | None) -> SpatiotemporalEvent:
    return SpatiotemporalEvent(
        id=event_id,
        entity_id="char_bilbo",
        time=NormalizedTime(),
        source_book=source_book,
        source_passage_id=source_passage_id,
    )


def test_event_provenance_coverage_flags_missing():
    events = [
        _ev("e1", source_book="The Hobbit", source_passage_id="ch1"),
        _ev("e2", source_book="The Hobbit", source_passage_id=None),
    ]
    result = validate_event_provenance_coverage(events, max_missing_ratio=0.0)
    assert result.checked_count == 2
    assert result.missing_count == 1
    assert result.is_valid is False


def test_event_provenance_coverage_passes_threshold():
    events = [
        _ev("e1", source_book="The Hobbit", source_passage_id="ch1"),
        _ev("e2", source_book="The Hobbit", source_passage_id="ch2"),
    ]
    result = validate_event_provenance_coverage(events, max_missing_ratio=0.05)
    assert result.missing_ratio == 0.0
    assert result.is_valid is True
