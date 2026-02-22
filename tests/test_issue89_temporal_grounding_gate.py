from pathlib import Path

from book_graph_analyzer.lore.events import Event
from book_graph_analyzer.spatiotemporal import (
    ExtractionBridge,
    TemporalGroundingGate,
    compute_temporal_grounding_metrics,
)


def _load_hobbit_lore_events() -> list[Event]:
    import json

    repo_root = Path(__file__).resolve().parents[1]
    hobbit = repo_root / "data" / "output" / "hobbit_events.json"
    payload = json.loads(hobbit.read_text(encoding="utf-8"))
    raw = payload.get("events", payload)
    if isinstance(raw, dict):
        raw = list(raw.values())
    return [Event.from_dict(e) for e in raw]


def test_issue89_hobbit_temporal_grounding_backfill_improves_coverage():
    events = _load_hobbit_lore_events()

    bridge = ExtractionBridge()
    no_backfill = bridge.bridge_events(events, source_book="The Hobbit", apply_backfill=False)
    with_backfill = bridge.bridge_events(events, source_book="The Hobbit", apply_backfill=True)

    before = compute_temporal_grounding_metrics(no_backfill.events)
    after = compute_temporal_grounding_metrics(with_backfill.events)

    assert after.grounded_ratio >= before.grounded_ratio
    assert after.era_ratio >= before.era_ratio
    assert after.year_or_interval_ratio >= before.year_or_interval_ratio


def test_issue89_hobbit_gate_can_fail_and_pass_with_explicit_thresholds():
    events = _load_hobbit_lore_events()
    bridge = ExtractionBridge()
    st_events_no_backfill = bridge.bridge_events(events, source_book="The Hobbit", apply_backfill=False).events
    st_events = bridge.bridge_events(events, source_book="The Hobbit", apply_backfill=True).events

    strict_gate = TemporalGroundingGate(
        min_grounded_ratio=0.90,
        min_era_ratio=0.90,
        min_year_or_interval_ratio=0.20,
    )
    permissive_gate = TemporalGroundingGate(
        min_grounded_ratio=0.90,
        min_era_ratio=0.90,
        min_year_or_interval_ratio=0.0,
    )

    strict_result = strict_gate.evaluate(st_events)
    permissive_result = permissive_gate.evaluate(st_events)
    strict_result_no_backfill = strict_gate.evaluate(st_events_no_backfill)

    assert strict_result_no_backfill.passed is False
    assert strict_result.passed is True
    assert permissive_result.passed is True
    assert permissive_result.metrics.to_dict()["metrics_version"] == "temporal-grounding-v1"
