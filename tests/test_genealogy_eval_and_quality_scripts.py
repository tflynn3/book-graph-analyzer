import json
from pathlib import Path

from scripts.check_genealogy_graph_quality import check
from scripts.eval_genealogy import _load_gold, evaluate


def test_eval_genealogy_reports_overall_and_per_book_metrics():
    rows = _load_gold(Path("tests/fixtures/genealogy_gold.jsonl"))
    report = evaluate(rows)
    assert "overall" in report
    assert "by_book" in report
    assert report["overall"]["precision"] >= 0.9
    assert report["overall"]["recall"] >= 0.75


def test_graph_quality_gate_passes_on_current_eval_report():
    report = json.loads(Path("data/output/genealogy_eval_report.json").read_text(encoding="utf-8"))
    quality = check(report)
    assert quality["thresholds"]["precision_ok"] is True
    assert quality["thresholds"]["recall_ok"] is True
    assert quality["graph_integrity"]["passed"] is True
