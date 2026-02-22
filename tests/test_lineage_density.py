import json

from book_graph_analyzer.worldbible.lineage_density import (
    LineageThreshold,
    build_lineages_from_corpus,
    compute_lineage_metrics,
    threshold_pass,
)


def test_build_lineages_from_corpus_uses_alias_hits():
    catalog = {
        "place_rivendell": {"rivendell", "imladris"},
        "char_gandalf": {"gandalf", "mithrandir"},
    }
    corpus = "Rivendell was fair. Imladris endured. Gandalf, called Mithrandir, came to Rivendell."

    lineages = build_lineages_from_corpus(corpus, catalog, source_passage_id="fellowship_of_ring", min_mentions=1)

    assert len(lineages) == 2
    riv = next(x for x in lineages if x["entity_id"] == "place_rivendell")
    assert len(riv["forms"]) >= 2
    assert riv["derivations"], "expected DERIVED_FROM candidates"


def test_compute_metrics_and_threshold():
    payload = [
        {
            "entity_id": "place_rivendell",
            "forms": [{"id": "a"}, {"id": "b"}],
            "derivations": [{"source_form_id": "a", "target_form_id": "b"}],
        }
    ]
    metrics = compute_lineage_metrics(payload, {"place_rivendell"})
    assert metrics["lineages"] == 1
    assert metrics["forms"] == 2
    assert metrics["derivations"] == 1
    assert metrics["join_rate"] == 1.0

    assert threshold_pass(metrics, LineageThreshold(min_lineages=1, min_forms=2, min_derivations=1, min_join_rate=0.95))
    assert not threshold_pass(metrics, LineageThreshold(min_lineages=2, min_forms=2, min_derivations=1, min_join_rate=0.95))
