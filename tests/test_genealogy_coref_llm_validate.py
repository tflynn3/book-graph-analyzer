import json

from book_graph_analyzer.worldbible.genealogy import (
    extract_genealogy_from_text,
    validate_llm_genealogy_proposals,
)


def _pairs(relations):
    return {(r.source_name, r.target_name, r.relation_type.value) for r in relations}


def test_coref_pronoun_resolution_local_window():
    text = "Aragorn, son of Arathorn, came to Rivendell. He father of Eldarion."
    rels = extract_genealogy_from_text(text, passage_id="p1")
    pairs = _pairs(rels)
    assert ("Aragorn", "Eldarion", "PARENT_OF") in pairs


def test_title_carry_over_between_adjacent_sentences():
    text = "King Aragorn son of Arathorn. The king father of Eldarion."
    rels = extract_genealogy_from_text(text, passage_id="p1")
    pairs = _pairs(rels)
    assert ("Aragorn", "Eldarion", "PARENT_OF") in pairs


def test_relation_contains_evidence_span_and_confidence_metadata():
    text = "Thingol father of Luthien."
    rel = next(r for r in extract_genealogy_from_text(text) if r.relation_type.value == "PARENT_OF")
    assert rel.evidence_start is not None and rel.evidence_end is not None
    assert rel.evidence_text
    assert 0.0 <= rel.confidence <= 1.0
    assert rel.resolution_confidence is not None


def test_llm_validator_rejects_bad_schema_and_evidence_and_low_confidence():
    text = "Aragorn son of Arathorn."
    proposals = [
        {"source_name": "Aragorn", "relation_type": "CHILD_OF"},  # schema missing
        {
            "source_name": "Aragorn",
            "target_name": "Arathorn",
            "relation_type": "CHILD_OF",
            "evidence_text": "bad span",
            "evidence_start": 0,
            "evidence_end": 5,
            "confidence": 0.95,
        },
        {
            "source_name": "Aragorn",
            "target_name": "Arathorn",
            "relation_type": "CHILD_OF",
            "evidence_text": "Aragorn son of Arathorn",
            "evidence_start": 0,
            "evidence_end": 24,
            "confidence": 0.2,
        },
    ]
    accepted, rejected = validate_llm_genealogy_proposals(text, proposals)
    assert not accepted
    assert len(rejected) == 3
    reason_codes = {r["reason_code"] for r in rejected}
    assert "schema_invalid" in reason_codes
    assert "evidence_misaligned" in reason_codes
    assert "low_confidence" in reason_codes


def test_llm_pipeline_accepts_valid_proposal_via_extractor():
    class _MockLLM:
        def generate(self, prompt: str) -> str:
            _ = prompt
            return json.dumps([
                {
                    "source_name": "Arwen",
                    "target_name": "Elrond",
                    "relation_type": "CHILD_OF",
                    "evidence_text": "Arwen daughter of Elrond",
                    "evidence_start": 0,
                    "evidence_end": 24,
                    "confidence": 0.78,
                }
            ])

    text = "Arwen daughter of Elrond."
    rels = extract_genealogy_from_text(text, llm_client=_MockLLM())
    assert ("Arwen", "Elrond", "CHILD_OF") in _pairs(rels)


def test_regression_recall_gain_vs_baseline_with_precision_guardrails():
    # Gold relations by source->target->type (inverse edges not counted in recall scoring)
    gold = {
        ("Aragorn", "Arathorn", "CHILD_OF"),
        ("Aragorn", "Eldarion", "PARENT_OF"),
        ("Arwen", "Elrond", "CHILD_OF"),
    }
    corpus = (
        "Aragorn, son of Arathorn, was the heir. "
        "He father of Eldarion. "
        "Lady Arwen daughter of Elrond."
    )

    # Baseline: only explicit Name relation Name pattern.
    baseline = {("Aragorn", "Arathorn", "CHILD_OF"), ("Arwen", "Elrond", "CHILD_OF")}

    extracted = {
        p for p in _pairs(extract_genealogy_from_text(corpus))
        if p[2] in {"CHILD_OF", "PARENT_OF"}
    }

    baseline_recall = len(gold & baseline) / len(gold)
    new_recall = len(gold & extracted) / len(gold)

    # Precision safeguard: no impossible self-links
    assert all(s != t for (s, t, _rt) in extracted)
    assert new_recall >= baseline_recall
    assert new_recall >= 1.0
