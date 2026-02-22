from __future__ import annotations

from book_graph_analyzer.lore.sociolinguistic_registers import (
    SociolinguisticRegisterClassifier,
    profile_corpus_registers,
)
from book_graph_analyzer.lore.depth import extract_lore_depth, link_broken_reference_candidates
from book_graph_analyzer.models.passage import Passage
from book_graph_analyzer.worldbible.editorial import detect_editorial_divergences
from book_graph_analyzer.worldbible.genealogy import (
    extract_genealogy_from_text,
    genealogy_to_json,
    build_descendant_tree,
)


def test_issue_47_acceptance_register_families_and_character_drift():
    samples = [
        {"entity_id": "char_galadriel", "order": 1, "text": "By vow and rite, thou shalt keep the hallowed oath."},
        {"entity_id": "char_galadriel", "order": 2, "text": "I keep record and annal in the lore-halls."},
        {"entity_id": "char_eomer", "order": 1, "text": "Raise shield and banner, captain, and march at dawn."},
        {"entity_id": "char_sam", "order": 1, "text": "We'll get bread and ale and head home by supper."},
        {"entity_id": "char_elrond", "order": 1, "text": "The lord of the house guards lineage and honor."},
    ]

    out = profile_corpus_registers(samples, classifier=SociolinguisticRegisterClassifier())

    # Acceptance: >= 4 register families detected.
    assert len(out.dominant_distribution) >= 4

    # Acceptance: per-character profile + chapter/order drift signal.
    assert "char_galadriel" in out.per_entity_latest
    assert any(d.current_register != d.baseline_register for d in out.strongest_drifts)


def test_issue_49_acceptance_multigenerational_and_trait_rationale_output():
    text = (
        "Elendil father of Isildur. "
        "Isildur father of Valandil. "
        "Valandil father of Eldacar."
    )
    relations = extract_genealogy_from_text(text, passage_id="p_lineage", house="House of Elendil")

    chain = build_descendant_tree(relations, character_id="char_elendil", depth=4)
    assert len(chain) >= 3

    # Acceptance: lineage-based trait rationale survives output artifacts.
    relations[0].inheritance_traits = ["kingship", "lineage-memory"]
    payload = genealogy_to_json(relations)
    assert any(r.get("inheritance_traits") for r in payload["relations"])


def test_issue_50_acceptance_artifacts_and_unresolved_contexts():
    text = (
        "They sang song of Elbereth beside the relic of Numenor. "
        "None remembered [[The Broken Crown]]. "
        "The unnamed artifact was carried east."
    )

    out = extract_lore_depth(text, source_book="The Silmarillion", passage_id="p_depth")
    assert out.artifacts, "Expected artifact/lore entities"
    assert out.broken_references, "Expected unresolved references"

    linked = link_broken_reference_candidates(out.broken_references)
    # Acceptance: unresolved references include context + inferred context type.
    assert any((r.context_before or r.context_after) and r.expected_type for r in linked)


def test_issue_51_acceptance_three_layers_and_inconsistency_reporting():
    passages = [
        Passage(
            id="p_core",
            text="Gil-galad fell in combat.",
            book="Silmarillion",
            chapter="1",
            chapter_num=1,
            paragraph_num=1,
            sentence_num=1,
            char_offset=0,
            source_id="src_silmarillion_1977",
            source_stratum="core_text",
            factual_claims={"gil_galad_fate": "fell"},
            avg_sentence_length=12,
            passive_ratio=0.10,
            dialogue_density=0.0,
        ),
        Passage(
            id="p_appendix",
            text="In appendix form, he was said to vanish westward.",
            book="Silmarillion",
            chapter="A",
            chapter_num=1,
            paragraph_num=2,
            sentence_num=1,
            char_offset=12,
            source_id="src_silmarillion_1977",
            source_stratum="appendix",
            factual_claims={"gil_galad_fate": "vanished"},
            avg_sentence_length=30,
            passive_ratio=0.45,
            dialogue_density=0.0,
        ),
        Passage(
            id="p_gloss",
            text="Glosses imply uncertain endings.",
            book="Silmarillion",
            chapter="G",
            chapter_num=1,
            paragraph_num=3,
            sentence_num=1,
            char_offset=22,
            source_id="src_silmarillion_1977",
            source_stratum="gloss",
            factual_claims={"gil_galad_fate": "unknown"},
            avg_sentence_length=7,
            passive_ratio=0.02,
            dialogue_density=0.2,
        ),
    ]

    divergences = detect_editorial_divergences(passages)
    represented_layers = {p.source_stratum for p in passages}

    assert len(represented_layers) >= 3
    assert any(d.kind == "factual" for d in divergences)
    assert any(d.kind == "style" for d in divergences)
