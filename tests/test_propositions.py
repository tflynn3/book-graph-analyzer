"""Tests for dense proposition extraction."""

from __future__ import annotations

from book_graph_analyzer.extract.ner import ExtractedEntity
from book_graph_analyzer.extract.propositions import PropositionExtractor
from book_graph_analyzer.extract.resolver import ResolvedEntity
from book_graph_analyzer.ingest.splitter import Passage
from book_graph_analyzer.models.propositions import ArgumentRole, PropositionKind, ReferenceClass


def _passage(text: str) -> Passage:
    return Passage(
        id="p-test",
        text=text,
        book="Test Book",
        chapter="Chapter 1",
        chapter_num=1,
        paragraph_num=1,
        sentence_num=1,
        char_offset=0,
    )


def _entity(
    text: str,
    canonical_id: str,
    canonical_name: str,
    entity_type: str,
    *,
    label: str,
    start: int,
) -> ResolvedEntity:
    return ResolvedEntity(
        extracted=ExtractedEntity(
            text=text,
            label=label,
            start_char=start,
            end_char=start + len(text),
        ),
        canonical_id=canonical_id,
        canonical_name=canonical_name,
        entity_type=entity_type,
        confidence=1.0,
        is_new=False,
    )


def test_proposition_extractor_builds_dense_sentence_layer():
    text = "Aragorn carried the sword to Rivendell with Frodo, the weary hobbit."
    passage = _passage(text)
    entities = [
        _entity("Aragorn", "aragorn", "Aragorn", "character", label="PERSON", start=0),
        _entity("the sword", "anduril", "Anduril", "object", label="OBJECT", start=16),
        _entity("Rivendell", "rivendell", "Rivendell", "place", label="PLACE", start=29),
        _entity("Frodo", "frodo", "Frodo Baggins", "character", label="PERSON", start=45),
    ]

    extractor = PropositionExtractor()
    result = extractor.extract_from_passage(passage, entities)

    assert len(result.propositions) >= 2
    assert any(prop.kind == PropositionKind.POSSESSION for prop in result.propositions)

    role_sets = [{arg.role for arg in prop.arguments} for prop in result.propositions]
    assert any(ArgumentRole.AGENT in roles and ArgumentRole.PATIENT in roles for roles in role_sets)
    carry = next(prop for prop in result.propositions if prop.predicate_lemma == "carry")
    assert any(
        arg.role == ArgumentRole.DESTINATION and arg.surface == "Rivendell" and arg.prep == "to"
        for arg in carry.arguments
    )
    assert any(
        arg.role == ArgumentRole.COMPANION and arg.surface == "Frodo" and arg.prep == "with"
        for arg in carry.arguments
    )
    assert any(
        arg.surface == "the weary hobbit" and arg.phrase_modifiers == ["weary"]
        for prop in result.propositions
        for arg in prop.arguments
    )


def test_proposition_extractor_promotes_noun_phrase_modifiers_and_relations():
    text = "Bilbo announced a magnificent party in Hobbiton."
    passage = _passage(text)
    entities = [
        _entity("Bilbo", "bilbo", "Bilbo Baggins", "character", label="PERSON", start=0),
        _entity(
            "Hobbiton",
            "hobbiton",
            "Hobbiton",
            "place",
            label="PLACE",
            start=text.index("Hobbiton"),
        ),
    ]

    extractor = PropositionExtractor()
    result = extractor.extract_from_passage(passage, entities)

    announce = next(prop for prop in result.propositions if prop.predicate_lemma == "announce")
    party = next(arg for arg in announce.arguments if arg.surface == "a magnificent party")

    assert party.phrase_id
    assert party.phrase_head == "party"
    assert party.phrase_modifiers == ["magnificent"]
    assert party.prep is None
    assert not any(prop.predicate_lemma == "announce_in" for prop in result.propositions)

    relation = next(rel for rel in announce.noun_phrase_relations if rel.source_phrase_id == party.phrase_id)
    assert relation.relation_type == "LOCATED_IN"
    assert relation.prep == "in"
    assert relation.target_surface == "Hobbiton"
    assert relation.target_entity_id == "hobbiton"
    assert not any(prop.predicate_lemma == "has_attribute" for prop in result.propositions)


def test_proposition_extractor_folds_verb_prepositions_into_parent_event():
    text = "Bilbo celebrated his birthday with a party."
    passage = _passage(text)
    entities = [
        _entity("Bilbo", "bilbo", "Bilbo Baggins", "character", label="PERSON", start=0),
    ]

    extractor = PropositionExtractor()
    result = extractor.extract_from_passage(passage, entities)

    celebrate = next(prop for prop in result.propositions if prop.predicate_lemma == "celebrate")
    party = next(arg for arg in celebrate.arguments if arg.surface == "a party")

    assert party.role == ArgumentRole.INSTRUMENT
    assert party.prep == "with"
    assert party.phrase_head == "party"
    assert not any(prop.predicate_lemma == "celebrate_with" for prop in result.propositions)


def test_proposition_extractor_classifies_unresolved_reference_types():
    extractor = PropositionExtractor()
    doc = extractor.nlp("I carried his hand to Mordor with that through the gate.")

    pronoun = extractor._build_argument(ArgumentRole.AGENT, doc[0:1], [])
    body_part = extractor._build_argument(ArgumentRole.PATIENT, doc[2:4], [])
    canon_candidate = extractor._build_argument(ArgumentRole.DESTINATION, doc[5:6], [])
    deictic = extractor._build_argument(ArgumentRole.TOPIC, doc[7:8], [])
    bridging = extractor._build_argument(ArgumentRole.LOCATION, doc[9:11], [])

    assert pronoun is not None and pronoun.reference_class == ReferenceClass.PRONOUN
    assert body_part is not None and body_part.reference_class == ReferenceClass.BODY_PART
    assert canon_candidate is not None and canon_candidate.reference_class == ReferenceClass.CANON_CANDIDATE
    assert deictic is not None and deictic.reference_class == ReferenceClass.DISCOURSE_DEICTIC
    assert bridging is not None and bridging.reference_class == ReferenceClass.BRIDGING


def test_proposition_extractor_reduces_false_canon_candidates():
    extractor = PropositionExtractor()
    doc = extractor.nlp("to last he Beregond the Ring")

    prep = extractor._build_argument(ArgumentRole.DESTINATION, doc[0:1], [])
    adjective = extractor._build_argument(ArgumentRole.ATTRIBUTE, doc[1:2], [])
    pronoun = extractor._build_argument(ArgumentRole.AGENT, doc[2:3], [])
    character = extractor._build_argument(ArgumentRole.AGENT, doc[3:4], [])
    ring = extractor._build_argument(ArgumentRole.PATIENT, doc[4:6], [])

    assert prep is not None and prep.reference_class != ReferenceClass.CANON_CANDIDATE
    assert adjective is not None and adjective.reference_class != ReferenceClass.CANON_CANDIDATE
    assert pronoun is not None and pronoun.reference_class == ReferenceClass.PRONOUN
    assert character is not None and character.reference_class == ReferenceClass.CANON_CANDIDATE
    assert ring is not None and ring.reference_class == ReferenceClass.CANON_CANDIDATE


def test_proposition_extractor_resolves_first_person_quote_to_speaker():
    text = '"I will go to Rivendell," said Aragorn.'
    passage = _passage(text)
    entities = [
        _entity("Rivendell", "rivendell", "Rivendell", "place", label="PLACE", start=14),
        _entity("Aragorn", "aragorn", "Aragorn", "character", label="PERSON", start=31),
    ]

    extractor = PropositionExtractor()
    result = extractor.extract_from_passage(passage, entities)

    movement_props = [prop for prop in result.propositions if prop.predicate_lemma == "go"]
    assert movement_props, "expected a movement proposition inside the quote"
    assert result.quotes and result.quotes[0].speaker_entity_id == "aragorn"
    assert movement_props[0].quote_id == result.quotes[0].id
    assert any(
        any(arg.role == ArgumentRole.AGENT and arg.entity_id == "aragorn" for arg in prop.arguments)
        for prop in movement_props
    )


def test_proposition_extractor_resolves_second_person_quote_to_addressee():
    text = '"You must go to Rivendell," Aragorn said to Frodo.'
    passage = _passage(text)
    entities = [
        _entity("Rivendell", "rivendell", "Rivendell", "place", label="PLACE", start=16),
        _entity("Aragorn", "aragorn", "Aragorn", "character", label="PERSON", start=29),
        _entity("Frodo", "frodo", "Frodo Baggins", "character", label="PERSON", start=45),
    ]

    extractor = PropositionExtractor()
    result = extractor.extract_from_passage(passage, entities)

    movement_props = [prop for prop in result.propositions if prop.predicate_lemma == "go"]
    assert movement_props, "expected a movement proposition inside the quote"
    assert result.quotes and result.quotes[0].speaker_entity_id == "aragorn"
    assert result.quotes[0].addressee_entity_id == "frodo"
    assert movement_props[0].quote_id == result.quotes[0].id
    assert any(
        any(arg.role == ArgumentRole.AGENT and arg.entity_id == "frodo" for arg in prop.arguments)
        for prop in movement_props
    )


def test_proposition_extractor_uses_recent_entities_for_cross_passage_pronouns():
    first = _passage("Frodo entered Mordor.")
    second = _passage("He carried the Ring.")
    first.id = "p1"
    second.id = "p2"

    extractor = PropositionExtractor()
    extractor.extract_from_passage(
        first,
        [
            _entity("Frodo", "frodo", "Frodo Baggins", "character", label="PERSON", start=0),
            _entity("Mordor", "mordor", "Mordor", "place", label="PLACE", start=14),
        ],
    )
    result = extractor.extract_from_passage(
        second,
        [
            _entity("the Ring", "one_ring", "The One Ring", "object", label="OBJECT", start=11),
        ],
    )

    possession_props = [prop for prop in result.propositions if prop.predicate_lemma == "carry"]
    assert possession_props, "expected a possession proposition in the second passage"
    assert any(
        any(arg.role == ArgumentRole.AGENT and arg.entity_id == "frodo" for arg in prop.arguments)
        for prop in possession_props
    )
