"""Regression tests for the shared graph extraction path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from book_graph_analyzer.extract.book_pipeline import build_entity_clusters, build_entity_id_map
from book_graph_analyzer.extract.extractor import ExtractionResult
from book_graph_analyzer.extract.ner import ExtractedEntity
from book_graph_analyzer.extract.propositions import PropositionExtractionResult
from book_graph_analyzer.extract.relationships import RelationshipExtractionResult
from book_graph_analyzer.extract.resolver import ResolvedEntity
from book_graph_analyzer.graph.writer import GraphWriter
from book_graph_analyzer.ingest.splitter import Passage
from book_graph_analyzer.models.propositions import (
    ArgumentRole,
    ExtractedQuote,
    ExtractedProposition,
    NounPhraseRelation,
    PropositionArgument,
    PropositionKind,
    ReferenceClass,
)
from book_graph_analyzer.models.relationships import ExtractedRelationship, RelationshipType
from scripts.build_sentence_graph_review import (
    ReviewGraphBuilder,
    SeedTaxonomy,
    is_reviewable_proposition,
    is_reviewable_unresolved_argument,
)
from book_graph_analyzer.voice.dialogue import DialogueLine


def _passage(pid: str, text: str) -> Passage:
    return Passage(
        id=pid,
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
    canonical_id: str | None,
    canonical_name: str | None,
    entity_type: str,
    *,
    label: str = "PERSON",
    start: int = 0,
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
        is_new=canonical_id is None,
    )


def test_build_entity_clusters_preserves_types_and_aliases():
    results = [
        ExtractionResult(
            passage=_passage("p1", "Gandalf arrived at Bag End."),
            entities=[
                _entity("Gandalf", "gandalf", "Gandalf", "character"),
                _entity("Bag End", "bag_end", "Bag End", "place", label="PLACE", start=20),
            ],
            raw_extractions=[],
        ),
        ExtractionResult(
            passage=_passage("p2", "The Grey Pilgrim carried Sting."),
            entities=[
                _entity("The Grey Pilgrim", "gandalf", "Gandalf", "character"),
                _entity("Sting", "sting", "Sting", "object", label="OBJECT", start=25),
            ],
            raw_extractions=[],
        ),
    ]

    clusters = build_entity_clusters(results)

    assert clusters["gandalf"].mention_count == 2
    assert "The Grey Pilgrim" in clusters["gandalf"].aliases
    assert clusters["bag_end"].entity_type == "place"
    assert clusters["sting"].entity_type == "object"


def test_build_entity_id_map_adds_safe_voice_name_variants():
    results = [
        ExtractionResult(
            passage=_passage("p1", "Mr. Strider spoke softly."),
            entities=[
                _entity("Mr. Strider", "aragorn", "Aragorn", "character"),
                _entity("Strider's", "aragorn", "Aragorn", "character", start=4),
            ],
            raw_extractions=[],
        ),
    ]

    entity_map = build_entity_id_map(results)

    assert entity_map["Aragorn"] == "aragorn"
    assert entity_map["aragorn"] == "aragorn"
    assert entity_map["Mr. Strider"] == "aragorn"
    assert entity_map["Strider"] == "aragorn"
    assert entity_map["strider"] == "aragorn"


def test_sentence_graph_review_skips_unanchored_unresolved_propositions():
    noisy = ExtractedProposition(
        id="prop-noisy",
        passage_id="p1",
        passage_text="Through.",
        book="Test Book",
        sentence_num=1,
        kind=PropositionKind.PREPOSITIONAL,
        predicate_lemma="through",
        predicate_text="through",
        arguments=[
            PropositionArgument(
                role=ArgumentRole.LOCATION,
                surface="through",
                reference_class=ReferenceClass.UNKNOWN,
            )
        ],
    )
    anchored = ExtractedProposition(
        id="prop-anchored",
        passage_id="p1",
        passage_text="Bilbo went through the gate.",
        book="Test Book",
        sentence_num=1,
        kind=PropositionKind.MOVEMENT,
        predicate_lemma="go_through",
        predicate_text="went through",
        arguments=[
            PropositionArgument(
                role=ArgumentRole.AGENT,
                surface="Bilbo",
                entity_id="bilbo",
                canonical_name="Bilbo Baggins",
                entity_type="character",
            ),
            PropositionArgument(
                role=ArgumentRole.LOCATION,
                surface="the gate",
                phrase_id="np-gate",
                phrase_head="gate",
            ),
        ],
    )

    assert not is_reviewable_proposition(noisy)
    assert is_reviewable_proposition(anchored)
    assert not is_reviewable_unresolved_argument(noisy.arguments[0])
    assert is_reviewable_unresolved_argument(
        PropositionArgument(
            role=ArgumentRole.DESTINATION,
            surface="Mordor",
            expected_type="place",
            reference_class=ReferenceClass.CANON_CANDIDATE,
        )
    )


def test_sentence_graph_review_adds_seed_taxonomy_when_entities_appear():
    passage = _passage(
        "p1",
        "Bilbo announced a party at Bag End in Hobbiton.",
    )
    builder = ReviewGraphBuilder(
        book_title="Test Book",
        taxonomy=SeedTaxonomy.from_seed_dir(Path("data/seeds")),
    )

    builder.add_passage(
        index=1,
        passage=passage,
        entities=ExtractionResult(
            passage=passage,
            entities=[
                _entity("Bilbo", "bilbo_baggins", "Bilbo Baggins", "character"),
                _entity("Bag End", "bag_end", "Bag End", "place", label="PLACE", start=29),
                _entity("Hobbiton", "hobbiton", "Hobbiton", "place", label="PLACE", start=40),
            ],
            raw_extractions=[],
        ),
        relationships=RelationshipExtractionResult(
            passage_id=passage.id,
            passage_text=passage.text,
            relationships=[],
            entities_involved=[],
        ),
        propositions=[],
        quotes=[],
    )

    assert builder.nodes["character_type:hobbit"]["label"] == "Hobbit"
    assert builder.nodes["place_type:dwelling"]["label"] == "Dwelling"
    assert builder.nodes["place_type:village"]["label"] == "Village"
    assert builder.nodes["entity:the_shire"]["label"] == "The Shire"
    assert builder.edges[
        "entity:bilbo_baggins->IS_A->character_type:hobbit"
    ]["first_seen_sentence"] == 1
    assert builder.edges["entity:bag_end->LOCATED_IN->entity:hobbiton"]["first_seen_sentence"] == 1
    assert builder.edges["entity:hobbiton->LOCATED_IN->entity:the_shire"]["first_seen_sentence"] == 1


class _RecordingWriter(GraphWriter):
    def __init__(self) -> None:
        self.entities_batch = []
        self.relationship_batch = []
        self.passages = []
        self.links = []
        self.unresolved_refs = []
        self.propositions = []
        self.proposition_arguments = []
        self.proposition_unresolved_links = []
        self.noun_phrases = []
        self.noun_phrase_arguments = []
        self.noun_phrase_relations = []
        self.quotes = []
        self.quote_links = []

    def initialize(self) -> None:
        return None

    def write_entities_batch(self, entities, book: str) -> int:
        self.entities_batch = list(entities)
        return len(self.entities_batch)

    def write_relationships_batch(self, relationships) -> int:
        self.relationship_batch = list(relationships)
        return len(self.relationship_batch)

    def write_broken_references_batch(self, refs) -> int:
        self.unresolved_refs = list(refs)
        return len(self.unresolved_refs)

    def write_passage(self, **kwargs) -> None:
        self.passages.append(kwargs["passage_id"])

    def link_entity_to_passage(self, entity_id: str, passage_id: str) -> None:
        self.links.append((entity_id, passage_id))

    def _write_proposition_nodes(self, proposition_batch) -> None:
        self.propositions = list(proposition_batch)

    def _write_proposition_argument_links(self, argument_batch) -> None:
        self.proposition_arguments = list(argument_batch)

    def _write_proposition_unresolved_links(self, unresolved_links) -> None:
        self.proposition_unresolved_links = list(unresolved_links)

    def _write_noun_phrase_nodes(self, noun_phrase_batch) -> None:
        self.noun_phrases = list(noun_phrase_batch)

    def _write_noun_phrase_argument_links(self, noun_phrase_argument_batch) -> None:
        self.noun_phrase_arguments = list(noun_phrase_argument_batch)

    def _write_noun_phrase_relations(self, noun_phrase_relation_batch) -> None:
        self.noun_phrase_relations = list(noun_phrase_relation_batch)

    def _write_quote_nodes(self, quote_batch) -> None:
        self.quotes = list(quote_batch)

    def _write_quote_proposition_links(self, proposition_batch) -> None:
        self.quote_links = [
            {"quote_id": item.get("quote_id"), "proposition_id": item.get("id")}
            for item in proposition_batch
            if item.get("quote_id")
        ]


class _RecordingVoiceWriter(GraphWriter):
    def __init__(self) -> None:
        self.voice_calls = []

    def write_character_voice(self, character_id: str, profile) -> None:
        self.voice_calls.append((character_id, profile))


class _QueryRecordingSession:
    def __init__(self, runs) -> None:
        self.runs = runs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def run(self, query, **params):
        self.runs.append((query, params))


class _QueryRecordingDriver:
    def __init__(self) -> None:
        self.runs = []

    def session(self):
        return _QueryRecordingSession(self.runs)


def test_write_passage_creates_document_hierarchy_nodes():
    driver = _QueryRecordingDriver()
    writer = GraphWriter(driver=driver)

    writer.write_passage(
        passage_id="p_test_book_c3_p2_s4",
        text="Bilbo announced a party.",
        book="Test Book",
        chapter_num=3,
        paragraph_num=2,
        sentence_num=4,
        chapter_title="Chapter 3 A Long Test",
    )

    assert len(driver.runs) == 1
    query, params = driver.runs[0]
    assert "MERGE (b:Book {id: $book_id})" in query
    assert "MERGE (c:Chapter {id: $chapter_id})" in query
    assert "MERGE (pg:Paragraph {id: $paragraph_id})" in query
    assert "MERGE (s:Sentence:Passage {id: $id})" in query
    assert "MERGE (b)-[:HAS_CHAPTER]->(c)" in query
    assert "MERGE (c)-[:HAS_PARAGRAPH]->(pg)" in query
    assert "MERGE (pg)-[:HAS_SENTENCE]->(s)" in query
    assert params["book_id"] == "test_book"
    assert params["chapter_id"] == "test_book_c3"
    assert params["paragraph_id"] == "test_book_c3_p2"


def test_write_extraction_results_keeps_passages_mentions_and_partial_relationships():
    passage_a = _passage("p1", "Gandalf met Bilbo.")
    passage_b = _passage("p2", "Bilbo greeted the dragon.")

    gandalf = _entity("Gandalf", "gandalf", "Gandalf", "character")
    bilbo = _entity("Bilbo", "bilbo", "Bilbo", "character")
    dragon = _entity("the dragon", None, None, "character")

    entity_results = [
        ExtractionResult(passage=passage_a, entities=[gandalf, bilbo], raw_extractions=[]),
        ExtractionResult(passage=passage_b, entities=[bilbo, dragon], raw_extractions=[]),
    ]
    relationship_results = [
        RelationshipExtractionResult(
            passage_id="p1",
            passage_text=passage_a.text,
            entities_involved=[gandalf, bilbo],
            relationships=[
                ExtractedRelationship(
                    subject_text="Gandalf",
                    subject_id="gandalf",
                    subject_type="character",
                    predicate=RelationshipType.MET,
                    predicate_raw="met",
                    object_text="Bilbo",
                    object_id="bilbo",
                    object_type="character",
                    passage_id="p1",
                    passage_text=passage_a.text,
                    confidence=0.9,
                    extraction_method="dependency",
                )
            ],
        ),
        RelationshipExtractionResult(
            passage_id="p2",
            passage_text=passage_b.text,
            entities_involved=[bilbo, dragon],
            relationships=[
                ExtractedRelationship(
                    subject_text="Bilbo",
                    subject_id="bilbo",
                    subject_type="character",
                    predicate=RelationshipType.SPOKE_TO,
                    predicate_raw="greeted",
                    object_text="the dragon",
                    object_id=None,
                    object_type="character",
                    passage_id="p2",
                    passage_text=passage_b.text,
                    confidence=0.8,
                    extraction_method="dependency",
                )
            ],
        ),
    ]

    writer = _RecordingWriter()
    stats = writer.write_extraction_results(entity_results, relationship_results, book="Test Book")

    assert stats["entities_written"] == 2
    assert stats["entity_mentions_written"] == 3
    assert stats["relationships_written"] == 1
    assert stats["passages_written"] == 2
    assert stats["mention_links_written"] == 3
    assert stats["unresolved_references_written"] == 1
    assert stats["unresolved_reference_classes"] == {"generic_np": 1}
    assert writer.passages == ["p1", "p2"]
    assert writer.links == [("gandalf", "p1"), ("bilbo", "p1"), ("bilbo", "p2")]
    assert writer.unresolved_refs[0].mention_text == "the dragon"
    assert writer.unresolved_refs[0].reference_class == ReferenceClass.GENERIC_NP


def test_write_extraction_results_classifies_relationship_pronouns_as_pronouns():
    passage = _passage("p1", "He said to Bilbo.")
    bilbo = _entity("Bilbo", "bilbo", "Bilbo", "character", start=11)

    entity_results = [
        ExtractionResult(passage=passage, entities=[bilbo], raw_extractions=[]),
    ]
    relationship_results = [
        RelationshipExtractionResult(
            passage_id="p1",
            passage_text=passage.text,
            entities_involved=[bilbo],
            relationships=[
                ExtractedRelationship(
                    subject_text="He",
                    subject_id=None,
                    subject_type=None,
                    predicate=RelationshipType.SPOKE_TO,
                    predicate_raw="said",
                    object_text="Bilbo",
                    object_id="bilbo",
                    object_type="character",
                    passage_id="p1",
                    passage_text=passage.text,
                    confidence=0.8,
                    extraction_method="dependency",
                )
            ],
        )
    ]

    writer = _RecordingWriter()
    stats = writer.write_extraction_results(entity_results, relationship_results, book="Test Book")

    assert stats["unresolved_reference_classes"] == {"pronoun": 1}
    assert writer.unresolved_refs[0].mention_text == "He"
    assert writer.unresolved_refs[0].reference_class == ReferenceClass.PRONOUN


def test_write_extraction_results_drops_invalid_projected_relationship_pairs():
    passage = _passage("p1", "Frodo traveled to Mordor with Sam.")

    frodo = _entity("Frodo", "frodo", "Frodo Baggins", "character")
    sam = _entity("Sam", "sam", "Samwise Gamgee", "character", start=30)
    mordor = _entity("Mordor", "mordor", "Mordor", "place", label="PLACE", start=18)

    entity_results = [
        ExtractionResult(passage=passage, entities=[frodo, mordor, sam], raw_extractions=[]),
    ]
    relationship_results = [
        RelationshipExtractionResult(
            passage_id="p1",
            passage_text=passage.text,
            entities_involved=[frodo, mordor, sam],
            relationships=[
                ExtractedRelationship(
                    subject_text="Frodo",
                    subject_id="frodo",
                    subject_type="character",
                    predicate=RelationshipType.TRAVELED_TO,
                    predicate_raw="traveled",
                    object_text="Mordor",
                    object_id="mordor",
                    object_type="place",
                    passage_id="p1",
                    passage_text=passage.text,
                    confidence=0.9,
                    extraction_method="dependency",
                ),
                ExtractedRelationship(
                    subject_text="Frodo",
                    subject_id="frodo",
                    subject_type="character",
                    predicate=RelationshipType.TRAVELED_TO,
                    predicate_raw="traveled",
                    object_text="Sam",
                    object_id="sam",
                    object_type="character",
                    passage_id="p1",
                    passage_text=passage.text,
                    confidence=0.9,
                    extraction_method="dependency",
                ),
                ExtractedRelationship(
                    subject_text="Frodo",
                    subject_id="frodo",
                    subject_type="character",
                    predicate=RelationshipType.POSSESSES,
                    predicate_raw="has",
                    object_text="Sam",
                    object_id="sam",
                    object_type="character",
                    passage_id="p1",
                    passage_text=passage.text,
                    confidence=0.9,
                    extraction_method="dependency",
                ),
            ],
        )
    ]

    writer = _RecordingWriter()
    stats = writer.write_extraction_results(entity_results, relationship_results, book="Test Book")

    assert stats["relationships_written"] == 1
    assert len(writer.relationship_batch) == 1
    assert writer.relationship_batch[0].predicate == RelationshipType.TRAVELED_TO
    assert writer.relationship_batch[0].object_id == "mordor"


def test_write_proposition_results_persists_dense_layer_links():
    passage = _passage("p1", "Frodo carried the Ring to Mordor with Sam.")
    proposition_results = [
        PropositionExtractionResult(
            passage=passage,
            propositions=[
                ExtractedProposition(
                    id="prop-main",
                    passage_id="p1",
                    passage_text=passage.text,
                    book="Test Book",
                    sentence_num=1,
                    clause_index=1,
                    kind=PropositionKind.POSSESSION,
                    predicate_lemma="carry",
                    predicate_text="carried",
                    clause_text="Frodo carried the Ring",
                    confidence=0.9,
                    arguments=[
                        PropositionArgument(
                            role=ArgumentRole.AGENT,
                            surface="Frodo",
                            entity_id="frodo",
                            canonical_name="Frodo Baggins",
                            entity_type="character",
                            confidence=1.0,
                        ),
                        PropositionArgument(
                            role=ArgumentRole.PATIENT,
                            surface="the Ring",
                            entity_id="one_ring",
                            canonical_name="The One Ring",
                            entity_type="object",
                            confidence=1.0,
                        ),
                    ],
                ),
                ExtractedProposition(
                    id="prop-dest",
                    passage_id="p1",
                    passage_text=passage.text,
                    book="Test Book",
                    sentence_num=1,
                    clause_index=2,
                    kind=PropositionKind.MOVEMENT,
                    predicate_lemma="carry_to",
                    predicate_text="carried to",
                    clause_text="to Mordor",
                    confidence=0.7,
                    arguments=[
                        PropositionArgument(
                            role=ArgumentRole.AGENT,
                            surface="Frodo",
                            entity_id="frodo",
                            canonical_name="Frodo Baggins",
                            entity_type="character",
                            confidence=1.0,
                        ),
                        PropositionArgument(
                            role=ArgumentRole.DESTINATION,
                            surface="Mordor",
                            entity_id=None,
                            expected_type="place",
                            reference_class=ReferenceClass.CANON_CANDIDATE,
                            confidence=0.0,
                        ),
                    ],
                ),
            ],
        )
    ]

    writer = _RecordingWriter()
    stats = writer.write_proposition_results(proposition_results, book="Test Book")

    assert stats["propositions_written"] == 2
    assert stats["argument_links_written"] == 3
    assert stats["unresolved_links_written"] == 1
    assert stats["unresolved_references_written"] == 1
    assert stats["unresolved_reference_classes"] == {"canon_candidate": 1}
    assert [item["id"] for item in writer.propositions] == ["prop-main", "prop-dest"]
    assert writer.proposition_arguments[0]["role"] == "agent"
    assert writer.proposition_unresolved_links[0]["role"] == "destination"
    assert writer.unresolved_refs[0].reference_class == ReferenceClass.CANON_CANDIDATE


def test_write_proposition_results_persists_noun_phrase_nodes_and_links():
    passage = _passage("p1", "Bilbo announced a magnificent party in Hobbiton.")
    phrase_id = "np-party"
    proposition_results = [
        PropositionExtractionResult(
            passage=passage,
            propositions=[
                ExtractedProposition(
                    id="prop-announce",
                    passage_id="p1",
                    passage_text=passage.text,
                    book="Test Book",
                    sentence_num=1,
                    clause_index=1,
                    kind=PropositionKind.ACTION,
                    predicate_lemma="announce",
                    predicate_text="announced",
                    clause_text=passage.text,
                    confidence=0.9,
                    arguments=[
                        PropositionArgument(
                            role=ArgumentRole.AGENT,
                            surface="Bilbo",
                            entity_id="bilbo",
                            canonical_name="Bilbo Baggins",
                            entity_type="character",
                            confidence=1.0,
                        ),
                        PropositionArgument(
                            role=ArgumentRole.PATIENT,
                            surface="a magnificent party",
                            expected_type="object",
                            reference_class=ReferenceClass.GENERIC_NP,
                            phrase_id=phrase_id,
                            phrase_head="party",
                            phrase_modifiers=["magnificent"],
                            mention_start=16,
                            mention_end=35,
                            confidence=0.0,
                        ),
                    ],
                    noun_phrase_relations=[
                        NounPhraseRelation(
                            source_phrase_id=phrase_id,
                            relation_type="LOCATED_IN",
                            target_surface="Hobbiton",
                            prep="in",
                            target_entity_id="hobbiton",
                            target_entity_type="place",
                            target_canonical_name="Hobbiton",
                            confidence=1.0,
                        )
                    ],
                )
            ],
        )
    ]

    writer = _RecordingWriter()
    stats = writer.write_proposition_results(proposition_results, book="Test Book")

    assert stats["propositions_written"] == 1
    assert stats["argument_links_written"] == 1
    assert stats["noun_phrase_nodes_written"] == 1
    assert stats["noun_phrase_argument_links_written"] == 1
    assert stats["noun_phrase_relation_links_written"] == 1
    assert stats["unresolved_references_written"] == 0
    assert writer.noun_phrases == [
        {
            "id": phrase_id,
            "surface": "a magnificent party",
            "head": "party",
            "modifiers": ["magnificent"],
            "book": "Test Book",
            "passage_id": "p1",
            "mention_start": 16,
            "mention_end": 35,
            "expected_type": "object",
            "reference_class": "generic_np",
            "confidence": 0.0,
        }
    ]
    assert writer.noun_phrase_arguments[0]["role"] == "patient"
    assert writer.noun_phrase_relations[0]["relation_type"] == "LOCATED_IN"
    assert writer.noun_phrase_relations[0]["target_entity_id"] == "hobbiton"


def test_write_proposition_results_persists_quote_nodes_and_links():
    passage = _passage("p1", '"I will go," said Aragorn.')
    proposition_results = [
        PropositionExtractionResult(
            passage=passage,
            quotes=[
                ExtractedQuote(
                    id="quote-1",
                    passage_id="p1",
                    text="I will go,",
                    quote_start=0,
                    quote_end=12,
                    speaker_name="Aragorn",
                    speaker_entity_id="aragorn",
                    speaker_canonical_name="Aragorn",
                    speech_verb="said",
                    attribution_confidence=0.9,
                )
            ],
            propositions=[
                ExtractedProposition(
                    id="prop-go",
                    passage_id="p1",
                    passage_text=passage.text,
                    book="Test Book",
                    sentence_num=1,
                    clause_index=1,
                    kind=PropositionKind.MOVEMENT,
                    predicate_lemma="go",
                    predicate_text="go",
                    quote_id="quote-1",
                    confidence=0.9,
                    arguments=[
                        PropositionArgument(
                            role=ArgumentRole.AGENT,
                            surface="I",
                            entity_id="aragorn",
                            canonical_name="Aragorn",
                            entity_type="character",
                            confidence=1.0,
                        )
                    ],
                )
            ],
        )
    ]

    writer = _RecordingWriter()
    stats = writer.write_proposition_results(proposition_results, book="Test Book")

    assert stats["quotes_written"] == 1
    assert writer.quotes[0]["id"] == "quote-1"
    assert writer.quotes[0]["speaker_entity_id"] == "aragorn"
    assert writer.propositions[0]["quote_id"] == "quote-1"
    assert writer.quote_links == [{"quote_id": "quote-1", "proposition_id": "prop-go"}]


def test_write_voice_analysis_results_skips_unmapped_profiles():
    writer = _RecordingVoiceWriter()
    voice_result = SimpleNamespace(
        profiles={
            "Aragorn": object(),
            "he": object(),
            "Wormtongue": object(),
        }
    )

    stats = writer.write_voice_analysis_results(
        voice_result=voice_result,
        book_id="test_book",
        entity_id_map={"Aragorn": "aragorn", "aragorn": "aragorn"},
    )

    assert stats["profiles_written"] == 1
    assert stats["profiles_skipped_unmapped"] == 2
    assert writer.voice_calls == [("aragorn", voice_result.profiles["Aragorn"])]


def test_write_voice_analysis_results_merges_aliases_by_entity_id():
    writer = _RecordingVoiceWriter()
    voice_result = SimpleNamespace(
        dialogue_by_speaker={
            "Aragorn": [DialogueLine(text="I will go.", speaker="Aragorn")],
            "Strider": [DialogueLine(text="We must move.", speaker="Strider")],
        },
        profiles={},
    )

    stats = writer.write_voice_analysis_results(
        voice_result=voice_result,
        book_id="test_book",
        entity_id_map={
            "Aragorn": "aragorn",
            "aragorn": "aragorn",
            "Strider": "aragorn",
            "strider": "aragorn",
        },
        min_lines_for_profile=1,
    )

    assert stats["profiles_written"] == 1
    assert stats["profiles_merged_aliases"] == 1
    assert len(writer.voice_calls) == 1
    char_id, profile = writer.voice_calls[0]
    assert char_id == "aragorn"
    assert profile.total_lines == 2
