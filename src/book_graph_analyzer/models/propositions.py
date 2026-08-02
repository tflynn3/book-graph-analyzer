"""Dense proposition-layer models for sentence-level semantic extraction."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PropositionKind(str, Enum):
    """Semantic proposition categories."""

    ACTION = "action"
    STATE = "state"
    ATTRIBUTE = "attribute"
    POSSESSION = "possession"
    MOVEMENT = "movement"
    SPEECH = "speech"
    APPOSITIVE = "appositive"
    PREPOSITIONAL = "prepositional"


class ReferenceClass(str, Enum):
    """High-level unresolved reference categories for review and routing."""

    PRONOUN = "pronoun"
    DISCOURSE_DEICTIC = "discourse_deictic"
    BRIDGING = "bridging"
    GENERIC_NP = "generic_np"
    BODY_PART = "body_part"
    CANON_CANDIDATE = "canon_candidate"
    UNKNOWN = "unknown"


class ArgumentRole(str, Enum):
    """Supported proposition argument roles."""

    AGENT = "agent"
    SUBJECT = "subject"
    PATIENT = "patient"
    RECIPIENT = "recipient"
    LOCATION = "location"
    SOURCE = "source"
    DESTINATION = "destination"
    INSTRUMENT = "instrument"
    COMPANION = "companion"
    TOPIC = "topic"
    ATTRIBUTE = "attribute"
    POSSESSOR = "possessor"
    POSSESSED = "possessed"
    REFERENT = "referent"
    DESCRIPTOR = "descriptor"


class PropositionArgument(BaseModel):
    """One proposition argument, optionally grounded to a canonical entity."""

    role: ArgumentRole
    surface: str
    entity_id: str | None = None
    canonical_name: str | None = None
    entity_type: str | None = None
    expected_type: str | None = None
    reference_class: ReferenceClass | None = None
    mention_start: int | None = None
    mention_end: int | None = None
    confidence: float = 0.0
    is_pronoun: bool = False
    prep: str | None = None
    phrase_id: str | None = None
    phrase_head: str | None = None
    phrase_modifiers: list[str] = Field(default_factory=list)


class NounPhraseRelation(BaseModel):
    """A relation originating from a first-class noun phrase argument."""

    source_phrase_id: str
    relation_type: str
    target_surface: str
    prep: str
    target_entity_id: str | None = None
    target_entity_type: str | None = None
    target_canonical_name: str | None = None
    target_phrase_id: str | None = None
    target_phrase_head: str | None = None
    target_phrase_modifiers: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ExtractedQuote(BaseModel):
    """A first-class quoted speech span with resolved participants when available."""

    id: str
    passage_id: str
    text: str
    quote_start: int
    quote_end: int
    speaker_name: str | None = None
    speaker_entity_id: str | None = None
    speaker_canonical_name: str | None = None
    addressee_entity_id: str | None = None
    addressee_canonical_name: str | None = None
    speech_verb: str | None = None
    attribution_confidence: float = 0.0
    is_question: bool = False
    is_exclamation: bool = False
    is_imperative: bool = False
    is_verse: bool = False
    audience_type: str | None = None
    context_type: str | None = None
    audience_confidence: float = 0.0


class ExtractedProposition(BaseModel):
    """A sentence-level proposition extracted from a passage."""

    id: str
    passage_id: str
    passage_text: str
    book: str | None = None
    sentence_num: int | None = None
    clause_index: int = 0
    kind: PropositionKind
    predicate_lemma: str
    predicate_text: str
    predicate_span_start: int | None = None
    predicate_span_end: int | None = None
    clause_text: str | None = None
    quote_id: str | None = None
    confidence: float = 0.0
    extraction_method: str = "dependency"
    modality: str = "asserted"
    polarity: str = "positive"
    arguments: list[PropositionArgument] = Field(default_factory=list)
    noun_phrase_relations: list[NounPhraseRelation] = Field(default_factory=list)
