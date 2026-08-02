"""Data models for entities and relationships."""

from book_graph_analyzer.models.entities import Character, Place, Object, Event, Concept
from book_graph_analyzer.models.passage import Passage
from book_graph_analyzer.models.worldbuilding import (
    EditorialLayer,
    GenealogyRelation,
    LanguageForm,
    LinguisticLineage,
    SourceStratum,
)
from book_graph_analyzer.models.lore_depth import (
    BrokenReference,
    LoreArtifact,
    LoreArtifactType,
    LoreDepthExtractionResult,
    ReferenceCandidate,
)
from book_graph_analyzer.models.propositions import (
    ArgumentRole,
    ExtractedProposition,
    NounPhraseRelation,
    PropositionArgument,
    PropositionKind,
    ReferenceClass,
)

__all__ = [
    "Character",
    "Place",
    "Object",
    "Event",
    "Concept",
    "Passage",
    # World-building layers (Issue #45)
    "EditorialLayer",
    "GenealogyRelation",
    "LanguageForm",
    "LinguisticLineage",
    "SourceStratum",
    "LoreArtifactType",
    "LoreArtifact",
    "BrokenReference",
    "LoreDepthExtractionResult",
    "ReferenceCandidate",
    "ArgumentRole",
    "ExtractedProposition",
    "NounPhraseRelation",
    "PropositionArgument",
    "PropositionKind",
    "ReferenceClass",
]
