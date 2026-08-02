"""Entity extraction pipeline for Book Graph Analyzer."""

from .extractor import EntityExtractor
from .ner import NERPipeline
from .resolver import EntityResolver
from .relationships import RelationshipExtractor
from .propositions import PropositionExtractor, PropositionExtractionResult
from .dynamic_resolver import DynamicEntityResolver
from .generic_extractor import GenericExtractor, BookAnalysis
from .book_pipeline import (
    BookGraphExtraction,
    build_entity_clusters,
    build_entity_id_map,
    extract_book_graph,
)

# v2 improvements (Issue #12)
from .normalizer import TextNormalizer, normalize_text, strip_artifacts, has_artifacts
from .coref import PronounResolver, detect_explicit_aliases, CoreferenceChain
from .disambiguation import DisambiguationDict
from .resolver_v2 import (
    EntityResolverV2,
    ResolvedEntityV2,
    ResolutionResultV2,
    ACCEPT_THRESHOLD,
    REVIEW_THRESHOLD,
)

__all__ = [
    # v1
    "EntityExtractor",
    "NERPipeline",
    "EntityResolver",
    "RelationshipExtractor",
    "PropositionExtractor",
    "PropositionExtractionResult",
    "DynamicEntityResolver",
    "GenericExtractor",
    "BookAnalysis",
    "BookGraphExtraction",
    "extract_book_graph",
    "build_entity_id_map",
    "build_entity_clusters",
    # v2
    "TextNormalizer",
    "normalize_text",
    "strip_artifacts",
    "has_artifacts",
    "PronounResolver",
    "detect_explicit_aliases",
    "CoreferenceChain",
    "DisambiguationDict",
    "EntityResolverV2",
    "ResolvedEntityV2",
    "ResolutionResultV2",
    "ACCEPT_THRESHOLD",
    "REVIEW_THRESHOLD",
]
