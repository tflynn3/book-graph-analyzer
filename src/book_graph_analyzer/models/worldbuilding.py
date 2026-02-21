"""World-building model stubs for Tolkien deep-lore integration.

These models extend the existing entity/relationship system with three
new layered concepts:

1. Linguistic Lineage — etymology chains across Tolkien's invented languages
2. Genealogy Relations — deep family trees with generational metadata
3. Editorial Layer Metadata — source-text provenance tracking

All models integrate with the existing pipeline via the models package
and are consumed by GraphWriter, LoreChecker, and context assembly.

See: docs/tolkien-worldbuilding-rfc.md
Milestone Issues: #45–#51
"""

from __future__ import annotations

from enum import Enum
import re
from pydantic import BaseModel, Field


# =============================================================================
# Linguistic Lineage (Issue #46)
# =============================================================================


class TolkienLanguage(str, Enum):
    """Languages of Arda relevant to etymology tracking."""

    QUENYA = "Quenya"
    SINDARIN = "Sindarin"
    WESTRON = "Westron"
    KHUZDUL = "Khuzdul"
    BLACK_SPEECH = "Black Speech"
    ADUNAIC = "Adûnaic"
    ROHIRRIC = "Rohirric"
    COMMON_SPEECH = "Common Speech"
    VALARIN = "Valarin"
    ENTISH = "Entish"
    PRIMITIVE_ELVISH = "Primitive Elvish"
    TELERIN = "Telerin"
    NOLDORIN = "Noldorin"
    OTHER = "Other"


class DerivationType(str, Enum):
    """How one language form derives from another."""

    TRANSLATION = "translation"  # Direct meaning-preserving translation
    ADAPTATION = "adaptation"  # Phonological adaptation between languages
    COGNATE = "cognate"  # Shared ancestral root
    COMPOUND = "compound"  # Formed by combining elements
    EPITHET = "epithet"  # Descriptive name/title
    LOANWORD = "loanword"  # Borrowed from another language


class LanguageForm(BaseModel):
    """A single attestation of a name/word in a specific language.

    Example: LanguageForm(form="Imladris", language=TolkienLanguage.SINDARIN,
                          entity_id="place_rivendell", gloss="Deep dale of the cleft")

    TODO(#46): Implement extraction logic in worldbible.extractor
    TODO(#46): Add Neo4j DERIVED_FROM chain writer in graph.writer
    """

    id: str
    form: str  # The actual word/name: "Imladris"
    language: TolkienLanguage
    entity_id: str | None = None  # FK to Character/Place/Object id
    gloss: str | None = None  # English meaning/translation
    phonetic: str | None = None  # IPA or rough pronunciation guide
    source_passage_id: str | None = None  # Where this form is attested


class LinguisticLineage(BaseModel):
    """An etymology chain linking forms of a name across languages.

    Example: Imladris (Sindarin) → Rivendell (Common Speech) → Karningul (Westron)

    TODO(#46): Integrate with extract.resolver for language-aware alias matching
    TODO(#46): Build query helpers for "all names for entity X across languages"
    """

    entity_id: str  # The entity this lineage describes
    forms: list[LanguageForm] = Field(default_factory=list)
    derivations: list[LanguageDerivation] = Field(default_factory=list)

    def primary_form(self, language: TolkienLanguage) -> LanguageForm | None:
        """Get the primary form in a given language."""
        for form in self.forms:
            if form.language == language:
                return form
        return None


class LanguageDerivation(BaseModel):
    """A directional derivation link between two language forms.

    TODO(#46): Map to Neo4j (:LanguageForm)-[:DERIVED_FROM]->(:LanguageForm)
    """

    source_form_id: str
    target_form_id: str
    derivation_type: DerivationType
    notes: str | None = None


# =============================================================================
# Deep Genealogy (Issue #47)
# =============================================================================


class GenealogyRelationType(str, Enum):
    """Fine-grained family relationship types beyond basic PARENT_OF/CHILD_OF."""

    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    SIBLING_OF = "SIBLING_OF"
    SPOUSE_OF = "SPOUSE_OF"
    GRANDPARENT_OF = "GRANDPARENT_OF"
    GRANDCHILD_OF = "GRANDCHILD_OF"
    ANCESTOR_OF = "ANCESTOR_OF"
    DESCENDANT_OF = "DESCENDANT_OF"
    FOSTER_PARENT_OF = "FOSTER_PARENT_OF"
    FOSTER_CHILD_OF = "FOSTER_CHILD_OF"
    HALF_SIBLING_OF = "HALF_SIBLING_OF"


class GenealogyRelation(BaseModel):
    """A genealogical relationship with depth and house metadata.

    Extends the basic RelationshipTriple with Tolkien-specific genealogy
    attributes: generational depth, house membership, and inherited traits.

    Example:
        GenealogyRelation(
            source_id="char_aragorn", target_id="char_elendil",
            relation_type=GenealogyRelationType.DESCENDANT_OF,
            generation_depth=39,
            house="House of Elendil",
            inheritance_traits=["longevity", "foresight", "right to rule Gondor"],
        )

    TODO(#47): Implement genealogy extraction patterns in extract.relationships
    TODO(#47): Build generational depth calculator from graph traversal
    TODO(#47): Family-tree context assembly in generate.context
    """

    source_id: str  # Character who IS the relation
    source_name: str | None = None
    target_id: str  # Character they are related TO
    target_name: str | None = None
    relation_type: GenealogyRelationType
    generation_depth: int | None = None  # How many generations apart
    house: str | None = None  # "House of Finwë", "House of Bëor", etc.
    inheritance_traits: list[str] = Field(default_factory=list)
    era: str | None = None  # When this relationship was established
    passage_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0


# =============================================================================
# Editorial Layer Metadata (Issue #48)
# =============================================================================


class EditorialStatus(str, Enum):
    """Publication/editorial status of a source text."""

    PUBLISHED = "published"  # Final published form (Silmarillion 1977, LOTR)
    DRAFT = "draft"  # Author's draft (HoME volumes)
    NOTES = "notes"  # Scattered notes and letters
    LETTER = "letter"  # From Tolkien's letters
    POSTHUMOUS_EDIT = "posthumous_edit"  # Christopher Tolkien's editorial work
    UNFINISHED = "unfinished"  # Unfinished Tales — explicitly incomplete


class AuthorPeriod(str, Enum):
    """Tolkien's writing periods, affecting interpretation of contradictions."""

    EARLY = "early"  # ~1917-1930: Book of Lost Tales era
    MIDDLE = "middle"  # ~1930-1950: Hobbit/LOTR writing period
    LATE = "late"  # ~1950-1973: Post-LOTR revisions and essays
    EDITORIAL = "editorial"  # Christopher Tolkien's editorial decisions


class EditorialLayer(BaseModel):
    """Metadata tracking which source text and editorial period a fact comes from.

    This enables provenance-aware lore checking: when two sources conflict,
    the editorial layer determines which version has priority.

    Example:
        EditorialLayer(
            source_id="src_silmarillion_1977",
            source_title="The Silmarillion",
            editorial_status=EditorialStatus.PUBLISHED,
            author_period=AuthorPeriod.EDITORIAL,
            publication_year=1977,
            authority_weight=0.85,
            notes="Christopher Tolkien's edited compilation",
        )

    TODO(#48): Integrate with ingest.loader for automatic source tagging
    TODO(#48): Add source_authority weighting to lore.conflicts resolution
    TODO(#48): Build (:Source) node writer in graph.writer
    """

    source_id: str
    source_title: str
    editorial_status: EditorialStatus
    author_period: AuthorPeriod
    publication_year: int | None = None
    editor: str | None = None  # e.g., "Christopher Tolkien"
    volume: str | None = None  # e.g., "HoME Vol. X"
    authority_weight: float = 1.0  # 0.0–1.0, used in conflict resolution
    notes: str | None = None

    @property
    def is_primary_canon(self) -> bool:
        """Whether this source is considered primary canon (LOTR, Hobbit, Silmarillion)."""
        return (
            self.editorial_status == EditorialStatus.PUBLISHED
            and self.author_period in (AuthorPeriod.MIDDLE, AuthorPeriod.LATE)
        )


# =============================================================================
# Canonical Source Registry (Issue #48)
# =============================================================================

# Pre-defined editorial layers for major Tolkien works.
# TODO(#48): Move to a YAML/JSON config file for user customization.

TOLKIEN_SOURCES: list[EditorialLayer] = [
    EditorialLayer(
        source_id="src_hobbit",
        source_title="The Hobbit",
        editorial_status=EditorialStatus.PUBLISHED,
        author_period=AuthorPeriod.MIDDLE,
        publication_year=1937,
        authority_weight=1.0,
    ),
    EditorialLayer(
        source_id="src_fellowship",
        source_title="The Fellowship of the Ring",
        editorial_status=EditorialStatus.PUBLISHED,
        author_period=AuthorPeriod.MIDDLE,
        publication_year=1954,
        authority_weight=1.0,
    ),
    EditorialLayer(
        source_id="src_two_towers",
        source_title="The Two Towers",
        editorial_status=EditorialStatus.PUBLISHED,
        author_period=AuthorPeriod.MIDDLE,
        publication_year=1954,
        authority_weight=1.0,
    ),
    EditorialLayer(
        source_id="src_return_king",
        source_title="The Return of the King",
        editorial_status=EditorialStatus.PUBLISHED,
        author_period=AuthorPeriod.MIDDLE,
        publication_year=1955,
        authority_weight=1.0,
    ),
    EditorialLayer(
        source_id="src_silmarillion_1977",
        source_title="The Silmarillion",
        editorial_status=EditorialStatus.PUBLISHED,
        author_period=AuthorPeriod.EDITORIAL,
        publication_year=1977,
        editor="Christopher Tolkien",
        authority_weight=0.85,
        notes="Posthumous compilation; some editorial decisions by Christopher Tolkien",
    ),
    EditorialLayer(
        source_id="src_unfinished_tales",
        source_title="Unfinished Tales",
        editorial_status=EditorialStatus.UNFINISHED,
        author_period=AuthorPeriod.LATE,
        publication_year=1980,
        editor="Christopher Tolkien",
        authority_weight=0.7,
    ),
    EditorialLayer(
        source_id="src_letters",
        source_title="The Letters of J.R.R. Tolkien",
        editorial_status=EditorialStatus.LETTER,
        author_period=AuthorPeriod.LATE,
        publication_year=1981,
        authority_weight=0.9,
        notes="Author's own statements about intent and lore",
    ),
]


def find_editorial_layer(source_name: str) -> EditorialLayer | None:
    """Find a known editorial layer by title/id with fuzzy normalization."""
    key = source_name.strip().lower().replace("_", " ").replace("-", " ")
    if not key:
        return None

    compact = " ".join(key.split())
    for layer in TOLKIEN_SOURCES:
        candidates = {
            layer.source_id.lower(),
            layer.source_title.lower(),
            layer.source_title.lower().replace("the ", "", 1),
        }
        if compact in candidates:
            return layer

        slug = layer.source_title.lower().replace("-", " ")
        if compact == " ".join(slug.split()):
            return layer

    # Loose contains/token fallback for common CLI/file-name forms.
    key_tokens = set(re.findall(r"[a-z0-9]+", compact))
    for layer in TOLKIEN_SOURCES:
        title = layer.source_title.lower()
        if compact in title or title in compact:
            return layer

        title_tokens = set(re.findall(r"[a-z0-9]+", title)) - {"the", "of"}
        if title_tokens and title_tokens.issubset(key_tokens):
            return layer

    return None


def infer_editorial_layer(path_or_title: str) -> EditorialLayer | None:
    """Infer source layer from a file path or title string."""
    normalized = path_or_title.replace("\\", "/").split("/")[-1]
    stem = normalized.rsplit(".", 1)[0]
    return find_editorial_layer(stem) or find_editorial_layer(path_or_title)
