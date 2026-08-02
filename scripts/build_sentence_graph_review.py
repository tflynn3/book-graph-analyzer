"""Build a static sentence-by-sentence graph review dataset.

The output JSON is intended for the local graph scrubber UI in
``tools/graph-scrubber``. It mirrors the document hierarchy and proposition
graph shape used by the Neo4j writer without requiring a running database.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from book_graph_analyzer.extract.book_pipeline import extract_book_graph
from book_graph_analyzer.extract.extractor import ExtractionResult
from book_graph_analyzer.extract.relationships import RelationshipExtractionResult
from book_graph_analyzer.ingest.splitter import Passage, split_into_passages
from book_graph_analyzer.models.propositions import ExtractedProposition, ExtractedQuote, PropositionArgument


DEFAULT_SOURCE = REPO_ROOT / "data/texts/lotr-corpus/fellowship.txt"
DEFAULT_OUTPUT = REPO_ROOT / "tools/graph-scrubber/graph_review.json"
DEFAULT_SEEDS = REPO_ROOT / "data/seeds"
DEFAULT_BOOK_TITLE = "The Fellowship of the Ring"


def slug_id(value: str) -> str:
    """Return the same style of stable slug used by graph IDs."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def truncate_label(value: str, limit: int) -> str:
    """Compact long labels for graph display data."""
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit - 1]}..."


def build_review_dataset(
    source: Path,
    *,
    book_title: str = DEFAULT_BOOK_TITLE,
    limit: int = 10,
    seed_dir: Path = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Analyze the first ``limit`` passages and build a review graph."""
    text = source.read_text(errors="replace")
    passages = split_into_passages(text, book_title)[:limit]
    extraction = extract_book_graph(passages, use_llm=False, seed_dir=seed_dir)

    builder = ReviewGraphBuilder(
        book_title=book_title,
        taxonomy=SeedTaxonomy.from_seed_dir(seed_dir),
    )

    entity_by_passage = {
        result.passage.id: result for result in extraction.entity_results
    }
    relationships_by_passage = {
        result.passage_id: result for result in extraction.relationship_results
    }
    propositions_by_passage = {
        result.passage.id: result for result in extraction.proposition_results
    }

    for index, passage in enumerate(passages, start=1):
        builder.add_passage(
            index=index,
            passage=passage,
            entities=entity_by_passage[passage.id],
            relationships=relationships_by_passage[passage.id],
            propositions=propositions_by_passage[passage.id].propositions,
            quotes=propositions_by_passage[passage.id].quotes or [],
        )

    return {
        "metadata": {
            "book_title": book_title,
            "source": str(source),
            "sentence_count": len(passages),
            "node_count": len(builder.nodes),
            "edge_count": len(builder.edges),
            "unique_entity_count": extraction.unique_entity_count,
            "resolved_mention_count": extraction.resolved_mention_count,
            "unresolved_entity_count": extraction.unresolved_entity_count,
            "total_relationships": extraction.total_relationships,
            "total_propositions": extraction.total_propositions,
        },
        "sentences": builder.sentences,
        "nodes": sorted(builder.nodes.values(), key=lambda n: (n["first_seen_sentence"], n["type"], n["label"])),
        "edges": sorted(builder.edges.values(), key=lambda e: (e["first_seen_sentence"], e["type"], e["source"], e["target"])),
        "sentence_summaries": builder.sentence_summaries,
    }


@dataclass(frozen=True)
class TaxonomyNode:
    """A stable semantic category node derived from seed data."""

    id: str
    type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


class SeedTaxonomy:
    """Look up story taxonomy structure from seed files."""

    def __init__(
        self,
        *,
        characters: dict[str, dict[str, Any]] | None = None,
        places: dict[str, dict[str, Any]] | None = None,
        objects: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.characters = characters or {}
        self.places = places or {}
        self.objects = objects or {}

    @classmethod
    def from_seed_dir(cls, seed_dir: Path) -> "SeedTaxonomy":
        return cls(
            characters=_load_seed_index(seed_dir / "characters.json"),
            places=_load_seed_index(seed_dir / "places.json"),
            objects=_load_seed_index(seed_dir / "objects.json"),
        )

    def entity_label(self, entity_id: str) -> str:
        seed = self.characters.get(entity_id) or self.places.get(entity_id) or self.objects.get(entity_id)
        return str(seed.get("canonical_name") or entity_id.replace("_", " ").title()) if seed else entity_id

    def entity_properties(self, entity_id: str) -> dict[str, Any]:
        seed = self.characters.get(entity_id) or self.places.get(entity_id) or self.objects.get(entity_id) or {}
        return {
            "canonical_id": entity_id,
            "race": seed.get("race"),
            "place_type": seed.get("type") if entity_id in self.places else None,
            "object_type": seed.get("type") if entity_id in self.objects else None,
            "description": seed.get("description"),
        }

    def taxonomies_for(self, entity_id: str, entity_type: str | None) -> list[TaxonomyNode]:
        normalized = (entity_type or "").lower()
        if normalized == "character":
            race = self.characters.get(entity_id, {}).get("race")
            if race:
                return [
                    TaxonomyNode(
                        id=f"character_type:{slug_id(race)}",
                        type="CharacterType",
                        label=str(race),
                        properties={"category": "race"},
                    )
                ]
        if normalized == "place":
            place_type = self.places.get(entity_id, {}).get("type")
            if place_type:
                return [
                    TaxonomyNode(
                        id=f"place_type:{slug_id(str(place_type))}",
                        type="PlaceType",
                        label=str(place_type).replace("_", " ").title(),
                        properties={"category": "place_type", "value": place_type},
                    )
                ]
        if normalized == "object":
            object_type = self.objects.get(entity_id, {}).get("type")
            if object_type:
                return [
                    TaxonomyNode(
                        id=f"object_type:{slug_id(str(object_type))}",
                        type="ObjectType",
                        label=str(object_type).replace("_", " ").title(),
                        properties={"category": "object_type", "value": object_type},
                    )
                ]
        return []

    def place_parent_id(self, entity_id: str) -> str | None:
        parent = self.places.get(entity_id, {}).get("parent_region")
        return str(parent) if parent else None

    def object_creator_id(self, entity_id: str) -> str | None:
        creator = self.objects.get(entity_id, {}).get("creator_id")
        return str(creator) if creator else None


class ReviewGraphBuilder:
    """Accumulate nodes and edges with first-sentence provenance."""

    def __init__(self, *, book_title: str, taxonomy: SeedTaxonomy | None = None) -> None:
        self.book_title = book_title
        self.book_slug = slug_id(book_title)
        self.taxonomy = taxonomy or SeedTaxonomy()
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.sentences: list[dict[str, Any]] = []
        self.sentence_summaries: list[dict[str, Any]] = []

    def add_passage(
        self,
        *,
        index: int,
        passage: Passage,
        entities: ExtractionResult,
        relationships: RelationshipExtractionResult,
        propositions: Iterable[ExtractedProposition],
        quotes: Iterable[ExtractedQuote],
    ) -> None:
        sentence_id = f"sentence:{passage.id}"
        book_id = f"book:{self.book_slug}"
        chapter_id = f"chapter:{self.book_slug}:c{passage.chapter_num}"
        paragraph_id = f"paragraph:{self.book_slug}:c{passage.chapter_num}:p{passage.paragraph_num}"

        before_nodes = set(self.nodes)
        before_edges = set(self.edges)

        self.sentences.append(
            {
                "index": index,
                "id": passage.id,
                "text": passage.text,
                "book": passage.book,
                "chapter": passage.chapter,
                "chapter_num": passage.chapter_num,
                "paragraph_num": passage.paragraph_num,
                "sentence_num": passage.sentence_num,
            }
        )

        self.add_node(
            book_id,
            index,
            type="Book",
            label=passage.book,
            properties={"title": passage.book},
        )
        self.add_node(
            chapter_id,
            index,
            type="Chapter",
            label=f"Chapter {passage.chapter_num}",
            properties={"number": passage.chapter_num, "title": passage.chapter},
        )
        self.add_node(
            paragraph_id,
            index,
            type="Paragraph",
            label=f"P{passage.paragraph_num}",
            properties={"number": passage.paragraph_num},
        )
        self.add_node(
            sentence_id,
            index,
            type="Sentence",
            label=f"S{index}",
            properties={
                "passage_id": passage.id,
                "number": passage.sentence_num,
                "text": passage.text,
            },
        )
        self.add_edge(book_id, chapter_id, "HAS_CHAPTER", index)
        self.add_edge(chapter_id, paragraph_id, "HAS_PARAGRAPH", index)
        self.add_edge(paragraph_id, sentence_id, "HAS_SENTENCE", index)

        quote_by_id = {quote.id: quote for quote in quotes}
        for quote in quote_by_id.values():
            self.add_quote(index, sentence_id, quote)

        for entity in entities.entities:
            entity_id = self.entity_node_id(entity)
            label = entity.canonical_name or entity.extracted.text
            self.add_node(
                entity_id,
                index,
                type=_entity_type_label(entity.entity_type),
                label=label,
                properties={
                    "canonical_id": entity.canonical_id,
                    "surface": entity.extracted.text,
                    "entity_type": entity.entity_type,
                    "confidence": entity.confidence,
                    "is_new": entity.is_new,
                },
            )
            self.add_edge(
                entity_id,
                sentence_id,
                "MENTIONED_IN",
                index,
                properties={
                    "surface": entity.extracted.text,
                    "start_char": entity.extracted.start_char,
                    "end_char": entity.extracted.end_char,
                },
            )
            if entity.canonical_id:
                self.add_entity_taxonomy(index, entity_id, entity.canonical_id, entity.entity_type)

        for relationship in relationships.relationships:
            source_id = self.entity_ref_node_id(
                relationship.subject_id,
                relationship.subject_text,
                relationship.subject_type,
            )
            target_id = self.entity_ref_node_id(
                relationship.object_id,
                relationship.object_text,
                relationship.object_type,
            )
            self.add_node(
                source_id,
                index,
                type=_entity_type_label(relationship.subject_type or "unknown"),
                label=relationship.subject_text,
                properties={"entity_type": relationship.subject_type},
            )
            if relationship.subject_id:
                self.add_entity_taxonomy(
                    index,
                    source_id,
                    relationship.subject_id,
                    relationship.subject_type,
                )
            self.add_node(
                target_id,
                index,
                type=_entity_type_label(relationship.object_type or "unknown"),
                label=relationship.object_text,
                properties={"entity_type": relationship.object_type},
            )
            if relationship.object_id:
                self.add_entity_taxonomy(
                    index,
                    target_id,
                    relationship.object_id,
                    relationship.object_type,
                )
            self.add_edge(
                source_id,
                target_id,
                relationship.predicate.value,
                index,
                properties={
                    "predicate_raw": relationship.predicate_raw,
                    "confidence": relationship.confidence,
                    "extraction_method": relationship.extraction_method,
                    "passage_id": passage.id,
                },
            )

        for proposition in propositions:
            if not is_reviewable_proposition(proposition):
                continue
            proposition_id = f"proposition:{proposition.id}"
            self.add_node(
                proposition_id,
                index,
                type="Proposition",
                label=proposition.predicate_lemma,
                properties={
                    "kind": proposition.kind.value,
                    "predicate_text": proposition.predicate_text,
                    "clause_text": proposition.clause_text,
                    "quote_id": proposition.quote_id,
                    "confidence": proposition.confidence,
                    "modality": proposition.modality,
                    "polarity": proposition.polarity,
                },
            )
            self.add_edge(
                sentence_id,
                proposition_id,
                "HAS_PROPOSITION",
                index,
                properties={"kind": proposition.kind.value},
            )
            if proposition.quote_id and proposition.quote_id in quote_by_id:
                self.add_edge(
                    f"quote:{proposition.quote_id}",
                    proposition_id,
                    "EXPRESSES",
                    index,
                )
            for argument in proposition.arguments:
                self.add_argument(index, proposition_id, argument)
            for relation in proposition.noun_phrase_relations:
                source_id = f"phrase:{relation.source_phrase_id}"
                target_id = (
                    f"entity:{relation.target_entity_id}"
                    if relation.target_entity_id
                    else f"phrase:{relation.target_phrase_id}"
                    if relation.target_phrase_id
                    else f"unresolved:{slug_id(relation.target_surface)}"
                )
                if relation.target_phrase_id:
                    self.add_node(
                        target_id,
                        index,
                        type="NounPhrase",
                        label=relation.target_surface,
                        properties={
                            "surface": relation.target_surface,
                            "head": relation.target_phrase_head,
                            "modifiers": relation.target_phrase_modifiers,
                        },
                    )
                    self.add_phrase_modifiers(
                        index,
                        phrase_node_id=target_id,
                        phrase_id=relation.target_phrase_id,
                        modifiers=relation.target_phrase_modifiers,
                    )
                elif relation.target_entity_id:
                    self.add_node(
                        target_id,
                        index,
                        type=_entity_type_label(relation.target_entity_type or "unknown"),
                        label=relation.target_canonical_name or relation.target_surface,
                        properties={
                            "canonical_id": relation.target_entity_id,
                            "surface": relation.target_surface,
                            "entity_type": relation.target_entity_type,
                        },
                    )
                    self.add_entity_taxonomy(
                        index,
                        target_id,
                        relation.target_entity_id,
                        relation.target_entity_type,
                    )
                else:
                    self.add_node(
                        target_id,
                        index,
                        type="UnresolvedReference",
                        label=relation.target_surface,
                        properties={"surface": relation.target_surface},
                    )
                self.add_edge(
                    source_id,
                    target_id,
                    relation.relation_type,
                    index,
                    properties={
                        "prep": relation.prep,
                        "confidence": relation.confidence,
                    },
                )

        new_nodes = sorted(
            (self.nodes[node_id] for node_id in set(self.nodes) - before_nodes),
            key=lambda node: (node["type"], node["label"], node["id"]),
        )
        new_edges = sorted(
            (self.edges[edge_id] for edge_id in set(self.edges) - before_edges),
            key=lambda edge: (edge["type"], edge["source"], edge["target"], edge["id"]),
        )
        self.sentence_summaries.append(
            {
                "index": index,
                "passage_id": passage.id,
                "new_node_ids": [node["id"] for node in new_nodes],
                "new_edge_ids": [edge["id"] for edge in new_edges],
                "new_node_count": len(new_nodes),
                "new_edge_count": len(new_edges),
            }
        )

    def add_quote(self, index: int, sentence_id: str, quote: ExtractedQuote) -> None:
        quote_id = f"quote:{quote.id}"
        self.add_node(
            quote_id,
            index,
            type="Quote",
            label=truncate_label(quote.text, 36),
            properties={
                "text": quote.text,
                "passage_id": quote.passage_id,
                "quote_start": quote.quote_start,
                "quote_end": quote.quote_end,
                "speaker_name": quote.speaker_name,
                "speech_verb": quote.speech_verb,
                "attribution_confidence": quote.attribution_confidence,
                "is_question": quote.is_question,
                "is_exclamation": quote.is_exclamation,
                "is_imperative": quote.is_imperative,
                "is_verse": quote.is_verse,
                "audience_type": quote.audience_type,
                "context_type": quote.context_type,
                "audience_confidence": quote.audience_confidence,
            },
        )
        self.add_edge(sentence_id, quote_id, "HAS_QUOTE", index)

        if quote.speaker_entity_id:
            speaker_id = f"entity:{quote.speaker_entity_id}"
            self.add_node(
                speaker_id,
                index,
                type="Character",
                label=quote.speaker_canonical_name or quote.speaker_name or quote.speaker_entity_id,
                properties={"canonical_id": quote.speaker_entity_id},
            )
            self.add_edge(speaker_id, quote_id, "SPOKE", index)
            self.add_entity_taxonomy(index, speaker_id, quote.speaker_entity_id, "character")

        if quote.addressee_entity_id:
            addressee_id = f"entity:{quote.addressee_entity_id}"
            self.add_node(
                addressee_id,
                index,
                type="Character",
                label=quote.addressee_canonical_name or quote.addressee_entity_id,
                properties={"canonical_id": quote.addressee_entity_id},
            )
            self.add_edge(quote_id, addressee_id, "ADDRESSED_TO", index)
            self.add_entity_taxonomy(index, addressee_id, quote.addressee_entity_id, "character")

    def add_argument(self, index: int, proposition_id: str, argument: PropositionArgument) -> None:
        if argument.entity_id:
            argument_id = f"entity:{argument.entity_id}"
            self.add_node(
                argument_id,
                index,
                type=_entity_type_label(argument.entity_type or "unknown"),
                label=argument.canonical_name or argument.surface,
                properties={
                    "canonical_id": argument.entity_id,
                    "surface": argument.surface,
                    "entity_type": argument.entity_type,
                },
            )
            self.add_entity_taxonomy(index, argument_id, argument.entity_id, argument.entity_type)
        elif argument.phrase_id:
            argument_id = f"phrase:{argument.phrase_id}"
            self.add_node(
                argument_id,
                index,
                type="NounPhrase",
                label=argument.surface,
                properties={
                    "surface": argument.surface,
                    "head": argument.phrase_head,
                    "modifiers": argument.phrase_modifiers,
                    "expected_type": argument.expected_type,
                    "reference_class": argument.reference_class.value
                    if argument.reference_class
                    else None,
                },
            )
            self.add_phrase_modifiers(
                index,
                phrase_node_id=argument_id,
                phrase_id=argument.phrase_id,
                modifiers=argument.phrase_modifiers,
            )
        else:
            if not is_reviewable_unresolved_argument(argument):
                return
            argument_id = f"unresolved:{slug_id(argument.surface)}"
            self.add_node(
                argument_id,
                index,
                type="UnresolvedReference",
                label=argument.surface,
                properties={
                    "surface": argument.surface,
                    "expected_type": argument.expected_type,
                    "reference_class": argument.reference_class.value
                    if argument.reference_class
                    else None,
                    "is_pronoun": argument.is_pronoun,
                },
            )

        self.add_edge(
            argument_id,
            proposition_id,
            "ARGUMENT_IN",
            index,
            role=argument.role.value,
            properties={
                "role": argument.role.value,
                "surface": argument.surface,
                "prep": argument.prep,
                "confidence": argument.confidence,
                "mention_start": argument.mention_start,
                "mention_end": argument.mention_end,
            },
            )

    def add_entity_taxonomy(
        self,
        index: int,
        entity_node_id: str,
        canonical_id: str,
        entity_type: str | None,
    ) -> None:
        """Attach seed-backed type and containment structure to an entity."""
        for taxonomy_node in self.taxonomy.taxonomies_for(canonical_id, entity_type):
            self.add_node(
                taxonomy_node.id,
                index,
                type=taxonomy_node.type,
                label=taxonomy_node.label,
                properties=taxonomy_node.properties,
            )
            self.add_edge(entity_node_id, taxonomy_node.id, "IS_A", index)

        if (entity_type or "").lower() == "place":
            parent_id = self.taxonomy.place_parent_id(canonical_id)
            if parent_id:
                parent_node_id = f"entity:{parent_id}"
                self.add_node(
                    parent_node_id,
                    index,
                    type="Place",
                    label=self.taxonomy.entity_label(parent_id),
                    properties=self.taxonomy.entity_properties(parent_id),
                )
                self.add_edge(entity_node_id, parent_node_id, "LOCATED_IN", index)
                for taxonomy_node in self.taxonomy.taxonomies_for(parent_id, "place"):
                    self.add_node(
                        taxonomy_node.id,
                        index,
                        type=taxonomy_node.type,
                        label=taxonomy_node.label,
                        properties=taxonomy_node.properties,
                    )
                    self.add_edge(parent_node_id, taxonomy_node.id, "IS_A", index)

        if (entity_type or "").lower() == "object":
            creator_id = self.taxonomy.object_creator_id(canonical_id)
            if creator_id:
                creator_node_id = f"entity:{creator_id}"
                self.add_node(
                    creator_node_id,
                    index,
                    type="Character",
                    label=self.taxonomy.entity_label(creator_id),
                    properties=self.taxonomy.entity_properties(creator_id),
                )
                self.add_edge(creator_node_id, entity_node_id, "CREATED", index)
                for taxonomy_node in self.taxonomy.taxonomies_for(creator_id, "character"):
                    self.add_node(
                        taxonomy_node.id,
                        index,
                        type=taxonomy_node.type,
                        label=taxonomy_node.label,
                        properties=taxonomy_node.properties,
                    )
                    self.add_edge(creator_node_id, taxonomy_node.id, "IS_A", index)

    def add_phrase_modifiers(
        self,
        index: int,
        *,
        phrase_node_id: str,
        phrase_id: str,
        modifiers: Iterable[str],
    ) -> None:
        for modifier in modifiers:
            modifier_id = f"modifier:{phrase_id}:{slug_id(modifier)}"
            self.add_node(
                modifier_id,
                index,
                type="Modifier",
                label=modifier,
                properties={"surface": modifier},
            )
            self.add_edge(
                phrase_node_id,
                modifier_id,
                "HAS_MODIFIER",
                index,
            )

    def add_node(
        self,
        node_id: str,
        sentence_index: int,
        *,
        type: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        properties = _clean(properties or {})
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "type": type,
                "label": label,
                "first_seen_sentence": sentence_index,
                "sentence_indices": [sentence_index],
                "properties": properties,
            }
            return

        node = self.nodes[node_id]
        if sentence_index not in node["sentence_indices"]:
            node["sentence_indices"].append(sentence_index)
        node["first_seen_sentence"] = min(node["first_seen_sentence"], sentence_index)
        node["properties"] = {**node["properties"], **{k: v for k, v in properties.items() if v not in (None, "", [])}}

    def add_edge(
        self,
        source: str,
        target: str,
        type: str,
        sentence_index: int,
        *,
        role: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        properties = _clean(properties or {})
        suffix = f":{slug_id(role)}" if role else ""
        edge_id = f"{source}->{type}{suffix}->{target}"
        if edge_id not in self.edges:
            self.edges[edge_id] = {
                "id": edge_id,
                "source": source,
                "target": target,
                "type": type,
                "role": role,
                "first_seen_sentence": sentence_index,
                "sentence_indices": [sentence_index],
                "properties": properties,
            }
            return

        edge = self.edges[edge_id]
        if sentence_index not in edge["sentence_indices"]:
            edge["sentence_indices"].append(sentence_index)
        edge["first_seen_sentence"] = min(edge["first_seen_sentence"], sentence_index)
        edge["properties"] = {**edge["properties"], **{k: v for k, v in properties.items() if v not in (None, "", [])}}

    @staticmethod
    def entity_node_id(entity: Any) -> str:
        if entity.canonical_id:
            return f"entity:{entity.canonical_id}"
        return f"unresolved:{slug_id(entity.extracted.text)}"

    @staticmethod
    def entity_ref_node_id(entity_id: str | None, text: str, entity_type: str | None) -> str:
        if entity_id:
            return f"entity:{entity_id}"
        return f"unresolved:{slug_id(entity_type or 'entity')}:{slug_id(text)}"


def _entity_type_label(entity_type: str | None) -> str:
    mapping = {
        "character": "Character",
        "place": "Place",
        "object": "Object",
        "event": "Event",
    }
    return mapping.get((entity_type or "unknown").lower(), "UnresolvedReference")


def _load_seed_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["id"]): item
        for item in raw
        if isinstance(item, dict) and item.get("id")
    }


def is_reviewable_proposition(proposition: ExtractedProposition) -> bool:
    """Return whether a proposition has enough anchoring to inspect in the UI."""
    if proposition.noun_phrase_relations:
        return True
    return any(arg.entity_id or arg.phrase_id for arg in proposition.arguments)


def is_reviewable_unresolved_argument(argument: PropositionArgument) -> bool:
    """Keep unresolved arguments that are semantically useful review targets."""
    if argument.reference_class is None:
        return False
    return argument.reference_class.value != "unknown"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean(item)
            for key, item in value.items()
            if item not in (None, "", [])
        }
    if isinstance(value, list):
        return [_clean(item) for item in value if item not in (None, "", [])]
    if hasattr(value, "model_dump"):
        return _clean(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return _clean(asdict(value))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--book-title", default=DEFAULT_BOOK_TITLE)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_review_dataset(
        args.source,
        book_title=args.book_title,
        limit=args.limit,
        seed_dir=args.seed_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = dataset["metadata"]
    print(
        f"Wrote {args.output} with {metadata['sentence_count']} sentences, "
        f"{metadata['node_count']} nodes, {metadata['edge_count']} edges."
    )


if __name__ == "__main__":
    main()
