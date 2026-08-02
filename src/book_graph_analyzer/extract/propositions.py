"""Dense proposition extraction for sentence-level semantic scaffolding."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import spacy

from ..ingest.splitter import Passage
from ..models.propositions import (
    ArgumentRole,
    ExtractedQuote,
    ExtractedProposition,
    NounPhraseRelation,
    PropositionArgument,
    PropositionKind,
    ReferenceClass,
)
from .coref import ALL_PRONOUNS
from .resolver import ResolvedEntity
from .spacy_loader import load_spacy_model
from ..voice.dialogue import DialogueLine, extract_dialogue

_SPEECH_LEMMAS = {
    "answer",
    "ask",
    "call",
    "cry",
    "reply",
    "say",
    "shout",
    "speak",
    "tell",
    "whisper",
}
_MOVEMENT_LEMMAS = {
    "arrive",
    "come",
    "cross",
    "enter",
    "flee",
    "follow",
    "go",
    "journey",
    "leave",
    "return",
    "ride",
    "travel",
    "walk",
    "wander",
}
_POSSESSION_LEMMAS = {
    "bear",
    "bring",
    "carry",
    "find",
    "give",
    "grasp",
    "hand",
    "have",
    "hold",
    "keep",
    "lose",
    "receive",
    "take",
    "use",
    "wear",
    "wield",
}
_STATE_LEMMAS = {"appear", "be", "become", "remain", "seem", "stand"}
_PREP_ROLE_MAP = {
    "at": ArgumentRole.LOCATION,
    "in": ArgumentRole.LOCATION,
    "on": ArgumentRole.LOCATION,
    "within": ArgumentRole.LOCATION,
    "inside": ArgumentRole.LOCATION,
    "into": ArgumentRole.DESTINATION,
    "onto": ArgumentRole.DESTINATION,
    "to": ArgumentRole.DESTINATION,
    "toward": ArgumentRole.DESTINATION,
    "towards": ArgumentRole.DESTINATION,
    "from": ArgumentRole.SOURCE,
    "through": ArgumentRole.LOCATION,
    "across": ArgumentRole.LOCATION,
    "under": ArgumentRole.LOCATION,
    "over": ArgumentRole.LOCATION,
    "behind": ArgumentRole.LOCATION,
    "before": ArgumentRole.LOCATION,
    "after": ArgumentRole.LOCATION,
    "beside": ArgumentRole.LOCATION,
    "near": ArgumentRole.LOCATION,
    "with": ArgumentRole.INSTRUMENT,
    "by": ArgumentRole.INSTRUMENT,
    "using": ArgumentRole.INSTRUMENT,
    "for": ArgumentRole.TOPIC,
    "about": ArgumentRole.TOPIC,
    "of": ArgumentRole.DESCRIPTOR,
}
_NOUN_PHRASE_PREP_RELATIONS = {
    "at": "LOCATED_IN",
    "in": "LOCATED_IN",
    "inside": "LOCATED_IN",
    "on": "LOCATED_IN",
    "within": "LOCATED_IN",
    "near": "NEAR",
    "beside": "NEAR",
    "to": "DESTINATION",
    "toward": "DESTINATION",
    "towards": "DESTINATION",
    "into": "DESTINATION",
    "from": "SOURCE",
    "of": "DESCRIBED_BY",
    "with": "ASSOCIATED_WITH",
}
_FIRST_SECOND_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}
_FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
_FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}
_SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "yourself", "yourselves"}
_WH_PRONOUNS = {"who", "whom", "whose", "whoever", "whomever"}
_DISCOURSE_DEICTICS = {"this", "that", "these", "those", "it", "what", "which"}
_PRONOUN_SURFACES = ALL_PRONOUNS | _FIRST_SECOND_PRONOUNS | _WH_PRONOUNS | _DISCOURSE_DEICTICS
_POSSESSIVE_MARKERS = {"my", "your", "his", "her", "its", "our", "their"}
_GENERIC_DETERMINERS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "some",
    "any",
    "each",
    "every",
    "all",
    "another",
}
_BODY_PART_HEADS = {
    "arm",
    "arms",
    "blood",
    "bone",
    "bones",
    "breast",
    "brow",
    "ear",
    "ears",
    "eye",
    "eyes",
    "face",
    "finger",
    "fingers",
    "foot",
    "feet",
    "hair",
    "hand",
    "hands",
    "head",
    "heart",
    "leg",
    "legs",
    "mouth",
    "neck",
    "skin",
    "voice",
}
_BRIDGING_HEADS = {
    "door",
    "doors",
    "edge",
    "end",
    "gate",
    "gates",
    "hall",
    "hallway",
    "house",
    "light",
    "path",
    "paths",
    "road",
    "roads",
    "room",
    "shadow",
    "side",
    "top",
    "way",
}
_GENERIC_NP_HEADS = {
    "thing",
    "things",
    "one",
    "ones",
    "someone",
    "somebody",
    "something",
    "anyone",
    "anybody",
    "anything",
    "nothing",
    "all",
}
_CANON_CANDIDATE_STOPWORDS = _PRONOUN_SURFACES | {
    "to",
    "from",
    "with",
    "at",
    "by",
    "in",
    "on",
    "of",
    "for",
    "into",
    "onto",
    "through",
    "over",
    "under",
    "before",
    "after",
    "last",
    "next",
    "many",
    "none",
    "some",
    "all",
    "any",
    "part",
    "length",
    "night",
    "day",
    "days",
}
_COMMON_NONCANON_HEADS = _GENERIC_NP_HEADS | _BODY_PART_HEADS | _BRIDGING_HEADS | {
    "ground",
    "sun",
    "moon",
    "wind",
    "tree",
    "trees",
    "men",
    "man",
    "folk",
    "others",
    "side",
    "road",
    "roads",
    "way",
}
_OBJECT_LIKE_HEADS = {
    "ring",
    "sword",
    "staff",
    "stone",
    "palantir",
    "palantír",
    "blade",
    "crown",
    "horn",
    "jewel",
    "jewels",
    "gem",
    "gems",
}
_PLACE_LIKE_HEADS = {
    "shire",
    "land",
    "realm",
    "tower",
    "forest",
    "wood",
    "hall",
    "gate",
    "pass",
    "vale",
    "river",
    "marsh",
    "fen",
    "deep",
    "city",
    "hill",
    "downs",
    "mountain",
    "mountains",
}


@dataclass
class PropositionExtractionResult:
    """Propositions extracted from one passage."""

    passage: Passage
    propositions: list[ExtractedProposition]
    quotes: list[ExtractedQuote] | None = None


@dataclass
class _ReferenceResolution:
    entity_id: str | None
    canonical_name: str | None
    entity_type: str | None
    confidence: float


@dataclass
class _EntityMemory:
    entity_id: str
    canonical_name: str | None
    entity_type: str
    surface: str
    passage_id: str
    mention_start: int
    mention_end: int


@dataclass
class _QuoteContext:
    quote_id: str
    quote_start: int
    quote_end: int
    speaker: _ReferenceResolution | None = None
    addressee: _ReferenceResolution | None = None
    quote: ExtractedQuote | None = None


@dataclass
class _PassageContext:
    passage_id: str
    entities: list[ResolvedEntity]
    recent_entities: list[_EntityMemory]
    quote_contexts: list[_QuoteContext]


class PropositionExtractor:
    """Extract a dense proposition layer from sentence-level passages."""

    def __init__(self) -> None:
        self._nlp: spacy.Language | None = None
        self._recent_entities: list[_EntityMemory] = []
        self._recent_entity_limit = 120

    @property
    def nlp(self) -> spacy.Language:
        if self._nlp is None:
            self._nlp = load_spacy_model("en_core_web_sm")
        return self._nlp

    def extract_from_passage(
        self,
        passage: Passage,
        entities: list[ResolvedEntity],
    ) -> PropositionExtractionResult:
        """Extract propositions from a single passage."""
        doc = self.nlp(passage.text)
        noun_chunks = list(doc.noun_chunks)
        chunk_map = {
            token.i: chunk
            for chunk in noun_chunks
            for token in chunk
        }
        context = self._build_passage_context(passage, entities)

        propositions: list[ExtractedProposition] = []
        clause_index = 0

        for sent in doc.sents:
            for token in sent:
                if token.pos_ not in {"VERB", "AUX"}:
                    continue
                if token.dep_ in {"aux", "auxpass"}:
                    continue

                clause_index += 1
                propositions.extend(
                    self._extract_verb_propositions(
                        token=token,
                        sent=sent,
                        passage=passage,
                        entities=entities,
                        context=context,
                        clause_index=clause_index,
                        chunk_map=chunk_map,
                    )
                )

            modifier_props = self._extract_modifier_propositions(
                sent=sent,
                passage=passage,
                entities=entities,
                context=context,
                chunk_map=chunk_map,
                start_index=clause_index,
            )
            propositions.extend(modifier_props)
            clause_index += len(modifier_props)

        self._remember_entities(passage, entities)
        return PropositionExtractionResult(
            passage=passage,
            propositions=self._deduplicate(propositions),
            quotes=[context.quote for context in context.quote_contexts if context.quote],
        )

    def _extract_verb_propositions(
        self,
        *,
        token,
        sent,
        passage: Passage,
        entities: list[ResolvedEntity],
        context: _PassageContext,
        clause_index: int,
        chunk_map,
    ) -> list[ExtractedProposition]:
        kind = self._kind_for_verb(token)
        subject_role = (
            ArgumentRole.AGENT
            if kind in {PropositionKind.ACTION, PropositionKind.MOVEMENT, PropositionKind.SPEECH, PropositionKind.POSSESSION}
            else ArgumentRole.SUBJECT
        )

        subject_spans = self._collect_dep_spans(
            token,
            sent,
            chunk_map,
            deps={"nsubj", "nsubjpass", "csubj"},
            fallback_to_head=True,
        )
        direct_args: list[tuple[ArgumentRole, object]] = []
        prep_args: list[tuple[ArgumentRole, object, str]] = []

        for child in token.children:
            if child.dep_ in {"dobj", "obj", "attr", "acomp", "oprd", "iobj", "dative", "ccomp", "xcomp"}:
                role = self._role_for_dep(child.dep_)
                for span in self._expand_argument_spans(child, sent, chunk_map):
                    direct_args.append((role, span))
            elif child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ != "pobj":
                        continue
                    for span in self._expand_argument_spans(pobj, sent, chunk_map):
                        role = self._role_for_prep(child.lemma_.lower(), span, context)
                        prep_args.append((role, span, child.text))

        for child in token.children:
            if child.dep_ not in {"advcl", "obl", "npadvmod"}:
                continue
            if child.pos_ not in {"PROPN", "NOUN", "PRON"}:
                continue
            if child.i <= sent.start:
                continue

            prep_token = child.nbor(-1)
            if prep_token.text.lower() not in _PREP_ROLE_MAP:
                continue

            for span in self._expand_argument_spans(child, sent, chunk_map):
                role = self._role_for_prep(prep_token.lemma_.lower(), span, context)
                prep_args.append((role, span, prep_token.text))

            for prep_child in child.children:
                if prep_child.dep_ != "prep":
                    continue
                for pobj in prep_child.children:
                    if pobj.dep_ != "pobj":
                        continue
                    for span in self._expand_argument_spans(pobj, sent, chunk_map):
                        role = self._role_for_prep(prep_child.lemma_.lower(), span, context)
                        prep_args.append((role, span, prep_child.text))

        if kind == PropositionKind.STATE and any(role == ArgumentRole.ATTRIBUTE for role, _ in direct_args):
            kind = PropositionKind.ATTRIBUTE

        propositions: list[ExtractedProposition] = []
        main_args: list[PropositionArgument] = []
        main_phrase_relations: list[NounPhraseRelation] = []
        quote_context = self._quote_context_for_span(token.idx, token.idx + len(token.text), context.quote_contexts)

        for span in subject_spans:
            arg = self._build_argument(subject_role, span, context)
            if arg:
                main_args.append(arg)
                main_phrase_relations.extend(self._build_noun_phrase_relations(arg, span, context, chunk_map))

        for role, span in direct_args:
            if kind == PropositionKind.SPEECH and role == ArgumentRole.TOPIC:
                continue
            arg = self._build_argument(role, span, context)
            if arg:
                main_args.append(arg)
                main_phrase_relations.extend(self._build_noun_phrase_relations(arg, span, context, chunk_map))

        for role, span, prep_text in prep_args:
            arg = self._build_argument(role, span, context)
            if arg:
                arg.prep = prep_text.lower()
                main_args.append(arg)
                main_phrase_relations.extend(self._build_noun_phrase_relations(arg, span, context, chunk_map))

        if main_args:
            propositions.append(
                self._make_proposition(
                    passage=passage,
                    clause_index=clause_index,
                    kind=kind,
                    predicate_lemma=token.lemma_.lower(),
                    predicate_text=token.text,
                    predicate_start=token.idx,
                    predicate_end=token.idx + len(token.text),
                    clause_text=doc_text(token.subtree),
                    arguments=main_args,
                    quote_id=quote_context.quote_id if quote_context else None,
                    noun_phrase_relations=main_phrase_relations,
                    confidence=self._confidence_for_arguments(main_args, base=0.74),
                )
            )

        return propositions

    def _extract_modifier_propositions(
        self,
        *,
        sent,
        passage: Passage,
        entities: list[ResolvedEntity],
        context: _PassageContext,
        chunk_map,
        start_index: int,
    ) -> list[ExtractedProposition]:
        propositions: list[ExtractedProposition] = []
        clause_index = start_index

        for token in sent:
            if token.dep_ == "poss" and token.head.pos_ in {"NOUN", "PROPN"}:
                clause_index += 1
                possessor = self._build_argument(
                    ArgumentRole.POSSESSOR,
                    self._best_span(token, sent, chunk_map),
                    context,
                )
                possessed = self._build_argument(
                    ArgumentRole.POSSESSED,
                    self._best_span(token.head, sent, chunk_map),
                    context,
                )
                if possessor and possessed:
                    propositions.append(
                        self._make_proposition(
                            passage=passage,
                            clause_index=clause_index,
                            kind=PropositionKind.POSSESSION,
                            predicate_lemma="have",
                            predicate_text="'s",
                            predicate_start=token.idx,
                            predicate_end=token.idx + len(token.text),
                            clause_text=doc_text(self._best_span(token.head, sent, chunk_map)),
                            arguments=[possessor, possessed],
                            confidence=self._confidence_for_arguments([possessor, possessed], base=0.66),
                        )
                    )

            if token.dep_ == "appos":
                clause_index += 1
                referent = self._build_argument(
                    ArgumentRole.REFERENT,
                    self._best_span(token.head, sent, chunk_map),
                    context,
                )
                descriptor = self._build_argument(
                    ArgumentRole.DESCRIPTOR,
                    self._best_span(token, sent, chunk_map),
                    context,
                )
                if referent and descriptor:
                    propositions.append(
                        self._make_proposition(
                            passage=passage,
                            clause_index=clause_index,
                            kind=PropositionKind.APPOSITIVE,
                            predicate_lemma="be",
                            predicate_text="appos",
                            predicate_start=token.idx,
                            predicate_end=token.idx + len(token.text),
                            clause_text=doc_text(self._best_span(token, sent, chunk_map)),
                            arguments=[referent, descriptor],
                            confidence=self._confidence_for_arguments([referent, descriptor], base=0.64),
                        )
                    )

        return propositions

    def _collect_dep_spans(
        self,
        token,
        sent,
        chunk_map,
        *,
        deps: set[str],
        fallback_to_head: bool,
    ) -> list:
        spans = []
        for child in token.children:
            if child.dep_ in deps:
                spans.extend(self._expand_argument_spans(child, sent, chunk_map))
        if spans or not fallback_to_head:
            return spans

        if token.dep_ == "conj" and token.head is not None:
            for child in token.head.children:
                if child.dep_ in deps:
                    spans.extend(self._expand_argument_spans(child, sent, chunk_map))
        return spans

    def _expand_argument_spans(self, token, sent, chunk_map) -> list:
        roots = [token]
        roots.extend(child for child in token.children if child.dep_ == "conj")
        spans = []
        seen = set()
        for root in roots:
            span = self._best_span(root, sent, chunk_map)
            key = (span.start_char, span.end_char)
            if key in seen:
                continue
            seen.add(key)
            spans.append(span)
        return spans

    def _best_span(self, token, sent, chunk_map):
        chunk = chunk_map.get(token.i)
        if chunk is not None:
            return chunk
        return token.doc[token.i: token.i + 1]

    def _build_argument(
        self,
        role: ArgumentRole,
        span,
        context: _PassageContext,
    ) -> PropositionArgument | None:
        if not isinstance(context, _PassageContext):
            context = _PassageContext(
                passage_id="unknown",
                entities=list(context),
                recent_entities=list(self._recent_entities),
                quote_contexts=[],
            )
        surface = span.text.strip()
        if not surface:
            return None

        match = self._best_entity_match(span.start_char, span.end_char, surface, context, role)
        expected_type = self._expected_type_for_role(role, match.entity_type if match else None)
        is_pronoun = self._is_pronoun_surface(surface)
        reference_class = None
        if not (match and match.entity_id):
            reference_class = self._reference_class_for_unresolved(surface, role, expected_type)
        phrase_id = None
        phrase_head = None
        phrase_modifiers: list[str] = []
        if not (match and match.entity_id) and not is_pronoun and self._span_is_noun_phrase(span):
            phrase_id = self._noun_phrase_id(context.passage_id, span.start_char, span.end_char, surface)
            phrase_head = self._phrase_head(span)
            phrase_modifiers = self._phrase_modifiers(span)

        return PropositionArgument(
            role=role,
            surface=surface,
            entity_id=match.entity_id if match and match.entity_id else None,
            canonical_name=match.canonical_name if match else None,
            entity_type=match.entity_type if match else None,
            expected_type=expected_type,
            reference_class=reference_class,
            mention_start=span.start_char,
            mention_end=span.end_char,
            confidence=float(match.confidence if match else (0.25 if is_pronoun else 0.0)),
            is_pronoun=is_pronoun,
            phrase_id=phrase_id,
            phrase_head=phrase_head,
            phrase_modifiers=phrase_modifiers,
        )

    def _build_noun_phrase_relations(
        self,
        source_arg: PropositionArgument,
        span,
        context: _PassageContext,
        chunk_map,
    ) -> list[NounPhraseRelation]:
        """Extract relations that are syntactically attached to a noun phrase."""
        if not source_arg.phrase_id or not hasattr(span, "root"):
            return []

        relations: list[NounPhraseRelation] = []
        for child in span.root.children:
            if child.dep_ != "prep":
                continue
            prep = child.lemma_.lower()
            relation_type = _NOUN_PHRASE_PREP_RELATIONS.get(prep)
            if not relation_type:
                continue
            for pobj in child.children:
                if pobj.dep_ != "pobj":
                    continue
                target_span = self._best_span(pobj, pobj.sent, chunk_map)
                target_role = self._role_for_prep(prep, target_span, context)
                target_arg = self._build_argument(target_role, target_span, context)
                if not target_arg:
                    continue
                if not target_arg.entity_id and not target_arg.phrase_id:
                    continue
                relations.append(
                    NounPhraseRelation(
                        source_phrase_id=source_arg.phrase_id,
                        relation_type=relation_type,
                        target_surface=target_arg.surface,
                        prep=child.text,
                        target_entity_id=target_arg.entity_id,
                        target_entity_type=target_arg.entity_type,
                        target_canonical_name=target_arg.canonical_name,
                        target_phrase_id=target_arg.phrase_id,
                        target_phrase_head=target_arg.phrase_head,
                        target_phrase_modifiers=list(target_arg.phrase_modifiers),
                        confidence=max(0.55, float(target_arg.confidence or 0.0)),
                    )
                )

        return relations

    @staticmethod
    def _span_is_noun_phrase(span) -> bool:
        return hasattr(span, "root") and span.root.pos_ in {"NOUN", "PROPN"}

    @staticmethod
    def _noun_phrase_id(passage_id: str, start_char: int, end_char: int, surface: str) -> str:
        signature = f"{passage_id}|{start_char}|{end_char}|{surface.lower()}"
        return f"np-{hashlib.sha1(signature.encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _phrase_head(span) -> str | None:
        if not hasattr(span, "root"):
            return None
        return span.root.lemma_.lower() if span.root.lemma_ else span.root.text.lower()

    @staticmethod
    def _phrase_modifiers(span) -> list[str]:
        if not hasattr(span, "root"):
            return []
        modifiers = []
        root = span.root
        for token in span:
            if token == root:
                continue
            if token.dep_ in {"amod", "compound", "nummod", "poss"}:
                modifiers.append(token.text)
        for child in root.children:
            if child.dep_ in {"amod", "compound", "nummod", "poss"} and child.text not in modifiers:
                modifiers.append(child.text)
        return modifiers

    def _is_pronoun_surface(self, surface: str) -> bool:
        normalized = self._normalize_surface(surface)
        return normalized in _PRONOUN_SURFACES

    def _reference_class_for_unresolved(
        self,
        surface: str,
        role: ArgumentRole,
        expected_type: str | None,
    ) -> ReferenceClass:
        return self.classify_unresolved_reference(surface, role=role, expected_type=expected_type)

    @classmethod
    def classify_unresolved_reference(
        cls,
        surface: str,
        *,
        role: ArgumentRole | None = None,
        expected_type: str | None = None,
    ) -> ReferenceClass:
        normalized = cls._normalize_surface(surface)
        tokens = cls._surface_tokens(normalized)
        if not tokens:
            return ReferenceClass.UNKNOWN

        if normalized in _DISCOURSE_DEICTICS:
            return ReferenceClass.DISCOURSE_DEICTIC
        if normalized in _PRONOUN_SURFACES:
            return ReferenceClass.PRONOUN

        head = tokens[-1]
        if head in _BODY_PART_HEADS:
            return ReferenceClass.BODY_PART

        if tokens[0] in _POSSESSIVE_MARKERS:
            return ReferenceClass.BRIDGING
        if tokens[0] in {"the", "this", "that", "these", "those"} and head in _BRIDGING_HEADS:
            return ReferenceClass.BRIDGING

        if cls._looks_like_canon_candidate(surface) or cls._looks_like_expected_entity(tokens, expected_type):
            return ReferenceClass.CANON_CANDIDATE

        if head in _GENERIC_NP_HEADS:
            return ReferenceClass.GENERIC_NP
        if tokens[0] in _GENERIC_DETERMINERS:
            return ReferenceClass.GENERIC_NP

        if expected_type in {"character", "place", "object"} and role in {
            ArgumentRole.AGENT,
            ArgumentRole.SUBJECT,
            ArgumentRole.RECIPIENT,
            ArgumentRole.COMPANION,
            ArgumentRole.LOCATION,
            ArgumentRole.SOURCE,
            ArgumentRole.DESTINATION,
        } and cls._looks_like_expected_entity(tokens, expected_type):
            return ReferenceClass.CANON_CANDIDATE

        return ReferenceClass.UNKNOWN

    @staticmethod
    def _normalize_surface(surface: str) -> str:
        return re.sub(r"\s+", " ", surface.strip().lower())

    @staticmethod
    def _surface_tokens(surface: str) -> list[str]:
        return re.findall(r"[a-z]+(?:'[a-z]+)?", surface)

    @staticmethod
    def _looks_like_canon_candidate(surface: str) -> bool:
        tokens = [token.strip(".,;:!?\"'`()[]{}") for token in surface.split()]
        content_tokens = [token for token in tokens if token and token.lower() not in _GENERIC_DETERMINERS]
        if not content_tokens:
            return False
        lowered = [token.lower() for token in content_tokens]
        if len(content_tokens) == 1 and lowered[0] in _CANON_CANDIDATE_STOPWORDS:
            return False
        head = lowered[-1]
        if head in _CANON_CANDIDATE_STOPWORDS:
            return False
        if head in _COMMON_NONCANON_HEADS and not any(token[:1].isupper() for token in content_tokens):
            return False
        return any(token[:1].isupper() for token in content_tokens)

    @staticmethod
    def _looks_like_expected_entity(tokens: list[str], expected_type: str | None) -> bool:
        if not tokens or not expected_type:
            return False

        head = tokens[-1]
        if head in _CANON_CANDIDATE_STOPWORDS or head in _COMMON_NONCANON_HEADS:
            return False

        if expected_type == "object":
            return head in _OBJECT_LIKE_HEADS
        if expected_type == "place":
            return head in _PLACE_LIKE_HEADS
        return False

    def _best_entity_match(
        self,
        start_char: int,
        end_char: int,
        surface: str,
        context: _PassageContext,
        role: ArgumentRole,
    ) -> _ReferenceResolution | None:
        best: _ReferenceResolution | None = None
        best_score = 0.0
        surface_lower = surface.lower()

        for entity in context.entities:
            estart = entity.extracted.start_char
            eend = entity.extracted.end_char
            overlap = max(0, min(end_char, eend) - max(start_char, estart))

            score = 0.0
            if overlap:
                score = overlap / max(1, max(end_char - start_char, eend - estart))
            elif surface_lower == entity.extracted.text.lower():
                score = 0.95
            elif min(len(surface_lower), len(entity.extracted.text.lower())) >= 4 and (
                entity.extracted.text.lower() in surface_lower or surface_lower in entity.extracted.text.lower()
            ):
                score = 0.72

            if entity.canonical_id:
                score += 0.15
            score += min(0.1, float(entity.confidence or 0.0) * 0.1)

            if score > best_score:
                best = self._resolution_from_entity(entity, score)
                best_score = score

        if best_score >= 0.6:
            return best

        exact_recent = self._best_recent_entity_match(surface, role, context)
        if exact_recent is not None:
            return exact_recent

        if self._is_pronoun_surface(surface):
            return self._resolve_pronoun_reference(surface, start_char, end_char, role, context)

        return None

    @staticmethod
    def _resolution_from_entity(entity: ResolvedEntity, confidence: float | None = None) -> _ReferenceResolution:
        return _ReferenceResolution(
            entity_id=entity.canonical_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            confidence=float(confidence if confidence is not None else (entity.confidence or 0.0)),
        )

    @staticmethod
    def _resolution_from_memory(memory: _EntityMemory, confidence: float) -> _ReferenceResolution:
        return _ReferenceResolution(
            entity_id=memory.entity_id,
            canonical_name=memory.canonical_name,
            entity_type=memory.entity_type,
            confidence=float(confidence),
        )

    def _build_passage_context(
        self,
        passage: Passage,
        entities: list[ResolvedEntity],
    ) -> _PassageContext:
        quote_contexts: list[_QuoteContext] = []
        dialogue = extract_dialogue(passage.text, passage_id=passage.id, nlp=self.nlp)
        for line in dialogue.dialogue_lines:
            if line.quote_start is None or line.quote_end is None:
                continue
            speaker = self._resolve_named_reference(line.speaker, entities)
            addressee = self._resolve_quote_addressee(line, entities, speaker)
            quote_id = self._quote_id(passage.id, line.quote_start, line.quote_end, line.text)
            quote = ExtractedQuote(
                id=quote_id,
                passage_id=passage.id,
                text=line.text,
                quote_start=line.quote_start,
                quote_end=line.quote_end,
                speaker_name=line.speaker,
                speaker_entity_id=speaker.entity_id if speaker else None,
                speaker_canonical_name=speaker.canonical_name if speaker else None,
                addressee_entity_id=addressee.entity_id if addressee else None,
                addressee_canonical_name=addressee.canonical_name if addressee else None,
                speech_verb=line.speech_verb,
                attribution_confidence=line.attribution_confidence,
                is_question=line.is_question,
                is_exclamation=line.is_exclamation,
                is_imperative=line.is_imperative,
                is_verse=line.is_verse,
                audience_type=line.audience_type,
                context_type=line.context_type,
                audience_confidence=line.audience_confidence,
            )
            quote_contexts.append(
                _QuoteContext(
                    quote_id=quote_id,
                    quote_start=line.quote_start,
                    quote_end=line.quote_end,
                    speaker=speaker,
                    addressee=addressee,
                    quote=quote,
                )
            )
        return _PassageContext(
            passage_id=passage.id,
            entities=entities,
            recent_entities=list(self._recent_entities),
            quote_contexts=quote_contexts,
        )

    @staticmethod
    def _quote_id(passage_id: str, quote_start: int, quote_end: int, text: str) -> str:
        digest = hashlib.sha1(f"{passage_id}|{quote_start}|{quote_end}|{text}".encode("utf-8")).hexdigest()[:16]
        return f"quote-{digest}"

    def _remember_entities(
        self,
        passage: Passage,
        entities: list[ResolvedEntity],
    ) -> None:
        for entity in entities:
            if not entity.canonical_id:
                continue
            self._recent_entities.append(
                _EntityMemory(
                    entity_id=entity.canonical_id,
                    canonical_name=entity.canonical_name,
                    entity_type=entity.entity_type,
                    surface=entity.extracted.text,
                    passage_id=passage.id,
                    mention_start=entity.extracted.start_char,
                    mention_end=entity.extracted.end_char,
                )
            )
        if len(self._recent_entities) > self._recent_entity_limit:
            self._recent_entities = self._recent_entities[-self._recent_entity_limit:]

    def _resolve_named_reference(
        self,
        name: str | None,
        entities: list[ResolvedEntity],
    ) -> _ReferenceResolution | None:
        if not name:
            return None

        normalized = self._normalize_surface(name)
        for entity in entities:
            if not entity.canonical_id:
                continue
            if self._entity_name_matches(normalized, entity.extracted.text, entity.canonical_name):
                return self._resolution_from_entity(entity, max(0.7, float(entity.confidence or 0.7)))

        for memory in reversed(self._recent_entities):
            if self._memory_name_matches(normalized, memory):
                return self._resolution_from_memory(memory, 0.68)

        return None

    def _resolve_quote_addressee(
        self,
        line: DialogueLine,
        entities: list[ResolvedEntity],
        speaker: _ReferenceResolution | None,
    ) -> _ReferenceResolution | None:
        candidate_names: list[str] = []
        for context_text in (line.context_after, line.context_before):
            if not context_text:
                continue
            candidate_names.extend(
                match.group(1)
                for match in re.finditer(
                    r"\b(?:to|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                    context_text,
                )
            )
        candidate_names.extend(
            match.group(1)
            for match in re.finditer(r"(?:^|,\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:[,!?.]|$)", line.text)
        )

        seen_names: set[str] = set()
        for name in candidate_names:
            normalized = self._normalize_surface(name)
            if normalized in seen_names:
                continue
            seen_names.add(normalized)
            candidate = self._resolve_named_reference(name, entities)
            if candidate is None:
                continue
            if speaker and candidate.entity_id == speaker.entity_id:
                continue
            candidate.confidence = max(candidate.confidence, 0.62)
            return candidate

        if any(token in _SECOND_PERSON_PRONOUNS for token in self._surface_tokens(line.text.lower())):
            others = self._character_candidates(entities, speaker)
            if len(others) == 1:
                return others[0]
        return None

    def _character_candidates(
        self,
        entities: list[ResolvedEntity],
        speaker: _ReferenceResolution | None,
    ) -> list[_ReferenceResolution]:
        candidates: list[_ReferenceResolution] = []
        seen_ids: set[str] = set()
        for entity in entities:
            if entity.entity_type != "character" or not entity.canonical_id:
                continue
            if speaker and entity.canonical_id == speaker.entity_id:
                continue
            if entity.canonical_id in seen_ids:
                continue
            seen_ids.add(entity.canonical_id)
            candidates.append(self._resolution_from_entity(entity, max(0.55, float(entity.confidence or 0.55))))
        return candidates

    def _best_recent_entity_match(
        self,
        surface: str,
        role: ArgumentRole,
        context: _PassageContext,
    ) -> _ReferenceResolution | None:
        if self._is_pronoun_surface(surface):
            return None

        normalized = self._normalize_surface(surface)
        expected_type = self._expected_type_for_role(role, None)
        for memory in reversed(context.recent_entities):
            if expected_type and memory.entity_type != expected_type:
                continue
            if self._memory_name_matches(normalized, memory):
                return self._resolution_from_memory(memory, 0.58)
        return None

    def _resolve_pronoun_reference(
        self,
        surface: str,
        start_char: int,
        end_char: int,
        role: ArgumentRole,
        context: _PassageContext,
    ) -> _ReferenceResolution | None:
        normalized = self._normalize_surface(surface)
        quote_context = self._quote_context_for_span(start_char, end_char, context.quote_contexts)

        if normalized in _FIRST_PERSON_SINGULAR and quote_context and quote_context.speaker:
            speaker = quote_context.speaker
            return _ReferenceResolution(
                entity_id=speaker.entity_id,
                canonical_name=speaker.canonical_name,
                entity_type=speaker.entity_type,
                confidence=max(speaker.confidence, 0.86),
            )

        if normalized in _SECOND_PERSON_PRONOUNS and quote_context and quote_context.addressee:
            addressee = quote_context.addressee
            return _ReferenceResolution(
                entity_id=addressee.entity_id,
                canonical_name=addressee.canonical_name,
                entity_type=addressee.entity_type,
                confidence=max(addressee.confidence, 0.74),
            )

        expected_type = self._expected_type_for_role(role, None)
        if normalized in {"he", "him", "his", "himself", "she", "her", "hers", "herself", "who", "whom", "whose"}:
            return self._resolve_recent_antecedent(
                start_char,
                context,
                allowed_types={"character"},
                expected_type=expected_type or "character",
            )

        if normalized in {"it", "its", "itself"}:
            allowed_types = {"object", "place"} if expected_type not in {"place", "object"} else {expected_type}
            return self._resolve_recent_antecedent(
                start_char,
                context,
                allowed_types=allowed_types,
                expected_type=expected_type,
            )

        return None

    def _resolve_recent_antecedent(
        self,
        start_char: int,
        context: _PassageContext,
        *,
        allowed_types: set[str],
        expected_type: str | None,
    ) -> _ReferenceResolution | None:
        seen_ids: set[str] = set()
        current_candidates = sorted(
            (
                entity
                for entity in context.entities
                if entity.canonical_id and entity.extracted.start_char < start_char
            ),
            key=lambda entity: entity.extracted.start_char,
            reverse=True,
        )
        for entity in current_candidates:
            if entity.canonical_id in seen_ids:
                continue
            seen_ids.add(entity.canonical_id)
            if entity.entity_type not in allowed_types:
                continue
            if expected_type and entity.entity_type != expected_type and expected_type in {"character", "place", "object"}:
                continue
            return self._resolution_from_entity(entity, max(0.6, float(entity.confidence or 0.6)))

        for memory in reversed(context.recent_entities):
            if memory.entity_id in seen_ids:
                continue
            seen_ids.add(memory.entity_id)
            if memory.entity_type not in allowed_types:
                continue
            if expected_type and memory.entity_type != expected_type and expected_type in {"character", "place", "object"}:
                continue
            return self._resolution_from_memory(memory, 0.56)

        return None

    @staticmethod
    def _quote_context_for_span(
        start_char: int,
        end_char: int,
        quote_contexts: list[_QuoteContext],
    ) -> _QuoteContext | None:
        for context in quote_contexts:
            if context.quote_start <= start_char and end_char <= context.quote_end:
                return context
        return None

    @staticmethod
    def _entity_name_matches(normalized: str, surface: str, canonical_name: str | None) -> bool:
        surface_norm = PropositionExtractor._normalize_surface(surface)
        if normalized == surface_norm:
            return True
        if canonical_name and normalized == PropositionExtractor._normalize_surface(canonical_name):
            return True
        return False

    @staticmethod
    def _memory_name_matches(normalized: str, memory: _EntityMemory) -> bool:
        if normalized == PropositionExtractor._normalize_surface(memory.surface):
            return True
        if memory.canonical_name and normalized == PropositionExtractor._normalize_surface(memory.canonical_name):
            return True
        return False

    def _kind_for_verb(self, token) -> PropositionKind:
        lemma = token.lemma_.lower()
        if lemma in _SPEECH_LEMMAS:
            return PropositionKind.SPEECH
        if lemma in _MOVEMENT_LEMMAS:
            return PropositionKind.MOVEMENT
        if lemma in _POSSESSION_LEMMAS:
            return PropositionKind.POSSESSION
        if lemma in _STATE_LEMMAS:
            return PropositionKind.STATE
        return PropositionKind.ACTION

    def _role_for_dep(self, dep: str) -> ArgumentRole:
        if dep in {"iobj", "dative"}:
            return ArgumentRole.RECIPIENT
        if dep in {"attr", "acomp", "oprd"}:
            return ArgumentRole.ATTRIBUTE
        if dep in {"ccomp", "xcomp"}:
            return ArgumentRole.TOPIC
        return ArgumentRole.PATIENT

    def _role_for_prep(self, prep: str, span, context: _PassageContext) -> ArgumentRole:
        base_role = _PREP_ROLE_MAP.get(prep, ArgumentRole.LOCATION)
        if prep != "with":
            return base_role

        match = self._best_entity_match(span.start_char, span.end_char, span.text, context, ArgumentRole.COMPANION)
        if match and match.entity_type == "character":
            return ArgumentRole.COMPANION
        return base_role

    def _expected_type_for_role(
        self,
        role: ArgumentRole,
        matched_type: str | None,
    ) -> str | None:
        if matched_type in {"character", "place", "object"}:
            return matched_type
        if role in {ArgumentRole.AGENT, ArgumentRole.SUBJECT, ArgumentRole.RECIPIENT, ArgumentRole.COMPANION, ArgumentRole.REFERENT}:
            return "character"
        if role in {ArgumentRole.LOCATION, ArgumentRole.SOURCE, ArgumentRole.DESTINATION}:
            return "place"
        if role in {ArgumentRole.PATIENT, ArgumentRole.INSTRUMENT, ArgumentRole.POSSESSED}:
            return "object"
        return None

    def _confidence_for_arguments(
        self,
        arguments: list[PropositionArgument],
        *,
        base: float,
    ) -> float:
        if not arguments:
            return max(0.0, min(1.0, base - 0.2))
        avg = sum(float(arg.confidence or 0.0) for arg in arguments) / len(arguments)
        return max(0.0, min(1.0, base + avg * 0.25))

    def _make_proposition(
        self,
        *,
        passage: Passage,
        clause_index: int,
        kind: PropositionKind,
        predicate_lemma: str,
        predicate_text: str,
        predicate_start: int,
        predicate_end: int,
        clause_text: str,
        arguments: list[PropositionArgument],
        confidence: float,
        noun_phrase_relations: list[NounPhraseRelation] | None = None,
        quote_id: str | None = None,
    ) -> ExtractedProposition:
        signature = "|".join(
            [
                passage.id,
                str(clause_index),
                kind.value,
                predicate_lemma,
                ",".join(
                    f"{arg.role.value}:{arg.entity_id or arg.surface.lower()}"
                    for arg in arguments
                ),
            ]
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
        return ExtractedProposition(
            id=f"prop-{digest}",
            passage_id=passage.id,
            passage_text=passage.text,
            book=passage.book,
            sentence_num=passage.sentence_num,
            clause_index=clause_index,
            kind=kind,
            predicate_lemma=predicate_lemma,
            predicate_text=predicate_text,
            predicate_span_start=predicate_start,
            predicate_span_end=predicate_end,
            clause_text=clause_text[:280],
            quote_id=quote_id,
            confidence=confidence,
            extraction_method="dependency",
            modality=self._modality_for_text(predicate_text),
            polarity="negative" if " not " in f" {passage.text.lower()} " else "positive",
            arguments=arguments,
            noun_phrase_relations=list(noun_phrase_relations or []),
        )

    def _modality_for_text(self, predicate_text: str) -> str:
        text = predicate_text.lower()
        if any(modal in text for modal in {"could", "might", "would", "should", "must"}):
            return "modal"
        return "asserted"

    def _deduplicate(
        self,
        propositions: list[ExtractedProposition],
    ) -> list[ExtractedProposition]:
        seen: dict[str, ExtractedProposition] = {}
        for proposition in propositions:
            key = "|".join(
                [
                    proposition.kind.value,
                    proposition.predicate_lemma,
                    ",".join(
                        f"{arg.role.value}:{arg.entity_id or arg.surface.lower()}"
                        for arg in proposition.arguments
                    ),
                ]
            )
            existing = seen.get(key)
            if existing is None or proposition.confidence > existing.confidence:
                seen[key] = proposition
        return list(seen.values())


def doc_text(span_like) -> str:
    """Return normalized text from a spaCy span-like iterable."""
    if hasattr(span_like, "text"):
        return span_like.text.strip()
    return " ".join(tok.text for tok in span_like).strip()
