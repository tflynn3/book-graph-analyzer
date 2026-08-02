"""Hosted-model fallback resolution for unresolved graph references."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from huggingface_hub import InferenceClient

CHARACTER_REFERENCE_CLASSES = {"canon_candidate", "bridging"}
NON_CHARACTER_EXPECTED_TYPES = {
    "artifact",
    "object",
    "place",
    "setting",
    "song",
    "poem",
    "location",
}
SECTION_NOISE_TOKENS = {
    "appendix",
    "appendices",
    "book",
    "chapter",
    "contents",
    "epilogue",
    "foreword",
    "index",
    "introduction",
    "part",
    "preface",
    "prologue",
}
PLURAL_GROUP_TOKENS = {
    "brandybucks",
    "dwarves",
    "elves",
    "fallohides",
    "harfoots",
    "hobbits",
    "men",
    "orcs",
    "riders",
    "stoors",
}
HONORIFIC_TOKENS = {"lady", "lord", "master", "miss", "mr", "mrs", "ms", "sir"}
TITLECASE_LOWER_WORDS = {"a", "an", "and", "at", "in", "of", "on", "the", "to"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "of",
    "the",
    "to",
    "in",
    "on",
    "for",
    "with",
    "mr",
    "mrs",
    "sir",
    "lady",
    "lord",
    "king",
}
ALIAS_HINTS = {
    "warily bilbo": ["Bilbo Baggins"],
    "bilbo": ["Bilbo Baggins"],
    "frodo": ["Frodo Baggins"],
    "strider": ["Aragorn"],
    "the lady": ["Galadriel"],
    "lady": ["Galadriel"],
    "gwaihir": ["Lord of the Eagles"],
    "the shadow": ["Sauron"],
    "his shadow": ["Sauron"],
    "shadow": ["Sauron"],
    "merry": ["Meriadoc Brandybuck"],
    "pippin": ["Peregrin Took"],
    "sam": ["Samwise Gamgee"],
    "samwise": ["Samwise Gamgee"],
    "theoden": ["Theoden"],
    "eomer": ["Eomer"],
    "eowyn": ["Eowyn"],
}


@dataclass
class InventoryEntity:
    entity_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class ResolutionSuggestion:
    ref_id: str
    mention_text: str
    source_book: str | None
    reference_class: str | None
    stage1_verdict: str
    action: str
    entity_id: str | None = None
    entity_name: str | None = None
    shortlist: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = "auto"
    applied: bool = False
    score: float = 0.0
    error: str | None = None

    def to_write_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("ref_id")
        return payload


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().replace("_", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(value: str | None) -> list[str]:
    normalized = normalize_text(value)
    return [token for token in normalized.split() if token and token not in STOPWORDS]


def raw_word_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", str(value).lower())


def core_name_tokens(value: str | None) -> list[str]:
    tokens = tokenize(value)
    while tokens and tokens[0] in HONORIFIC_TOKENS:
        tokens = tokens[1:]
    return tokens


def canonicalize_entity_name_for_display(value: str | None) -> str:
    raw = " ".join(str(value or "").strip().split())
    if not raw:
        return ""
    if normalize_text(raw) == "none":
        return ""

    words: list[str] = []
    for token in raw.split():
        lower = token.lower()
        if lower in TITLECASE_LOWER_WORDS and words:
            words.append(lower)
            continue
        if "-" in token:
            parts = []
            for part in token.split("-"):
                if not part:
                    continue
                part_lower = part.lower()
                if part_lower in TITLECASE_LOWER_WORDS and parts:
                    parts.append(part_lower)
                elif part[:1].islower() or part.isupper():
                    parts.append(part[:1].upper() + part[1:])
                else:
                    parts.append(part)
            words.append("-".join(parts))
            continue
        if token[:1].islower() or token.isupper():
            words.append(token[:1].upper() + token[1:])
        else:
            words.append(token)
    return " ".join(words)


def materialized_entity_id(value: str | None) -> str:
    normalized = normalize_text(value)
    return normalized.replace(" ", "_")


def build_inventory_entities(rows: list[dict[str, Any]]) -> list[InventoryEntity]:
    inventory: list[InventoryEntity] = []
    for row in rows:
        entity_id = str(row.get("entity_id") or "").strip()
        canonical_name = str(row.get("canonical_name") or "").strip()
        if not entity_id or not canonical_name:
            continue
        aliases = [
            str(alias).strip()
            for alias in (row.get("aliases") or [])
            if str(alias).strip()
        ]
        inventory.append(
            InventoryEntity(
                entity_id=entity_id,
                canonical_name=canonical_name,
                aliases=aliases,
            )
        )
    return inventory


def build_inventory_lookup(inventory: list[InventoryEntity]) -> dict[str, InventoryEntity]:
    lookup: dict[str, InventoryEntity] = {}
    for entity in inventory:
        lookup[normalize_text(entity.canonical_name)] = entity
        for alias in entity.aliases:
            lookup.setdefault(normalize_text(alias), entity)
    return lookup


def build_unique_token_index(inventory: list[InventoryEntity]) -> dict[str, list[str]]:
    token_map: dict[str, list[str]] = defaultdict(list)
    for entity in inventory:
        for token in set(tokenize(entity.canonical_name)):
            token_map[token].append(entity.canonical_name)
        for alias in entity.aliases:
            for token in set(tokenize(alias)):
                token_map[token].append(entity.canonical_name)
    return {token: names for token, names in token_map.items() if len(set(names)) == 1}


def normalize_enumish(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", "_", str(value).strip().lower())


def is_character_like_reference(row: dict[str, Any]) -> bool:
    ref_class = normalize_enumish(row.get("reference_class"))
    expected_type = normalize_enumish(row.get("expected_type"))
    if ref_class and ref_class not in CHARACTER_REFERENCE_CLASSES:
        return False
    if expected_type in NON_CHARACTER_EXPECTED_TYPES:
        return False
    if is_obvious_section_or_group_noise(row):
        return False
    return True


def is_obvious_section_or_group_noise(row: dict[str, Any]) -> bool:
    mention_tokens = raw_word_tokens(row.get("mention_text"))
    context_tokens = raw_word_tokens(row.get("context_text"))
    if not mention_tokens:
        return True

    if any(token.isdigit() for token in mention_tokens):
        return True

    if any(token in SECTION_NOISE_TOKENS for token in mention_tokens):
        return True

    if any(token in SECTION_NOISE_TOKENS for token in context_tokens):
        return True

    content_tokens = [token for token in mention_tokens if token not in STOPWORDS]
    if content_tokens and all(token in PLURAL_GROUP_TOKENS for token in content_tokens):
        return True

    mention_norm = normalize_text(row.get("mention_text"))
    context_norm = normalize_text(row.get("context_text"))
    if context_norm and mention_norm and mention_norm == context_norm and len(context_tokens) <= 12:
        return True

    return False


def is_safe_existing_auto_apply(mention_text: str, entity: InventoryEntity) -> bool:
    mention_norm = normalize_text(mention_text)
    mention_tokens = core_name_tokens(mention_text)
    if not mention_tokens:
        return False

    hint_targets = {
        canonical_name
        for alias_key, names in ALIAS_HINTS.items()
        if normalize_text(alias_key) == mention_norm
        for canonical_name in names
    }
    if entity.canonical_name in hint_targets:
        return True

    for variant in [entity.canonical_name, *entity.aliases]:
        variant_tokens = core_name_tokens(variant)
        if not variant_tokens:
            continue
        if mention_tokens == variant_tokens:
            return True
        if len(mention_tokens) == 1 and mention_tokens[0] in variant_tokens:
            return True

    return False


def group_materializable_new_entity_suggestions(
    rows: list[dict[str, Any]],
    *,
    min_support: int,
    min_score: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        display_name = canonicalize_entity_name_for_display(row.get("llm_resolution_entity_name"))
        if not display_name:
            continue
        grouped[normalize_text(display_name)].append(row)

    candidates: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        display_name = canonicalize_entity_name_for_display(
            group_rows[0].get("llm_resolution_entity_name")
        )
        support = len(group_rows)
        avg_score = (
            sum(float(row.get("llm_resolution_score") or 0.0) for row in group_rows)
            / support
        )
        if support < min_support or avg_score < min_score:
            continue

        aliases = sorted(
            {
                alias
                for alias in (
                    canonicalize_entity_name_for_display(row.get("mention_text"))
                    for row in group_rows
                )
                if alias and normalize_text(alias) != normalize_text(display_name)
            }
        )
        candidates.append(
            {
                "entity_id": materialized_entity_id(display_name),
                "canonical_name": display_name,
                "aliases": aliases,
                "support": support,
                "avg_score": round(avg_score, 4),
                "ref_ids": [
                    str(row.get("id") or "")
                    for row in group_rows
                    if str(row.get("id") or "")
                ],
                "source_books": sorted(
                    {
                        str(row.get("source_book") or "").strip()
                        for row in group_rows
                        if str(row.get("source_book") or "").strip()
                    }
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            -int(item["support"]),
            -float(item["avg_score"]),
            str(item["canonical_name"]),
        )
    )
    return candidates


def candidate_shortlist(
    row: dict[str, Any],
    inventory: list[InventoryEntity],
    unique_token_index: dict[str, list[str]],
    limit: int,
) -> list[str]:
    mention_norm = normalize_text(row.get("mention_text"))
    mention_tokens = tokenize(row.get("mention_text"))
    context_tokens = set(tokenize(row.get("context_text")))
    scored: dict[str, float] = {}

    def bump(name: str, points: float) -> None:
        if name:
            scored[name] = scored.get(name, 0.0) + points

    for candidate in row.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = normalize_text(str(candidate.get("canonical_id") or ""))
        surface = str(candidate.get("surface") or "").strip()
        confidence = float(candidate.get("confidence") or 0.0)
        for entity in inventory:
            if normalize_text(entity.entity_id) == candidate_id:
                bump(entity.canonical_name, 120.0 + confidence * 10.0)
                if surface and normalize_text(surface) == mention_norm:
                    bump(entity.canonical_name, 10.0)
                break

    for alias_key, names in ALIAS_HINTS.items():
        if mention_norm == alias_key or alias_key in mention_norm:
            for name in names:
                bump(name, 100.0)

    for entity in inventory:
        name_norm = normalize_text(entity.canonical_name)
        alias_norms = [normalize_text(alias) for alias in entity.aliases]
        all_names = [name_norm, *alias_norms]
        all_tokens = set()
        for alias in [entity.canonical_name, *entity.aliases]:
            all_tokens.update(tokenize(alias))

        if mention_norm in all_names:
            bump(entity.canonical_name, 80.0)
        elif any(mention_norm and mention_norm in alias for alias in all_names):
            bump(entity.canonical_name, 35.0)
        elif any(alias and alias in mention_norm for alias in all_names):
            bump(entity.canonical_name, 20.0)

        overlap = len(set(mention_tokens) & all_tokens)
        if overlap:
            bump(entity.canonical_name, overlap * 15.0)

        context_overlap = len(context_tokens & all_tokens)
        if context_overlap:
            bump(entity.canonical_name, min(context_overlap, 2) * 3.0)

    for token in mention_tokens:
        for name in unique_token_index.get(token, []):
            bump(name, 20.0)

    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    filtered = [name for name, score in ranked if score >= 10.0]
    return filtered[:limit]


def parse_assignment_line(
    content: str | None,
    reasoning_content: str | None,
    pattern: str,
    fallback_value: str,
) -> dict[str, str]:
    text = "\n".join(part for part in [content, reasoning_content] if part)
    if not text.strip():
        return {"value": fallback_value, "entity": ""}

    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        entity = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
        if normalize_text(entity) in {"", "none"}:
            entity = ""
        return {"value": value, "entity": entity}

    value_match = re.search(
        r"\b(character|reject|existing|new_entity)\b",
        text,
        flags=re.IGNORECASE,
    )
    value = value_match.group(1).lower() if value_match else fallback_value
    entity = ""
    entity_match = re.search(r"ENTITY\s*[:=]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if entity_match:
        entity = entity_match.group(1).strip()
    return {"value": value, "entity": entity}


class StagedHFUnresolvedResolver:
    """Resolve unresolved references through a staged hosted-model workflow."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        provider: str = "auto",
        timeout: float = 180.0,
        prompt_prefix: str = "",
        candidate_limit: int = 6,
    ) -> None:
        self.model = model
        self.provider = provider
        self.prompt_prefix = prompt_prefix.strip()
        self.candidate_limit = candidate_limit
        self.client = InferenceClient(provider=provider, timeout=timeout)

    def resolve_batch(
        self,
        refs: list[dict[str, Any]],
        inventory: list[InventoryEntity],
        *,
        apply_existing: bool = True,
    ) -> list[ResolutionSuggestion]:
        lookup = build_inventory_lookup(inventory)
        unique_token_index = build_unique_token_index(inventory)
        suggestions: list[ResolutionSuggestion] = []
        for row in refs:
            suggestions.append(
                self.resolve_reference(
                    row,
                    inventory=inventory,
                    lookup=lookup,
                    unique_token_index=unique_token_index,
                    apply_existing=apply_existing,
                )
            )
        return suggestions

    def resolve_reference(
        self,
        row: dict[str, Any],
        *,
        inventory: list[InventoryEntity],
        lookup: dict[str, InventoryEntity],
        unique_token_index: dict[str, list[str]],
        apply_existing: bool,
    ) -> ResolutionSuggestion:
        suggestion = ResolutionSuggestion(
            ref_id=str(row.get("id") or ""),
            mention_text=str(row.get("mention_text") or ""),
            source_book=row.get("source_book"),
            reference_class=row.get("reference_class"),
            stage1_verdict="skipped",
            action="skipped",
            model=self.model,
            provider=self.provider,
            notes=[],
        )
        if not suggestion.ref_id or not suggestion.mention_text:
            suggestion.notes.append("missing_id_or_mention")
            return suggestion

        if not is_character_like_reference(row):
            suggestion.notes.append("non_character_like")
            return suggestion

        stage1 = self._stage1_character_or_reject(row)
        suggestion.stage1_verdict = stage1
        if stage1 == "reject":
            suggestion.action = "reject"
            suggestion.score = 0.7
            return suggestion

        shortlist = candidate_shortlist(row, inventory, unique_token_index, self.candidate_limit)
        suggestion.shortlist = shortlist

        if shortlist:
            action, entity_name = self._stage2_existing_or_new(row, shortlist)
            if action == "existing" and entity_name:
                entity = lookup.get(normalize_text(entity_name))
                if entity is not None:
                    suggestion.action = "existing"
                    suggestion.entity_id = entity.entity_id
                    suggestion.entity_name = entity.canonical_name
                    suggestion.applied = bool(
                        apply_existing
                        and is_safe_existing_auto_apply(
                            suggestion.mention_text,
                            entity,
                        )
                    )
                    suggestion.score = 0.85 if suggestion.applied else 0.72
                    if suggestion.applied:
                        suggestion.notes.append("existing_from_shortlist")
                    else:
                        suggestion.notes.append("existing_requires_review")
                    return suggestion
                suggestion.notes.append("stage2_existing_not_in_inventory")
            else:
                suggestion.notes.append("stage2_new_entity")

        suggestion.action = "new_entity"
        suggestion.entity_name = self._stage3_canonicalize_new_entity(row)
        suggestion.score = 0.6 if suggestion.entity_name else 0.45
        if suggestion.entity_name:
            suggestion.notes.append("new_entity_name_generated")
        else:
            suggestion.notes.append("new_entity_name_missing")
        return suggestion

    def _chat(self, prompt: str, *, max_tokens: int) -> tuple[str | None, str | None]:
        rendered = f"{self.prompt_prefix}\n{prompt}" if self.prompt_prefix else prompt
        output = self.client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": rendered}],
            max_tokens=max_tokens,
            temperature=0,
        )
        message = output.choices[0].message
        return getattr(message, "content", None), getattr(message, "reasoning_content", None)

    def _stage1_character_or_reject(self, row: dict[str, Any]) -> str:
        prompt = (
            "Decide whether the mention refers to a single LOTR character/person "
            "or should be rejected.\n"
            "Reject places, objects, songs, plural groups, families, titles for "
            "groups, and unclear references.\n"
            "Return exactly one line:\n"
            "VERDICT=<character|reject>\n\n"
            f"Mention: {row.get('mention_text')}\n"
            f"Reference class: {row.get('reference_class') or '-'}\n"
            f"Expected type: {row.get('expected_type') or '-'}\n"
            f"Context: {row.get('context_text') or ''}\n"
        )
        content, reasoning = self._chat(prompt, max_tokens=16)
        parsed = parse_assignment_line(
            content,
            reasoning,
            r"VERDICT\s*=\s*(character|reject)",
            "reject",
        )
        return parsed["value"]

    def _stage2_existing_or_new(self, row: dict[str, Any], shortlist: list[str]) -> tuple[str, str]:
        candidate_block = ", ".join(shortlist)
        prompt = (
            "Resolve one LOTR character mention against existing graph candidates.\n"
            "The mention may be an alias, epithet, title, or nickname of a candidate.\n"
            "If the mention refers to a candidate's canonical identity, choose "
            "existing and use the exact candidate name.\n"
            "If none of the candidates fit, choose new_entity.\n"
            "Return exactly one line:\n"
            "ACTION=<existing|new_entity>; ENTITY=<exact candidate name or NONE>\n\n"
            f"Existing candidates: {candidate_block}\n"
            f"Mention: {row.get('mention_text')}\n"
            f"Context: {row.get('context_text') or ''}\n"
        )
        content, reasoning = self._chat(prompt, max_tokens=32)
        parsed = parse_assignment_line(
            content,
            reasoning,
            r"ACTION\s*=\s*(existing|new_entity)\s*;\s*ENTITY\s*=\s*([^\n\r]+)",
            "new_entity",
        )
        return parsed["value"], parsed["entity"]

    def _stage3_canonicalize_new_entity(self, row: dict[str, Any]) -> str:
        prompt = (
            "Canonicalize one LOTR character/person mention that is not already in the graph.\n"
            "Return exactly one line:\n"
            "ENTITY=<best canonical character name or NONE>\n\n"
            f"Mention: {row.get('mention_text')}\n"
            f"Context: {row.get('context_text') or ''}\n"
        )
        content, reasoning = self._chat(prompt, max_tokens=24)
        parsed = parse_assignment_line(
            content,
            reasoning,
            r"ENTITY\s*=\s*([^\n\r]+)",
            "reject",
        )
        return parsed["entity"] or parsed["value"]
