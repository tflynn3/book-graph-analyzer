from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from book_graph_analyzer.lore.sociolinguistic_registers import (
    SociolinguisticRegisterClassifier,
    ground_character_entity_id,
)
from book_graph_analyzer.models.worldbuilding import EditorialLayer, infer_editorial_layer


_GENERIC_AGENT_TOKENS = {
    "group", "party", "company", "army", "host", "troops", "people", "others",
    "elves", "dwarves", "men", "orcs", "goblins", "forces", "all", "they",
}


@dataclass
class MaterializationStats:
    books_processed: int = 0
    passages_written: int = 0
    passage_provenance_links: int = 0
    character_nodes_ensured: int = 0
    register_profiles_written: int = 0
    register_observations_written: int = 0
    editorial_links_written: int = 0


def _iter_events(payload: dict) -> list[dict]:
    events = payload.get("events", [])
    if isinstance(events, dict):
        return [v for v in events.values() if isinstance(v, dict)]
    if isinstance(events, list):
        return [v for v in events if isinstance(v, dict)]
    return []


def _looks_character_name(raw: str) -> bool:
    txt = re.sub(r"\s+", " ", raw.strip())
    if not txt or len(txt) < 2:
        return False
    low = txt.lower()
    if low.startswith("the "):
        low = low[4:]
    if low in _GENERIC_AGENT_TOKENS:
        return False
    if any(tok in low for tok in [" and ", " or ", "/", "&"]):
        return False
    tokens = re.findall(r"[A-Za-zÀ-ÿ'-]+", txt)
    if not tokens:
        return False
    if len(tokens) > 3:
        return False
    stop = {"the", "a", "an", "of", "from", "to"}
    core = [t for t in tokens if t.lower() not in stop]
    if not core:
        return False
    return all(len(t) >= 2 for t in core)


def _extract_character_names(agent: str | None, patient: str | None) -> list[str]:
    names: list[str] = []
    for field in [agent or "", patient or ""]:
        parts = re.split(r",|\band\b|;", field, flags=re.IGNORECASE)
        for p in parts:
            name = p.strip(" .\t\n\r\"'")
            if _looks_character_name(name):
                names.append(name)
    dedup: list[str] = []
    for n in names:
        if n.lower() not in {d.lower() for d in dedup}:
            dedup.append(n)
    return dedup


def _book_slug_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    return stem[:-7] if stem.endswith("_events") else stem


def _display_title_from_slug(slug: str) -> str:
    known = {
        "twotowers": "The Two Towers",
        "return": "The Return of the King",
        "fellowship": "The Fellowship of the Ring",
        "hobbit": "The Hobbit",
        "silmarillion": "The Silmarillion",
    }
    return known.get(slug, slug.replace("_", " ").title())


def _fallback_source(book_title: str, slug: str) -> EditorialLayer:
    return EditorialLayer(
        source_id=f"src_{slug}",
        source_title=book_title,
        editorial_status="published",
        author_period="middle",
        authority_weight=0.9,
    )


def materialize_from_event_artifacts(writer, event_files: list[Path], max_events_per_book: int = 120) -> dict:
    clf = SociolinguisticRegisterClassifier()
    stats = MaterializationStats()
    per_book: dict[str, dict[str, int]] = {}

    for ef in event_files:
        payload = json.loads(ef.read_text(encoding="utf-8"))
        events = _iter_events(payload)
        if not events:
            continue

        slug = _book_slug_from_filename(ef)
        guessed_title = _display_title_from_slug(slug)
        source = infer_editorial_layer(guessed_title) or infer_editorial_layer(slug) or _fallback_source(guessed_title, slug)
        book_title = source.source_title

        stats.books_processed += 1
        per_book[book_title] = {
            "events": 0,
            "characters": 0,
            "profiles": 0,
            "observations": 0,
            "editorial_links": 0,
            "passages": 0,
            "passage_attested": 0,
        }

        char_texts: dict[str, list[str]] = defaultdict(list)
        observations: list[tuple[str, str, str]] = []
        seen_chars: set[str] = set()

        attested_chars: set[str] = set()
        for e in events[:max_events_per_book]:
            event_id = str(e.get("id") or per_book[book_title]["events"] + 1)
            text = (e.get("description") or e.get("source_text") or "").strip()
            if not text:
                continue
            per_book[book_title]["events"] += 1

            passage_id = f"{slug}:event:{event_id}"
            writer.write_passage(
                passage_id=passage_id,
                text=text,
                book=book_title,
                chapter_num=0,
                paragraph_num=0,
                sentence_num=int(per_book[book_title]["events"]),
                source_id=source.source_id,
                source_title=source.source_title,
                source_stratum=getattr(source.default_stratum, "value", "core_text"),
                source_authority_weight=float(source.authority_weight),
                provenance_tags=["events", "auto_materialized"],
            )
            writer.write_passage_provenance(
                passage_id=passage_id,
                source_id=source.source_id,
                source_title=source.source_title,
                source_stratum=getattr(source.default_stratum, "value", "core_text"),
                authority_weight=float(source.authority_weight),
                confidence=1.0,
            )
            stats.passages_written += 1
            stats.passage_provenance_links += 1
            per_book[book_title]["passages"] += 1
            per_book[book_title]["passage_attested"] += 1

            for name in _extract_character_names(e.get("agent"), e.get("patient")):
                char_id = ground_character_entity_id(name)
                if not char_id:
                    continue
                writer.ensure_character_node(char_id, name)
                stats.character_nodes_ensured += 1
                seen_chars.add(char_id)
                char_texts[char_id].append(text)
                observations.append((char_id, text, passage_id))

                if char_id not in attested_chars:
                    writer.write_editorial_provenance(char_id, source, confidence=0.95)
                    attested_chars.add(char_id)
                    stats.editorial_links_written += 1
                    per_book[book_title]["editorial_links"] += 1

        for char_id, texts in char_texts.items():
            profile = clf.classify(" ".join(texts[:80]))
            writer.write_register_profile(char_id, profile, source_passage_id=observations[0][2] if observations else None)
            stats.register_profiles_written += 1
            per_book[book_title]["profiles"] += 1

        for char_id, text, passage_id in observations:
            obs = clf.classify(text)
            writer.write_register_observation(
                char_id,
                obs,
                observed_at=slug,
                source_passage_id=passage_id,
            )
            stats.register_observations_written += 1
            per_book[book_title]["observations"] += 1

        per_book[book_title]["characters"] = len(seen_chars)

    return {
        "global": stats.__dict__,
        "per_book": per_book,
    }
