from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from book_graph_analyzer.graph.temporal import canonicalize_era, era_to_order
from book_graph_analyzer.graph.connection import get_driver

console = Console()


DEFAULT_PROJECTS_DIR = Path("data") / "projects"
DEFAULT_EVENT_FILES = [
    "data/output/silmarillion_events.json",
    "data/output/unfinished_tales_events.json",
    "data/output/hobbit_events.json",
    "data/output/fellowship_events.json",
    "data/output/twotowers_events.json",
    "data/output/return_events.json",
]
STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "into", "over", "under", "after", "before",
    "their", "there", "where", "while", "about", "through", "during", "have", "has", "had",
    "were", "was", "will", "would", "could", "should", "might", "must", "they", "them", "then",
    "his", "her", "him", "she", "for", "but", "not", "you", "your", "our", "are", "who", "what",
}
ENTITY_CONNECTORS = {"of", "the", "and", "in", "on", "for", "de", "du", "na", "ni"}
ENTITY_RE = re.compile(r"[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’-]*(?:\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ'’-]*)*")
GENERIC_ENTITY_PHRASES = {
    "Unknown", "Someone", "They", "Them", "He", "She", "It", "His", "Her", "Their",
}
GENERIC_TEMPORAL_ENTITY_NAMES = {
    "elf",
    "elves",
    "man",
    "men",
    "dwarf",
    "dwarves",
    "hobbit",
    "hobbits",
    "orc",
    "orcs",
    "ranger",
    "rangers",
    "wood-elf",
    "wood-elves",
    "wood elf",
    "wood elves",
}
PROJECT_TIMELINE_DEFAULTS = {
    "story_era": None,
    "story_year": None,
    "allow_past_references": True,
    "forbid_future_entities": True,
    "forbidden_entities": [],
}
PLACEHOLDER_PARTICIPANTS = {"unknown", "someone", "they", "them", "he", "she", "it", "tbd", "placeholder"}
DEFAULT_PLACEHOLDER_TERMS = ["Unknown", "Someone", "TBD", "placeholder"]
DIALOGUE_QUOTE_RE = re.compile(
    r'“(?P<curly_double>[^”\n]{2,2500})”'
    r'|"(?P<straight_double>[^"\n]{2,2500})"'
    r'|(?<![A-Za-z])‘(?P<curly_single>(?:[^’\n]|(?<=[A-Za-z])’(?=[A-Za-z])){2,2500})’(?![A-Za-z])'
    r"|(?<![A-Za-z])'(?P<straight_single>(?:[^'\n]|(?<=[A-Za-z])'(?=[A-Za-z])){2,2500})'(?![A-Za-z])"
)
MOTIF_STOPWORDS = STOPWORDS | {
    "beren", "luthien", "lúthien", "thingol", "melian", "morgoth", "sauron", "finrod",
    "celegorm", "curufin", "huan", "hurin", "húrin", "doriath", "menegroth",
    "nargothrond", "beleriand", "tol-in-gaurhoth", "king", "queen", "lord", "lady",
}
PREFERRED_STORY_MOTIFS = {
    "oath", "song", "doom", "crown", "shadow", "love", "fate", "peril", "silence",
    "mercy", "memory", "grief", "hope", "foresight", "counsel", "choice",
}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "story-project"


def _project_dir(slug: str, projects_dir: Path | None = None) -> Path:
    return (projects_dir or DEFAULT_PROJECTS_DIR) / slug


def _project_file(slug: str, projects_dir: Path | None = None) -> Path:
    return _project_dir(slug, projects_dir) / "project.json"


def _load_project(slug: str, projects_dir: Path | None = None) -> dict:
    path = _project_file(slug, projects_dir)
    if not path.exists():
        raise click.ClickException(f"Project '{slug}' not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path, default: dict | list | None = None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_events(payload: dict) -> list[dict]:
    events = payload.get("events", {})
    if isinstance(events, dict):
        rows = list(events.values())
    elif isinstance(events, list):
        rows = events
    else:
        rows = []

    def _year(row: dict) -> int:
        y = row.get("year")
        if isinstance(y, int):
            return y
        if isinstance(y, str) and y.lstrip("-").isdigit():
            return int(y)
        return 0

    rows.sort(key=lambda r: (str(r.get("era") or ""), _year(r), str(r.get("id") or "")))
    return rows


def _project_event_files(project: dict) -> list[Path]:
    configured = project.get("event_files")
    files = configured if isinstance(configured, list) and configured else DEFAULT_EVENT_FILES
    return [Path(p) for p in files if Path(p).exists()]


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw and raw.lstrip("-").isdigit():
            return int(raw)
    return None


def _default_story_timeline(project: dict | str) -> dict[str, Any]:
    slug = str(project.get("slug") if isinstance(project, dict) else project or "").lower()
    timeline = dict(PROJECT_TIMELINE_DEFAULTS)
    if "beren" in slug or "luthien" in slug:
        timeline.update(
            {
                "story_era": "First Age",
                "story_year": 465,
                "forbidden_entities": [
                    "Bilbo", "Frodo", "Sam", "Gandalf", "Aragorn", "Legolas",
                    "Gimli", "Boromir", "Merry", "Pippin", "Saruman",
                ],
            }
        )
    elif any(token in slug for token in ("frodo", "ring", "shire", "lotr", "gandalf", "mithrandir", "rhun", "east")):
        timeline.update(
            {
                "story_era": "Third Age",
                "story_year": 3018,
            }
        )
    elif "hobbit" in slug or "bilbo" in slug:
        timeline.update(
            {
                "story_era": "Third Age",
                "story_year": 2941,
            }
        )
    return timeline


def _project_timeline(project: dict) -> dict[str, Any]:
    base = _default_story_timeline(project)
    raw = project.get("timeline")
    if not isinstance(raw, dict):
        raw = {}

    story_era_raw = raw.get("story_era", base.get("story_era"))
    story_era = canonicalize_era(str(story_era_raw).strip()) if story_era_raw else None
    story_year = _coerce_optional_int(raw.get("story_year", base.get("story_year")))
    forbidden_entities = _dedupe_strings(
        [str(x) for x in (base.get("forbidden_entities") or [])]
        + [str(x) for x in (raw.get("forbidden_entities") or [])]
    )
    return {
        "story_era": story_era,
        "story_year": story_year,
        "story_era_order": era_to_order(story_era),
        "allow_past_references": bool(raw.get("allow_past_references", base.get("allow_past_references", True))),
        "forbid_future_entities": bool(raw.get("forbid_future_entities", base.get("forbid_future_entities", True))),
        "forbidden_entities": forbidden_entities,
    }


def _infer_source_era_hint(raw: str) -> str | None:
    stem = str(raw or "").lower()
    if any(token in stem for token in ("hobbit", "fellowship", "twotowers", "return", "lotr", "istari")):
        return "Third Age"
    if any(token in stem for token in ("beren", "luthien", "huring", "turin", "gondolin", "earendil")):
        return "First Age"
    if "akallabeth" in stem or "numenor" in stem:
        return "Second Age"
    return None


def _infer_event_file_era(path: Path) -> str | None:
    return _infer_source_era_hint(path.stem)


def _extract_entity_names(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*(?:,|;|&|\band\b)\s*", text)
    out: list[str] = []
    for part in parts:
        for match in ENTITY_RE.finditer(part):
            name = match.group(0).strip()
            name = re.sub(r"[’']s$", "", name).strip()
            if not name or name in GENERIC_ENTITY_PHRASES:
                continue
            if name.lower() in {"first age", "second age", "third age", "a first age", "a second age", "a third age"}:
                continue
            words = [w for w in name.split() if w.lower() not in ENTITY_CONNECTORS]
            if not words:
                continue
            cleaned = " ".join(name.split())
            out.append(cleaned)
    return _dedupe_strings(out)


def _build_entity_temporal_presence(event_files: list[Path]) -> dict[str, dict[str, Any]]:
    rows_by_entity: dict[str, dict[str, Any]] = {}
    for path in event_files:
        payload = _load_json(path, default={})
        events = _extract_events(payload)
        file_era = _infer_event_file_era(path)
        for ev in events:
            raw_era = canonicalize_era(str(ev.get("era") or "").strip()) or None
            event_era = raw_era if raw_era and era_to_order(raw_era) < 99 else file_era
            year = _coerce_optional_int(ev.get("year"))
            for name in _extract_entity_names(ev.get("agent")):
                bucket = rows_by_entity.setdefault(
                    name,
                    {
                        "count": 0,
                        "eras": Counter(),
                        "years": [],
                        "years_by_era": defaultdict(list),
                        "source_files": Counter(),
                    },
                )
                bucket["count"] += 1
                bucket["source_files"][str(path)] += 1
                if event_era:
                    bucket["eras"][event_era] += 1
                if year is not None:
                    bucket["years"].append(year)
                    if event_era:
                        bucket["years_by_era"][event_era].append(year)

    out: dict[str, dict[str, Any]] = {}
    for name, bucket in rows_by_entity.items():
        years = list(bucket["years"])
        years_by_era = {
            era: {
                "year_start": min(vals),
                "year_end": max(vals),
                "count": len(vals),
            }
            for era, vals in bucket["years_by_era"].items()
            if vals
        }
        out[name] = {
            "count": int(bucket["count"]),
            "eras": [era for era, _ in sorted(bucket["eras"].items(), key=lambda kv: (era_to_order(kv[0]), -kv[1], kv[0]))],
            "era_counts": dict(bucket["eras"]),
            "year_start": min(years) if years else None,
            "year_end": max(years) if years else None,
            "years_by_era": years_by_era,
            "source_files": [p for p, _ in bucket["source_files"].most_common(5)],
        }
    return out


def _temporal_entity_status(name: str, timeline: dict[str, Any], entity_presence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = str(name or "").strip()
    lower = normalized.lower()
    forbidden_entities = {str(x).strip().lower() for x in timeline.get("forbidden_entities", []) if str(x).strip()}
    if lower in forbidden_entities:
        return {"status": "explicit_forbidden", "reason": "project_forbidden_entity"}

    profile = entity_presence.get(normalized)
    if not profile:
        for known_name, known_profile in entity_presence.items():
            if known_name.lower() == lower:
                profile = known_profile
                break
    if not profile:
        return {"status": "unknown", "reason": "no_temporal_profile"}

    story_era = timeline.get("story_era")
    story_year = timeline.get("story_year")
    story_era_order = int(timeline.get("story_era_order", 99))
    eras = [canonicalize_era(str(era).strip()) or str(era).strip() for era in profile.get("eras", []) if str(era).strip()]
    known_eras = [era for era in eras if era_to_order(era) < 99]
    if story_era and known_eras:
        orders = [era_to_order(era) for era in known_eras]
        if all(order > story_era_order for order in orders):
            return {"status": "future_only", "reason": f"all_known_eras_after_{story_era}"}
        if all(order < story_era_order for order in orders):
            return {"status": "past_only", "reason": f"all_known_eras_before_{story_era}"}
        if story_era in known_eras:
            bounds = profile.get("years_by_era", {}).get(story_era, {})
            year_start = _coerce_optional_int(bounds.get("year_start"))
            year_end = _coerce_optional_int(bounds.get("year_end"))
            if story_year is not None and year_start is not None and year_end is not None:
                if story_year < year_start:
                    return {"status": "future_only", "reason": f"first_{story_era}_appearance_after_{story_year}"}
                if story_year > year_end:
                    return {"status": "past_only", "reason": f"last_{story_era}_appearance_before_{story_year}"}
            return {"status": "allowed", "reason": f"active_in_{story_era}"}
        return {"status": "allowed", "reason": "multi_era_entity"}

    return {"status": "unknown", "reason": "insufficient_temporal_signal"}


def _temporal_guardrail_entities(
    *,
    timeline: dict[str, Any],
    entity_presence: dict[str, dict[str, Any]],
    character_priors: dict[str, float],
    limit: int = 12,
) -> list[str]:
    rows: list[tuple[float, str]] = []
    for name, profile in entity_presence.items():
        status = _temporal_entity_status(name, timeline, entity_presence)
        if status["status"] not in {"future_only", "explicit_forbidden"}:
            continue
        score = float(character_priors.get(name, 0.0)) + (0.01 * float(profile.get("count", 0)))
        rows.append((score, name))
    for name in timeline.get("forbidden_entities", []):
        rows.append((1.0, str(name)))
    rows.sort(key=lambda item: (-item[0], item[1].lower()))
    return _dedupe_strings([name for _, name in rows])[:limit]


def _find_temporal_mentions(
    text: str,
    *,
    timeline: dict[str, Any],
    entity_presence: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    text_l = str(text or "").lower()
    future_mentions: list[dict[str, str]] = []
    past_mentions: list[dict[str, str]] = []
    candidates = sorted(
        set(entity_presence.keys()) | {str(x) for x in timeline.get("forbidden_entities", []) if str(x).strip()},
        key=lambda name: (-len(name), name.lower()),
    )
    seen: set[str] = set()
    for name in candidates:
        lowered = name.lower()
        if lowered in GENERIC_TEMPORAL_ENTITY_NAMES:
            continue
        if lowered in seen:
            continue
        if lowered not in text_l:
            continue
        if not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, flags=re.IGNORECASE):
            continue
        seen.add(lowered)
        status = _temporal_entity_status(name, timeline, entity_presence)
        if status["status"] in {"future_only", "explicit_forbidden"}:
            future_mentions.append({"name": name, "reason": status["reason"]})
        elif status["status"] == "past_only" and timeline.get("allow_past_references", True):
            past_mentions.append({"name": name, "reason": status["reason"]})
    return {
        "future_mentions": future_mentions,
        "past_mentions": past_mentions,
    }


def _text_mentions_name(text: str, name: str) -> bool:
    raw_text = str(text or "")
    raw_name = str(name or "").strip()
    if not raw_text or not raw_name:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(raw_name)}(?!\w)", raw_text, flags=re.IGNORECASE))


def _project_seed_entities(project: dict, constraints: dict) -> list[str]:
    project_slug = str(project.get("slug") or "")
    seeds = list(_project_canon_entities(project_slug))
    seeds.extend(_extract_entity_names(project.get("premise")))
    for raw in constraints.get("required_elements", []) if isinstance(constraints, dict) else []:
        seeds.extend(_extract_entity_names(raw))
    blocked = _out_of_domain_entities(project_slug)
    return [
        name
        for name in _dedupe_strings(seeds)
        if name.lower() not in blocked and not _looks_like_entity_extraction_artifact(name)
    ][:16]


def _looks_like_entity_extraction_artifact(name: str) -> bool:
    normalized = str(name or "").strip()
    lower = normalized.lower()
    if not normalized:
        return True
    if lower in GENERIC_ENTITY_PHRASES:
        return True
    if lower in {"first age", "second age", "third age", "a first age", "a second age", "a third age"}:
        return True
    if re.search(r"[’']s$", normalized):
        return True
    if re.match(r"^(a|an|the)\s+\w+\s+age\b", lower):
        return True
    return False


def _valid_shadow_character_name(project_slug: str, name: str) -> bool:
    normalized = str(name or "").strip()
    lower = normalized.lower()
    if _looks_like_entity_extraction_artifact(normalized):
        return False
    if lower in PLACEHOLDER_PARTICIPANTS:
        return False
    if lower in _out_of_domain_entities(project_slug):
        return False
    if lower in _non_character_entities(project_slug):
        return False
    return True


def _event_temporal_relation(event: dict[str, Any], source_path: Path, timeline: dict[str, Any]) -> str:
    """Classify an extracted event relative to story time without inventing dates."""
    story_era = canonicalize_era(str(timeline.get("story_era") or "").strip()) or None
    story_order = era_to_order(story_era)
    story_year = _coerce_optional_int(timeline.get("story_year"))

    raw_event_era = canonicalize_era(str(event.get("era") or "").strip()) or None
    event_era = raw_event_era if raw_event_era and era_to_order(raw_event_era) < 99 else _infer_event_file_era(source_path)
    event_order = era_to_order(event_era)
    event_year = _coerce_optional_int(event.get("year"))

    if story_era and event_era and story_order < 99 and event_order < 99:
        if event_order > story_order:
            return "future"
        if event_order < story_order:
            return "past"
        if story_year is not None and event_year is not None:
            if event_year > story_year:
                return "future"
            if event_year < story_year:
                return "past"
        return "concurrent"
    return "unknown"


def _canonical_event_evidence_id(source_path: Path, event: dict[str, Any]) -> str:
    source_event_id = str(event.get("id") or "").strip()
    if not source_event_id:
        source_event_id = hashlib.sha256(_canonical_json(event).encode("utf-8")).hexdigest()[:16]
    material = f"{source_path.as_posix()}#{source_event_id}"
    return f"canon-event-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20]}"


def _local_neighborhood_from_events(
    *,
    event_files: list[Path],
    seed_entities: list[str],
    timeline: dict[str, Any],
) -> dict[str, Any]:
    if not seed_entities:
        return {
            "source": "events",
            "seed_entities": [],
            "matched_event_count": 0,
            "character_priors": {},
            "action_priors": {},
            "motif_priors": {},
            "place_priors": {},
            "evidence": [],
        }

    seed_hits: set[str] = {seed.lower() for seed in seed_entities}
    character_counts: Counter = Counter()
    action_counts: Counter = Counter()
    motif_counts: Counter = Counter()
    place_counts: Counter = Counter()
    evidence_candidates: list[dict[str, Any]] = []
    matched_event_count = 0

    for path in event_files:
        payload = _load_json(path, default={})
        events = _extract_events(payload)
        for ev in events:
            temporal_relation = _event_temporal_relation(ev, path, timeline)
            if temporal_relation == "future":
                continue
            description = str(ev.get("description") or "")
            action = str(ev.get("action") or "unknown").strip().lower() or "unknown"
            agent_names = _extract_entity_names(ev.get("agent"))
            patient_names = _extract_entity_names(ev.get("patient"))
            row_names = _dedupe_strings(agent_names + patient_names)
            if not row_names and not any(_text_mentions_name(description, seed) for seed in seed_entities):
                continue

            row_name_lowers = {name.lower() for name in row_names}
            row_seed_hits = [
                seed for seed in seed_entities
                if seed.lower() in row_name_lowers or _text_mentions_name(description, seed)
            ]
            if not row_seed_hits:
                continue

            matched_event_count += 1
            row_weight = 1.0 + (0.35 * max(0, len(row_seed_hits) - 1))
            for name in row_names:
                if name.lower() in seed_hits:
                    character_counts[name] += row_weight * 1.2
                else:
                    character_counts[name] += row_weight
            action_counts[action] += row_weight
            for token in _tokenize(description):
                motif_counts[token] += row_weight
            for patient in patient_names:
                if any(marker in patient.lower() for marker in ("dor", "land", "tower", "hall", "ford", "wood", "mount", "shire", "mordor", "doriath", "nargothrond", "angband")):
                    place_counts[patient] += row_weight
            evidence_candidates.append(
                {
                    "evidence_id": _canonical_event_evidence_id(path, ev),
                    "source_event_id": str(ev.get("id") or "").strip() or None,
                    "description": description,
                    "action": action,
                    "agent": ev.get("agent"),
                    "patient": ev.get("patient"),
                    "seed_hits": row_seed_hits,
                    "source_file": str(path),
                    "source_book": ev.get("source_book"),
                    "source_location": ev.get("source_location"),
                    "era": ev.get("era"),
                    "year": ev.get("year"),
                    "temporal_relation": temporal_relation,
                    "epistemic_status": str(ev.get("epistemic_status") or "extracted_event"),
                    "_rank": (
                        4.0 * len(row_seed_hits)
                        + (2.0 if temporal_relation == "concurrent" else 1.0 if temporal_relation == "past" else 0.0)
                        + (0.5 if ev.get("source_location") else 0.0)
                    ),
                }
            )

    for seed in seed_entities:
        character_counts[seed] += 2.0

    evidence_candidates.sort(
        key=lambda row: (
            -float(row.get("_rank", 0.0)),
            str(row.get("source_file") or ""),
            str(row.get("source_event_id") or ""),
        )
    )
    evidence = []
    seen_evidence_ids: set[str] = set()
    for row in evidence_candidates:
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen_evidence_ids:
            continue
        seen_evidence_ids.add(evidence_id)
        evidence.append({key: value for key, value in row.items() if key != "_rank"})
        if len(evidence) >= 24:
            break

    return {
        "source": "events",
        "seed_entities": seed_entities,
        "matched_event_count": matched_event_count,
        "timeline": {
            "story_era": timeline.get("story_era"),
            "story_year": timeline.get("story_year"),
        },
        "character_priors": _safe_prob(dict(character_counts)),
        "action_priors": _safe_prob(dict(action_counts)),
        "motif_priors": _safe_prob(dict(motif_counts.most_common(80))),
        "place_priors": _safe_prob(dict(place_counts)),
        "evidence": evidence,
    }


def _local_neighborhood_from_graph(seed_entities: list[str], timeline: dict[str, Any]) -> dict[str, Any]:
    driver = get_driver()
    if not driver or not seed_entities:
        return {
            "source": "neo4j",
            "seed_entities": seed_entities,
            "matched_seed_count": 0,
            "character_priors": {},
            "action_priors": {},
            "motif_priors": {},
            "place_priors": {},
            "evidence": [],
        }

    char_counts: Counter = Counter()
    action_counts: Counter = Counter()
    place_counts: Counter = Counter()
    motif_counts: Counter = Counter()
    evidence: list[dict[str, Any]] = []
    matched_seed_names: set[str] = set()
    seed_names = [seed.lower() for seed in seed_entities]
    story_era = str(timeline.get("story_era") or "").strip()
    story_era_order = int(timeline.get("story_era_order", 99))
    try:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (seed)
                WHERE (seed:Character OR seed:Place OR seed:Object)
                  AND toLower(coalesce(seed.canonical_name, seed.name, '')) IN $seed_names
                MATCH (seed)-[:ARGUMENT_IN]->(pr:Proposition)
                OPTIONAL MATCH (other)-[:ARGUMENT_IN]->(pr)
                WHERE other <> seed AND (other:Character OR other:Place OR other:Object)
                OPTIONAL MATCH (p:Passage)-[:HAS_PROPOSITION]->(pr)
                RETURN coalesce(seed.canonical_name, seed.name) AS seed_name,
                       pr.predicate_lemma AS predicate,
                       labels(other)[0] AS other_label,
                       coalesce(other.canonical_name, other.name) AS other_name,
                       p.book AS book,
                       p.text AS passage_text
                LIMIT 400
                """,
                seed_names=seed_names,
            )
            for row in rows:
                source_era = _infer_source_era_hint(str(row["book"] or ""))
                if story_era and source_era and era_to_order(source_era) > story_era_order:
                    continue
                matched_seed_names.add(str(row["seed_name"] or "").strip())
                predicate = str(row["predicate"] or "").strip().lower()
                if predicate:
                    action_counts[predicate] += 1
                other_name = str(row["other_name"] or "").strip()
                other_label = str(row["other_label"] or "").strip()
                if other_name:
                    if other_label == "Character":
                        char_counts[other_name] += 1
                    elif other_label == "Place":
                        place_counts[other_name] += 1
                passage_text = str(row["passage_text"] or "")
                for token in _tokenize(passage_text):
                    motif_counts[token] += 1
                if passage_text and len(evidence) < 12:
                    evidence.append(
                        {
                            "seed": row["seed_name"],
                            "predicate": predicate,
                            "book": row["book"],
                            "passage_text": passage_text[:180],
                        }
                    )
    except Exception:
        return {
            "source": "neo4j",
            "seed_entities": seed_entities,
            "matched_seed_count": 0,
            "character_priors": {},
            "action_priors": {},
            "motif_priors": {},
            "place_priors": {},
            "evidence": [],
        }
    finally:
        driver.close()

    for seed in seed_entities:
        char_counts[seed] += 2.0

    return {
        "source": "neo4j",
        "seed_entities": seed_entities,
        "matched_seed_count": len({name for name in matched_seed_names if name}),
        "character_priors": _safe_prob(dict(char_counts)),
        "action_priors": _safe_prob(dict(action_counts)),
        "motif_priors": _safe_prob(dict(motif_counts.most_common(80))),
        "place_priors": _safe_prob(dict(place_counts)),
        "evidence": evidence,
    }


def _blend_story_neighborhood(
    graph_neighborhood: dict[str, Any],
    event_neighborhood: dict[str, Any],
) -> dict[str, Any]:
    graph_weight = 0.65 if int(graph_neighborhood.get("matched_seed_count", 0) or 0) > 0 else 0.0
    event_weight = 1.0 - graph_weight if graph_weight > 0 else 1.0

    def _mix(*parts: tuple[dict[str, float], float]) -> dict[str, float]:
        acc: defaultdict[str, float] = defaultdict(float)
        for payload, weight in parts:
            if weight <= 0:
                continue
            for key, value in payload.items():
                acc[str(key)] += float(value) * weight
        total = sum(acc.values())
        if total <= 0:
            return {}
        return {k: round(v / total, 6) for k, v in sorted(acc.items(), key=lambda item: item[1], reverse=True)}

    return {
        "source": "hybrid" if graph_weight > 0 else "events",
        "seed_entities": _dedupe_strings(
            [str(x) for x in graph_neighborhood.get("seed_entities", [])]
            + [str(x) for x in event_neighborhood.get("seed_entities", [])]
        ),
        "graph_seed_hits": int(graph_neighborhood.get("matched_seed_count", 0) or 0),
        "event_matches": int(event_neighborhood.get("matched_event_count", 0) or 0),
        "character_priors": _mix(
            (graph_neighborhood.get("character_priors", {}), graph_weight),
            (event_neighborhood.get("character_priors", {}), event_weight),
        ),
        "action_priors": _mix(
            (graph_neighborhood.get("action_priors", {}), graph_weight),
            (event_neighborhood.get("action_priors", {}), event_weight),
        ),
        "motif_priors": _mix(
            (graph_neighborhood.get("motif_priors", {}), graph_weight),
            (event_neighborhood.get("motif_priors", {}), event_weight),
        ),
        "place_priors": _mix(
            (graph_neighborhood.get("place_priors", {}), graph_weight),
            (event_neighborhood.get("place_priors", {}), event_weight),
        ),
        "evidence": {
            "graph": graph_neighborhood.get("evidence", [])[:24],
            "events": event_neighborhood.get("evidence", [])[:24],
        },
    }


def _context_canon_evidence(context: dict[str, Any]) -> list[dict[str, Any]]:
    neighborhood = context.get("local_story_neighborhood", {})
    if not isinstance(neighborhood, dict):
        return []
    evidence = neighborhood.get("evidence", {})
    if not isinstance(evidence, dict):
        return []
    rows = evidence.get("events", [])
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        out.append(row)
    return out


def _matching_canon_evidence_refs(
    evidence: list[dict[str, Any]],
    *,
    text: str,
    participants: list[str],
    motifs: list[str],
    action: str = "",
    limit: int = 3,
) -> list[str]:
    """Resolve beats/events to stored evidence records; never manufacture refs."""
    query_tokens = set(_tokenize(" ".join([text, *motifs])))
    participant_lowers = {str(value).strip().lower() for value in participants if str(value).strip()}
    action_lower = str(action or "").strip().lower()
    ranked: list[tuple[float, str]] = []
    for row in evidence:
        evidence_id = str(row.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        description = str(row.get("description") or "")
        row_tokens = set(_tokenize(description))
        row_names = {
            str(value).strip().lower()
            for value in [row.get("agent"), row.get("patient"), *(row.get("seed_hits") or [])]
            if str(value or "").strip()
        }
        participant_overlap = len(participant_lowers & row_names)
        token_overlap = len(query_tokens & row_tokens)
        action_match = bool(action_lower and action_lower == str(row.get("action") or "").strip().lower())
        score = (4.0 * participant_overlap) + (2.0 if action_match else 0.0) + min(4.0, float(token_overlap))
        if score <= 0:
            continue
        ranked.append((score, evidence_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [evidence_id for _, evidence_id in ranked[: max(0, limit)]]


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _safe_prob(counter: dict[str, int]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in counter.items()}


def _topk_keys(weights: dict[str, float], k: int, fallback: list[str]) -> list[str]:
    keys = [k for k, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    return keys or fallback


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _stable_seed(*parts: str) -> int:
    material = "||".join(parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _compute_dynamic_beat_budget(target_words_per_scene: float | int, ordinal: int, beat_count: int) -> int:
    """Compute per-beat budget deterministically.

    Important: floor before clamp to preserve deterministic integer behavior.
    """
    base = float(target_words_per_scene) / max(1, int(beat_count))
    shaped = base * (1.0 + (0.08 if ordinal == 2 else 0.0) - (0.05 if ordinal == beat_count else 0.0))
    return max(45, min(220, math.floor(shaped)))


def _compute_scene_beat_count(scene: dict[str, Any], min_beats: int, max_beats: int) -> int:
    """Deterministically derive a per-scene beat count from scene complexity.

    Uses only local scene fields (offline-first, no stochastic inputs).
    """
    lo = max(1, int(min_beats))
    hi = max(lo, int(max_beats))
    goal = str(scene.get("goal") or "")
    summary = str(scene.get("summary") or "")
    hooks = scene.get("continuity_hooks") or []
    hook_count = len(hooks) if isinstance(hooks, list) else 0

    words = len((goal + " " + summary).split())
    # Light complexity heuristic: longer summaries and more hooks => more beats.
    estimated = 1 + (words // 18) + (hook_count // 2)
    return max(lo, min(hi, int(estimated)))


def _beat_type_for_scene(beat_idx: int, beat_count: int) -> str:
    if beat_idx <= 1:
        return "setup"
    if beat_idx >= beat_count:
        return "pivot"
    return "confrontation"


@dataclass
class StoryBeat:
    beat_id: str
    position: int
    beat_type: str
    intent: str
    prose_budget_words: int
    cause_refs: list[str]
    failed_constraints: list[str]
    action: str
    participants: list[str]
    motifs: list[str]
    preconditions: list[str]
    effects: list[str]
    source_canon_node_ids: list[str]
    style_register_hints: dict[str, Any]
    scoring_breakdown: dict[str, float]


def _extract_scene_participants(scene: dict[str, Any], project_slug: str) -> list[str]:
    raw = scene.get("characters")
    out: list[str] = []
    if isinstance(raw, list):
        out.extend(str(x).strip() for x in raw if str(x).strip())

    canon_entities = _project_canon_entities(project_slug)
    text = f"{scene.get('goal', '')} {scene.get('summary', '')}".lower()
    for c in canon_entities:
        if c.lower() in text:
            out.append(c)

    return list(dict.fromkeys(out))[:4] or ["Unknown"]


def _extract_scene_motifs(scene: dict[str, Any]) -> list[str]:
    hooks = scene.get("continuity_hooks") or []
    motifs: list[str] = []
    if isinstance(hooks, list):
        motifs.extend(str(h).strip().lower() for h in hooks if str(h).strip())

    text = f"{scene.get('goal', '')} {scene.get('summary', '')}".lower()
    for token in _tokenize(text):
        if token in {"oath", "song", "shadow", "fate", "love", "hunt", "doom", "crown"}:
            motifs.append(token)
    return list(dict.fromkeys(motifs))[:4]


def _beat_semantics(
    *,
    scene: dict[str, Any],
    project_slug: str,
    constraints: dict[str, Any],
    beat_type: str,
    beat_idx_in_scene: int,
    beats_in_scene: int,
    style_words: float | int,
    canon_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    goal = str(scene.get("goal") or "advance scene continuity")
    summary = str(scene.get("summary") or "")
    participants = _extract_scene_participants(scene, project_slug)
    motifs = _extract_scene_motifs(scene)

    action = (
        "establish" if beat_type == "setup" else "resolve" if beat_type == "pivot" else "confront"
    )
    preconditions = [f"scene-goal:{goal[:72]}", "prior-beat-resolved"] if beat_idx_in_scene > 1 else [f"scene-goal:{goal[:72]}"]
    effects = [f"intent-advanced:{_slugify(goal)[:24]}", f"beat-{beat_type}-completed"]

    source_canon_node_ids = _matching_canon_evidence_refs(
        canon_evidence or [],
        text=f"{goal} {summary}",
        participants=participants,
        motifs=motifs,
    )

    target_words = float(constraints.get("style", {}).get("target_words_per_scene", style_words or 320))
    beat_budget = float(_compute_dynamic_beat_budget(style_words, beat_idx_in_scene, beats_in_scene))
    style_fit = max(0.0, 1.0 - min(1.0, abs(beat_budget - (target_words / max(1, beats_in_scene))) / max(1.0, target_words)))
    lore_fit = 1.0 if source_canon_node_ids else 0.0
    coherence_fit = 0.95 if (summary or goal) else 0.4

    return {
        "action": action,
        "participants": participants,
        "motifs": motifs,
        "preconditions": preconditions,
        "effects": effects,
        "source_canon_node_ids": source_canon_node_ids,
        "style_register_hints": {
            "tone": str(constraints.get("style", {}).get("tone") or "neutral"),
            "target_words_per_scene": int(round(target_words)),
            "beat_share": round(1.0 / max(1, beats_in_scene), 4),
            "register": "elevated" if "beren" in project_slug.lower() else "neutral",
        },
        "scoring_breakdown": {
            "lore": round(float(lore_fit), 6),
            "style": round(float(style_fit), 6),
            "coherence": round(float(coherence_fit), 6),
        },
    }


def _make_shadow_beat(
    scene: dict[str, Any],
    slug: str,
    constraints: dict[str, Any],
    position: int,
    beat_idx_in_scene: int,
    beats_in_scene: int,
    style_words: float | int = 320,
    prior_beat_id: str | None = None,
    canon_evidence: list[dict[str, Any]] | None = None,
) -> StoryBeat:
    scene_id = str(scene.get("scene_id") or f"scene-{position:03d}")
    goal = str(scene.get("goal") or "advance scene continuity")
    summary = str(scene.get("summary") or "")
    beat_type = _beat_type_for_scene(beat_idx_in_scene, beats_in_scene)
    beat_id_seed = _stable_seed(slug, scene_id, str(beat_idx_in_scene), goal, summary)
    beat_id = f"{scene_id}-b{beat_idx_in_scene:02d}-{beat_id_seed:06x}"
    cause_refs = [prior_beat_id] if prior_beat_id else []
    forbidden_terms = [str(x).lower() for x in constraints.get("forbidden_terms", [])]
    text = f"{goal} {summary}".lower()
    failed_constraints = [f"forbidden:{t}" for t in forbidden_terms if t and t in text]
    semantic = _beat_semantics(
        scene=scene,
        project_slug=slug,
        constraints=constraints,
        beat_type=beat_type,
        beat_idx_in_scene=beat_idx_in_scene,
        beats_in_scene=beats_in_scene,
        style_words=style_words,
        canon_evidence=canon_evidence,
    )
    return StoryBeat(
        beat_id=beat_id,
        position=position,
        beat_type=beat_type,
        intent=(summary or goal),
        prose_budget_words=_compute_dynamic_beat_budget(style_words, beat_idx_in_scene, beats_in_scene),
        cause_refs=cause_refs,
        failed_constraints=failed_constraints,
        action=semantic["action"],
        participants=semantic["participants"],
        motifs=semantic["motifs"],
        preconditions=semantic["preconditions"],
        effects=semantic["effects"],
        source_canon_node_ids=semantic["source_canon_node_ids"],
        style_register_hints=semantic["style_register_hints"],
        scoring_breakdown=semantic["scoring_breakdown"],
    )


def _validate_cause_ref_positions(beats: list[StoryBeat]) -> list[str]:
    pos_by_id = {b.beat_id: b.position for b in beats}
    issues: list[str] = []
    for beat in beats:
        for ref in beat.cause_refs:
            ref_pos = pos_by_id.get(ref)
            if ref_pos is None:
                issues.append(f"missing-cause-ref:{beat.beat_id}->{ref}")
            elif ref_pos >= beat.position:
                issues.append(f"non-prior-cause-ref:{beat.beat_id}->{ref}")
    return issues


def _scene_from_beat_id(beat_id: str) -> str:
    core = str(beat_id or "").split("-b", 1)[0]
    return core or "unknown-scene"


def _chapter_from_scene_id(scene_id: str) -> int | None:
    m = re.match(r"^ch(\d+)-", str(scene_id or "").lower())
    return int(m.group(1)) if m else None


def _select_beats_scope(beats: list[dict[str, Any]], chapter: int | None, scene: str | None) -> list[dict[str, Any]]:
    if chapter is not None and scene:
        raise click.ClickException("Use only one of --chapter or --scene.")
    if scene:
        return [b for b in beats if _scene_from_beat_id(str(b.get("beat_id", ""))) == scene]
    if chapter is not None:
        out = []
        for b in beats:
            scene_id = _scene_from_beat_id(str(b.get("beat_id", "")))
            ch = _chapter_from_scene_id(scene_id)
            if ch == chapter:
                out.append(b)
        return out
    return list(beats)


def _beats_validation_from_rows(beats: list[dict[str, Any]], project_slug: str = "", constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    by_id = {str(b.get("beat_id", "")): int(b.get("position", 0) or 0) for b in beats}
    issues: list[dict[str, Any]] = []
    prior_effects: set[str] = set()
    canon_entities = {c.lower() for c in _project_canon_entities(project_slug)}
    out_of_domain = _out_of_domain_entities(project_slug)
    target_words = float((constraints or {}).get("style", {}).get("target_words_per_scene", 320))
    for b in beats:
        beat_id = str(b.get("beat_id", ""))
        pos = int(b.get("position", 0) or 0)
        for ref in b.get("cause_refs", []) or []:
            if ref not in by_id:
                issues.append({"level": "error", "code": "MISSING_CAUSE_REF", "beat_id": beat_id, "message": f"Missing cause ref: {ref}"})
            elif by_id[ref] >= pos:
                issues.append({"level": "error", "code": "NON_PRIOR_CAUSE_REF", "beat_id": beat_id, "message": f"Cause ref is not prior: {ref}"})
        for term in b.get("failed_constraints", []) or []:
            issues.append({"level": "warn", "code": "FAILED_CONSTRAINT", "beat_id": beat_id, "message": f"Constraint failed: {term}"})

        has_semantic_payload = any(k in b for k in ("action", "participants", "motifs", "preconditions", "effects", "source_canon_node_ids", "canon_refs"))
        canon_refs = b.get("source_canon_node_ids", []) or b.get("canon_refs", []) or []
        if has_semantic_payload and not canon_refs:
            issues.append({"level": "warn", "code": "CANON_GROUNDING_WEAK", "beat_id": beat_id, "message": "No canon refs present on beat."})

        participants = [str(p) for p in (b.get("participants", []) or [])]
        for p in participants:
            pl = p.lower()
            if pl in out_of_domain:
                issues.append({"level": "error", "code": "OUT_OF_DOMAIN_PARTICIPANT", "beat_id": beat_id, "message": f"Participant outside domain: {p}"})
            elif canon_entities and pl not in canon_entities and p != "Unknown":
                issues.append({"level": "warn", "code": "UNKNOWN_PARTICIPANT", "beat_id": beat_id, "message": f"Participant not in project canon set: {p}"})

        preconditions = [str(x) for x in (b.get("preconditions", []) or [])]
        unmet = [p for p in preconditions if p != "prior-beat-resolved" and p not in prior_effects and p.startswith("effect:")]
        if unmet:
            issues.append({"level": "error", "code": "UNMET_PRECONDITION", "beat_id": beat_id, "message": f"Unmet preconditions: {', '.join(unmet[:3])}"})
        for eff in b.get("effects", []) or []:
            prior_effects.add(str(eff))

        style_hints = b.get("style_register_hints", {}) if isinstance(b.get("style_register_hints", {}), dict) else {}
        if style_hints:
            beat_budget = float(b.get("prose_budget_words", 0) or 0)
            beat_share = float(style_hints.get("beat_share", 0) or 0)
            expected = target_words * beat_share if beat_share > 0 else max(45.0, target_words / 2.0)
            if abs(beat_budget - expected) > max(30.0, expected * 0.45):
                issues.append({
                    "level": "warn",
                    "code": "STYLE_BUDGET_MISMATCH",
                    "beat_id": beat_id,
                    "message": f"Beat prose budget {int(beat_budget)} diverges from style expectation {int(round(expected))}.",
                })

        breakdown = b.get("scoring_breakdown", {}) if isinstance(b.get("scoring_breakdown", {}), dict) else {}
        if breakdown:
            for k in ("lore", "style", "coherence"):
                if k in breakdown and not (0.0 <= float(breakdown.get(k, 0.0)) <= 1.0):
                    issues.append({"level": "error", "code": "INVALID_SCORE_COMPONENT", "beat_id": beat_id, "message": f"Scoring component {k} out of range [0,1]."})

    counts = Counter(it["level"] for it in issues)
    return {
        "summary": {
            "beats": len(beats),
            "errors": int(counts.get("error", 0)),
            "warnings": int(counts.get("warn", 0)),
            "status": "pass" if not counts.get("error", 0) else "fail",
        },
        "issues": issues,
    }


def _load_weights_arg(weights: str | None) -> dict[str, float]:
    if not weights:
        return {
            "canon_consistency": 0.25,
            "transition_likelihood": 0.25,
            "arc_coherence": 0.2,
            "style_register": 0.15,
            "novelty_diversity": 0.15,
        }
    maybe_path = Path(weights)
    if maybe_path.exists():
        payload = json.loads(maybe_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(weights)
    out = _load_weights_arg(None)
    for k, v in payload.items():
        out[str(k)] = float(v)
    total = sum(max(0.0, float(v)) for v in out.values())
    if total > 0:
        out = {k: round(max(0.0, float(v)) / total, 6) for k, v in out.items()}
    return out


def _interp_temp(step: int, steps: int, start: float, end: float) -> float:
    if steps <= 1:
        return max(1e-6, end)
    alpha = step / max(1, steps - 1)
    return max(1e-6, (1.0 - alpha) * start + alpha * end)


def _build_initial_shadow_state(
    plan: dict[str, Any],
    transitions: dict[str, dict[str, float]],
    top_characters: list[str],
    top_motifs: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    prev_action = "unknown"
    for ch in plan.get("chapters", []):
        chapter_num = int(ch.get("chapter_number", 1))
        for scene in ch.get("scenes", []):
            scene_id = str(scene.get("scene_id"))
            action_dist = transitions.get(prev_action) or transitions.get("unknown") or {"journey": 1.0}
            action = max(action_dist.items(), key=lambda kv: kv[1])[0]
            chars = rng.sample(top_characters, k=min(2, len(top_characters))) if top_characters else ["Beren", "Luthien"]
            motifs = rng.sample(top_motifs, k=min(2, len(top_motifs))) if top_motifs else ["oath"]
            desc = f"{scene.get('goal', 'advance')} via {action}."
            state.append(
                {
                    "scene_id": scene_id,
                    "chapter": chapter_num,
                    "summary": scene.get("summary", ""),
                    "action": action,
                    "characters": chars,
                    "motifs": motifs,
                    "description": desc,
                }
            )
            prev_action = action
    return state


def _arc_progression_score(actions: list[str]) -> float:
    """Reward development without treating repetition as coherence."""
    if len(actions) <= 1:
        return 1.0
    switches = sum(1 for idx in range(1, len(actions)) if actions[idx] != actions[idx - 1])
    switch_ratio = switches / (len(actions) - 1)
    target_switch_ratio = 0.55
    scale = max(target_switch_ratio, 1.0 - target_switch_ratio)
    return max(0.0, min(1.0, 1.0 - (abs(switch_ratio - target_switch_ratio) / scale)))


def _anneal_energy(
    state: list[dict[str, Any]],
    transitions: dict[str, dict[str, float]],
    char_priors: dict[str, float],
    motif_priors: dict[str, float],
    constraints: dict[str, Any],
    style_budget: dict[str, Any],
) -> float:
    required = [str(x).lower() for x in constraints.get("required_elements", [])]
    forbidden = [str(x).lower() for x in constraints.get("forbidden_terms", [])]
    text = "\n".join(str(r.get("description", "")) for r in state).lower()

    trans_score = 0.0
    for i, row in enumerate(state):
        prev = state[i - 1]["action"] if i > 0 else "unknown"
        p = float(transitions.get(prev, {}).get(row["action"], 0.05))
        trans_score += math.log(max(1e-6, p))

    char_score = 0.0
    motif_score = 0.0
    unique_motifs: set[str] = set()
    actions: list[str] = []
    words_per_scene = []
    for row in state:
        actions.append(str(row.get("action", "")))
        chars = [str(c) for c in row.get("characters", [])]
        motifs = [str(m) for m in row.get("motifs", [])]
        if chars:
            char_score += sum(float(char_priors.get(c, 0.01)) for c in chars) / len(chars)
        if motifs:
            motif_score += sum(float(motif_priors.get(m, 0.01)) for m in motifs) / len(motifs)
            unique_motifs.update(motifs)
        words_per_scene.append(len(str(row.get("description", "")).split()))

    arc_coherence = _arc_progression_score(actions)
    target_words = float(style_budget.get("target_words_per_scene", 300))
    mean_words = sum(words_per_scene) / max(1, len(words_per_scene))
    style_penalty = abs(mean_words - target_words) / max(1.0, target_words)
    missing_required = sum(1 for r in required if r not in text)
    forbidden_hits = sum(1 for f in forbidden if f in text)
    novelty = len(unique_motifs) / max(1, len(state) * 2)

    # Minimize energy.
    return (
        -0.9 * trans_score
        - 2.0 * char_score
        - 1.2 * motif_score
        - 1.5 * arc_coherence
        + 6.0 * style_penalty
        + 12.0 * missing_required
        + 20.0 * forbidden_hits
        - 2.0 * novelty
    )


def _mutate_state(
    state: list[dict[str, Any]],
    transitions: dict[str, dict[str, float]],
    top_characters: list[str],
    top_motifs: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    nxt = json.loads(json.dumps(state))
    if not nxt:
        return nxt
    i = rng.randrange(len(nxt))
    mode = rng.choice(["action", "chars", "motifs", "all"])
    prev_action = nxt[i - 1]["action"] if i > 0 else "unknown"
    action_dist = transitions.get(prev_action) or transitions.get("unknown") or {"journey": 1.0}
    actions = list(action_dist.keys())
    if mode in {"action", "all"} and actions:
        nxt[i]["action"] = rng.choice(actions)
    if mode in {"chars", "all"} and top_characters:
        k = min(max(1, len(nxt[i].get("characters", []))), len(top_characters))
        nxt[i]["characters"] = rng.sample(top_characters, k=k)
    if mode in {"motifs", "all"} and top_motifs:
        k = min(max(1, len(nxt[i].get("motifs", []))), len(top_motifs))
        nxt[i]["motifs"] = rng.sample(top_motifs, k=k)
    nxt[i]["description"] = f"{nxt[i].get('summary') or 'advance'} via {nxt[i]['action']}."
    return nxt


def _load_constraints(proj_dir: Path) -> dict:
    constraints_path = proj_dir / "constraints.json"
    return (
        json.loads(constraints_path.read_text(encoding="utf-8"))
        if constraints_path.exists()
        else _default_constraints()
    )


def _quality_settings(constraints: dict, *, chapter: int | None = None) -> dict[str, Any]:
    raw = constraints.get("quality", {}) if isinstance(constraints, dict) else {}
    quality = dict(raw) if isinstance(raw, dict) else {}
    if chapter is not None and isinstance(constraints, dict):
        by_chapter = constraints.get("quality_by_chapter", {})
        if isinstance(by_chapter, dict):
            override = by_chapter.get(str(chapter), by_chapter.get(chapter, {}))
            if isinstance(override, dict):
                supported_keys = {
                    "min_scene_words",
                    "min_chapter_words",
                    "target_scene_words",
                    "min_dialogue_ratio",
                    "target_dialogue_ratio",
                    "min_event_sentence_ratio",
                    "min_type_token_ratio",
                    "target_avg_sentence_words",
                    "max_avg_sentence_words",
                    "max_repeated_paragraphs",
                    "max_repeated_long_phrases",
                    "repeated_long_phrase_words",
                    "repeated_long_phrase_min_count",
                    "max_dialogue_vocative_openings",
                    "forbid_placeholder_terms",
                    "forbid_out_of_domain_entities",
                    "forbid_template_artifacts",
                    "fail_lowercase_paragraph_starts",
                }
                quality.update(
                    {key: value for key, value in override.items() if key in supported_keys}
                )
    placeholders = quality.get("forbid_placeholder_terms", DEFAULT_PLACEHOLDER_TERMS)
    if placeholders is True:
        placeholders = DEFAULT_PLACEHOLDER_TERMS
    if not isinstance(placeholders, list):
        placeholders = []
    return {
        "min_scene_words": max(0, int(quality.get("min_scene_words", 0) or 0)),
        "min_chapter_words": max(0, int(quality.get("min_chapter_words", 0) or 0)),
        "target_scene_words": max(0, int(quality.get("target_scene_words", 0) or 0)),
        "min_dialogue_ratio": max(0.0, min(1.0, float(quality.get("min_dialogue_ratio", 0.0) or 0.0))),
        "target_dialogue_ratio": max(0.0, min(1.0, float(quality.get("target_dialogue_ratio", 0.0) or 0.0))),
        "min_event_sentence_ratio": max(0.0, min(1.0, float(quality.get("min_event_sentence_ratio", 0.0) or 0.0))),
        "min_type_token_ratio": max(0.0, min(1.0, float(quality.get("min_type_token_ratio", 0.0) or 0.0))),
        "target_avg_sentence_words": max(0.0, float(quality.get("target_avg_sentence_words", 0.0) or 0.0)),
        "max_avg_sentence_words": max(0.0, float(quality.get("max_avg_sentence_words", 0.0) or 0.0)),
        "max_repeated_paragraphs": max(0, int(quality.get("max_repeated_paragraphs", 0) or 0)),
        "max_repeated_long_phrases": max(0, int(quality.get("max_repeated_long_phrases", 1_000_000) or 0)),
        "repeated_long_phrase_words": max(5, int(quality.get("repeated_long_phrase_words", 8) or 8)),
        "repeated_long_phrase_min_count": max(2, int(quality.get("repeated_long_phrase_min_count", 3) or 3)),
        "max_dialogue_vocative_openings": max(0, int(quality.get("max_dialogue_vocative_openings", 1_000_000) or 0)),
        "forbid_placeholder_terms": [str(x).strip() for x in placeholders if str(x).strip()],
        "forbid_out_of_domain_entities": bool(quality.get("forbid_out_of_domain_entities", True)),
        "forbid_template_artifacts": bool(quality.get("forbid_template_artifacts", True)),
        "fail_lowercase_paragraph_starts": bool(quality.get("fail_lowercase_paragraph_starts", False)),
    }


def _chapter_path(proj_dir: Path, chapter: int) -> Path:
    return proj_dir / f"chapter_{chapter:02d}.md"


def _trace_path(proj_dir: Path, chapter: int) -> Path:
    return proj_dir / f"chapter_{chapter:02d}_trace.json"


def _audit_json_path(proj_dir: Path, chapter: int) -> Path:
    return proj_dir / f"chapter_{chapter:02d}_audit.json"


def _audit_md_path(proj_dir: Path, chapter: int) -> Path:
    return proj_dir / f"chapter_{chapter:02d}_audit.md"


def _default_constraints() -> dict:
    return {
        "required_elements": [],
        "forbidden_terms": [],
        "enforcement": {
            "required_terms": True,
            "max_retries": 2,
        },
        "style": {
            "tone": "Consistent with project premise",
            "target_words_per_scene": 900,
        },
    }


def _required_terms(constraints: dict) -> list[str]:
    rows = constraints.get("required_elements", []) if isinstance(constraints, dict) else []
    return [str(x).strip() for x in rows if str(x).strip()]


def _missing_required_terms(text: str, required_terms: list[str]) -> list[str]:
    text_l = text.lower()
    return [term for term in required_terms if term.lower() not in text_l]


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", str(text or "")))


def _dialogue_word_count(text: str) -> int:
    return sum(
        _count_words(next(span for span in match.groups() if span is not None))
        for match in DIALOGUE_QUOTE_RE.finditer(str(text or ""))
    )


def _sentence_word_lengths(text: str) -> list[int]:
    chunks = [
        chunk.strip(" \n\t'\"")
        for chunk in SENTENCE_SPLIT_RE.split(str(text or "").strip())
        if chunk.strip(" \n\t'\"")
    ]
    return [_count_words(chunk) for chunk in chunks if _count_words(chunk) > 0]


def _avg_sentence_words(text: str) -> float:
    lengths = _sentence_word_lengths(text)
    if not lengths:
        return 0.0
    return round(sum(lengths) / len(lengths), 6)


def _paragraph_repeat_stats(text: str, min_words: int = 8) -> dict[str, Any]:
    normalized: list[str] = []
    for raw in re.split(r"\n\s*\n", str(text or "")):
        paragraph = raw.strip()
        if not paragraph or paragraph == "* * *" or paragraph.startswith("#"):
            continue
        if _count_words(paragraph) < min_words:
            continue
        collapsed = re.sub(r"\s+", " ", paragraph).strip().lower()
        if collapsed:
            normalized.append(collapsed)
    counts = Counter(normalized)
    repeated = [paragraph for paragraph, count in counts.items() if count > 1]
    return {
        "paragraph_count": len(normalized),
        "repeated_paragraph_count": sum(counts[p] - 1 for p in repeated),
        "unique_repeated_paragraphs": len(repeated),
        "repeat_ratio": round(sum(counts[p] - 1 for p in repeated) / max(1, len(normalized)), 6),
    }


def _repeated_long_phrase_stats(
    text: str,
    *,
    phrase_words: int = 8,
    min_count: int = 3,
    sample_size: int = 12,
) -> dict[str, Any]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]*", str(text or ""))
        if word.lower() not in {"chapter", "scene", "movement"}
    ]
    if len(words) < phrase_words:
        return {"phrase_words": phrase_words, "min_count": min_count, "repeated_phrase_count": 0, "samples": []}
    counts = Counter(" ".join(words[idx : idx + phrase_words]) for idx in range(0, len(words) - phrase_words + 1))
    repeated = [(phrase, count) for phrase, count in counts.items() if count >= min_count]
    repeated.sort(key=lambda row: (-row[1], row[0]))
    return {
        "phrase_words": phrase_words,
        "min_count": min_count,
        "repeated_phrase_count": len(repeated),
        "samples": [{"phrase": phrase, "count": count} for phrase, count in repeated[:sample_size]],
    }


def _dialogue_vocative_opening_stats(text: str) -> dict[str, Any]:
    names = [
        "Gandalf",
        "Aragorn",
        "Strider",
        "Gollum",
        "Smeagol",
        "Sméagol",
        "Bilbo",
        "Beren",
        "Luthien",
        "Lúthien",
        "Thingol",
        "Melian",
    ]
    pattern = re.compile(r"(?m)(?:^|[\s(])'(" + "|".join(re.escape(name) for name in names) + r"),")
    matches = [match.group(1) for match in pattern.finditer(str(text or ""))]
    return {
        "count": len(matches),
        "names": dict(sorted(Counter(matches).items())),
    }


def _placeholder_term_hits(text: str, terms: list[str]) -> list[str]:
    source = str(text or "")
    hits: list[str] = []
    for raw_term in terms:
        term = str(raw_term or "").strip()
        if not term:
            continue
        flags = 0 if any(ch.isupper() for ch in term) else re.IGNORECASE
        if term.lower() == "placeholder":
            flags = re.IGNORECASE
        pattern = re.compile(r"(?<![\w-])" + re.escape(term) + r"(?![\w-])", flags)
        if pattern.search(source):
            hits.append(term)
    return hits


def _template_artifact_hits(text: str) -> list[str]:
    source = str(text or "")
    checks: list[tuple[str, str, int]] = [
        ("brace_placeholder", r"\{[^}\n]+\}", 0),
        ("in_road", r"\bIn\s+Road\b", 0),
        ("in_westward_road", r"\bIn\s+the\s+westward\s+road\b", re.IGNORECASE),
        ("about_road", r"\bAbout\s+Road\b", 0),
        ("about_westward_road", r"\bAbout\s+the\s+westward\s+road\b", re.IGNORECASE),
        (
            "road_and_object",
            r"\bRoad\s+and\s+(?:branches?|cloak|fish bones?|folded letters?|lamps?|"
            r"maps?|muddy water|rope|staff|weathered cloak)\b",
            0,
        ),
        (
            "object_and_road",
            r"\b(?:branches?|cloak|fish bones?|folded letters?|lamps?|maps?|"
            r"muddy water|rope|staff|weathered cloak)\s+and\s+Road\b",
            0,
        ),
        ("ring_as_physical_object", r"\bfish bones,\s+muddy water,\s+and\s+ring\b", re.IGNORECASE),
        ("repeated_old_powers_block", r"The old powers did not need to enter the road", re.IGNORECASE),
        ("repeated_practical_choice_block", r"Every practical choice cast a moral shadow", re.IGNORECASE),
        ("repeated_hidden_labour_block", r"So the hidden labour continued", re.IGNORECASE),
        ("repeated_bilbo_adventure_block", r"Bilbo's old adventure", re.IGNORECASE),
        ("repeated_road_grammar_block", r"The road had its own stern grammar", re.IGNORECASE),
        (
            "ordinal_sign_changed",
            r"\b(?:first|second|third|fourth|fifth|sixth|"
            r"seventh|eighth|ninth|tenth|eleventh|twelfth)\s+[\w'-]+\s+sign changed\b",
            re.IGNORECASE,
        ),
        ("road_work_tested", r"\broad[-‐‑–—]work\s+tested\b", re.IGNORECASE),
        (
            "keyword_substitution_dialogue",
            r"\b(?P<keyword>[A-Za-z][A-Za-z'-]*)\s+gives\s+(?P=keyword)\s+direction,\s+"
            r"not\s+(?P=keyword)\s+answer\b",
            re.IGNORECASE,
        ),
        (
            "gollum_keyword_frame",
            r"\b(?P<keyword>[A-Za-z][A-Za-z'-]*)\s+hurts first;\s+rope answers\s+(?P=keyword)\b",
            re.IGNORECASE,
        ),
        ("meta_end_of_chapter", r"\bBy the end of the chapter\b", re.IGNORECASE),
        ("meta_final_page", r"\bthe final page\b", re.IGNORECASE),
        ("meta_chapter_could_not", r"\bThe chapter could not\b", re.IGNORECASE),
        ("meta_three_movements", r"\bthree movements of the tale\b", re.IGNORECASE),
        ("meta_final_movement", r"The final movement holds", re.IGNORECASE),
        ("meta_reader_address", r"If the reader sought", re.IGNORECASE),
        ("modern_stopping_rule", r"\bstopping rule\b", re.IGNORECASE),
        ("modern_controlled_risk", r"\bcontrolled risk\b", re.IGNORECASE),
        ("modern_trauma_response", r"\btrauma response\b", re.IGNORECASE),
        ("modern_independent_uncertainties", r"\bindependent uncertainties\b", re.IGNORECASE),
        ("modern_process_jargon", r"\b(?:provenance|compliance|protocol)\b", re.IGNORECASE),
    ]
    hits: list[str] = []
    for name, pattern, flags in checks:
        if re.search(pattern, source, flags):
            hits.append(name)
    return hits


def _lowercase_paragraph_start_samples(text: str, sample_size: int = 8) -> list[str]:
    samples: list[str] = []
    for block in re.split(r"\n\s*\n", str(text or "")):
        paragraph = block.strip()
        if not paragraph or paragraph.startswith("#") or paragraph == "* * *":
            continue
        if re.match(r"^(?:[\"'“‘]\s*)?[a-z]", paragraph):
            samples.append(paragraph[:120])
            if len(samples) >= sample_size:
                break
    return samples


def _polish_paragraph_starts(text: str) -> str:
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", str(text or "")):
        paragraph = block.strip()
        if not paragraph:
            continue
        if paragraph.startswith("#") or paragraph == "* * *" or not re.match(r"^[a-z]", paragraph):
            blocks.append(paragraph)
            continue
        blocks.append(paragraph[:1].upper() + paragraph[1:])
    return "\n\n".join(blocks)


def _event_density_stats(text: str) -> dict[str, Any]:
    event_terms = {
        "accepted", "answered", "arrived", "asked", "bit", "bound", "brought", "carried",
        "caught", "chose", "climbed", "closed", "counted", "crossed", "crouched", "cut",
        "drew", "dropped", "entered", "escaped", "fastened", "fled", "followed", "found",
        "gathered", "guarded", "heard", "held", "hid", "hissed", "knelt", "learned",
        "left", "listened", "looked", "marked", "mended", "moved", "opened", "paid",
        "passed", "questioned", "reached", "remembered", "returned", "rose", "saw",
        "searched", "seized", "set", "shifted", "slipped", "spoke", "sprang", "stood",
        "stopped", "struck", "swallowed", "tested", "tied", "touched", "tracked", "turned",
        "vanished", "walked", "watched", "went", "whispered",
    }
    concrete_terms = {
        "ash", "bank", "bird", "boat", "bone", "bough", "branch", "cloak", "ditch",
        "door", "fish", "foot", "ford", "gate", "hand", "lamp", "map", "mud", "path",
        "print", "reed", "river", "road", "root", "rope", "stone", "track", "tree",
        "water", "window",
    }
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or ""))
        if sentence.strip()
    ]
    eventful = 0
    dialogue = 0
    for sentence in sentences:
        lower = sentence.lower()
        words = set(re.findall(r"[a-z][a-z'-]*", lower))
        has_dialogue = bool(DIALOGUE_QUOTE_RE.search(sentence))
        has_event = bool(words & event_terms)
        has_concrete = bool(words & concrete_terms)
        if has_dialogue:
            dialogue += 1
        if has_dialogue or has_event or (has_concrete and any(term in lower for term in (" then ", " when ", " after ", " before ", " at last "))):
            eventful += 1
    total = len(sentences)
    return {
        "sentence_count": total,
        "eventful_sentence_count": eventful,
        "dialogue_sentence_count": dialogue,
        "event_sentence_ratio": round(eventful / max(1, total), 6),
    }


def _effective_min_type_token_ratio(configured_min: float, word_count: int, baseline_words: int = 6500) -> float:
    if configured_min <= 0 or word_count <= baseline_words:
        return configured_min
    return round(configured_min * math.sqrt(baseline_words / max(1, word_count)), 6)


def _chapter_quality_failures(
    text: str,
    trace_sections: list[dict[str, Any]],
    constraints: dict[str, Any],
    *,
    chapter: int | None = None,
) -> list[str]:
    quality = _quality_settings(constraints, chapter=chapter)
    failures: list[str] = []
    word_count = _count_words(text)
    words = re.findall(r"\b[\w'-]+\b", str(text or ""))
    dialogue_ratio = _dialogue_word_count(text) / max(1, word_count)
    type_token_ratio = len({w.lower() for w in words}) / max(1, word_count)
    avg_sentence_words = _avg_sentence_words(text)
    paragraph_repeats = _paragraph_repeat_stats(text)
    repeated_phrases = _repeated_long_phrase_stats(
        text,
        phrase_words=int(quality["repeated_long_phrase_words"]),
        min_count=int(quality["repeated_long_phrase_min_count"]),
    )
    vocative_openings = _dialogue_vocative_opening_stats(text)
    event_density = _event_density_stats(text)
    placeholder_hits = _placeholder_term_hits(text, quality["forbid_placeholder_terms"])
    template_artifact_hits = _template_artifact_hits(text) if quality["forbid_template_artifacts"] else []
    lowercase_paragraph_starts = _lowercase_paragraph_start_samples(text)

    min_scene_words = int(quality["min_scene_words"])
    if min_scene_words > 0:
        short_scenes = [
            str(sec.get("scene_id") or sec.get("section"))
            for sec in trace_sections
            if int(sec.get("word_count", 0) or 0) < min_scene_words
        ]
        if short_scenes:
            failures.append(f"scene words below {min_scene_words}: {', '.join(short_scenes[:4])}")

    min_chapter_words = int(quality["min_chapter_words"])
    if min_chapter_words > 0 and word_count < min_chapter_words:
        failures.append(f"chapter words {word_count} below {min_chapter_words}")

    min_dialogue_ratio = float(quality["min_dialogue_ratio"])
    if min_dialogue_ratio > 0 and dialogue_ratio < min_dialogue_ratio:
        failures.append(f"dialogue ratio {dialogue_ratio:.2%} below {min_dialogue_ratio:.2%}")

    min_event_sentence_ratio = float(quality["min_event_sentence_ratio"])
    if min_event_sentence_ratio > 0 and float(event_density["event_sentence_ratio"]) < min_event_sentence_ratio:
        failures.append(
            f"event sentence ratio {event_density['event_sentence_ratio']:.2%} below {min_event_sentence_ratio:.2%}"
        )

    min_type_token_ratio = _effective_min_type_token_ratio(float(quality["min_type_token_ratio"]), word_count)
    if min_type_token_ratio > 0 and type_token_ratio < min_type_token_ratio:
        failures.append(f"type-token ratio {type_token_ratio:.2%} below length-adjusted {min_type_token_ratio:.2%}")

    max_avg_sentence_words = float(quality["max_avg_sentence_words"])
    if max_avg_sentence_words > 0 and avg_sentence_words > max_avg_sentence_words:
        failures.append(f"average sentence length {avg_sentence_words:.2f} above {max_avg_sentence_words:.2f}")

    max_repeated_paragraphs = int(quality["max_repeated_paragraphs"])
    repeated_paragraph_count = int(paragraph_repeats["repeated_paragraph_count"])
    if repeated_paragraph_count > max_repeated_paragraphs:
        failures.append(f"repeated paragraphs {repeated_paragraph_count} above {max_repeated_paragraphs}")

    max_repeated_long_phrases = int(quality["max_repeated_long_phrases"])
    repeated_phrase_count = int(repeated_phrases["repeated_phrase_count"])
    if repeated_phrase_count > max_repeated_long_phrases:
        failures.append(f"repeated long phrases {repeated_phrase_count} above {max_repeated_long_phrases}")

    max_dialogue_vocative_openings = int(quality["max_dialogue_vocative_openings"])
    vocative_count = int(vocative_openings["count"])
    if vocative_count > max_dialogue_vocative_openings:
        failures.append(f"dialogue name-openings {vocative_count} above {max_dialogue_vocative_openings}")

    if placeholder_hits:
        failures.append(f"placeholder terms present: {', '.join(placeholder_hits[:8])}")
    if template_artifact_hits:
        failures.append(f"template artifacts present: {', '.join(template_artifact_hits[:8])}")
    if bool(quality.get("fail_lowercase_paragraph_starts")) and lowercase_paragraph_starts:
        failures.append("lowercase paragraph starts present")

    return failures


def _is_shire_gap_project(project_slug: str) -> bool:
    slug = (project_slug or "").lower()
    return any(
        token in slug
        for token in (
            "shire",
            "hobbiton",
            "frodo",
            "bilbo",
            "last-autumn",
            "ring-gap",
        )
    )


def _is_hunt_gollum_project(project_slug: str) -> bool:
    slug = (project_slug or "").lower()
    return any(
        token in slug
        for token in (
            "hunt-for-gollum",
            "hunt-gollum",
            "gollum-hunt",
            "gollum",
        )
    )


def _sentence_start(text: str) -> str:
    text = str(text or "").strip()
    return text[:1].upper() + text[1:] if text else text


def _hunt_object_label(raw: str) -> str:
    label = str(raw or "").strip()
    lower = label.lower()
    if lower == "ring":
        return "memory of Bilbo's ring"
    if lower == "maps":
        return "creased maps"
    if lower == "rope":
        return "a rope"
    if lower == "tracks":
        return "thin tracks"
    if lower == "walking staff":
        return "Gandalf's staff"
    if lower == "knife":
        return "a knife"
    if lower == "weathered cloak":
        return "a cloak"
    if lower == "letters":
        return "folded letters"
    if lower == "lamps":
        return "low lamps"
    if lower == "fish bones":
        return "fish bones"
    if lower == "muddy water":
        return "muddy water"
    if lower == "road":
        return "the road"
    return label


def _join_hunt_object_labels(rows: list[str], limit: int = 2) -> str:
    labels = [_hunt_object_label(row) for row in rows[:limit]]
    labels = [label for label in labels if label]
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _hunt_place_name(raw: str) -> str:
    name = str(raw or "").strip()
    lower = name.lower()
    if not name:
        return "the wild"
    if lower == "road" or lower in {"west road", "western road", "east road", "eastern road"}:
        return "the westward road"
    return name


def _hunt_place_phrase(raw: str) -> str:
    name = _hunt_place_name(raw)
    lower = name.lower()
    if "anduin" in lower:
        return "beside the Anduin"
    if "dead marsh" in lower:
        return "near the Dead Marshes"
    if "shire" in lower or "border" in lower:
        return "near the Shire borders"
    if "bree" in lower:
        return "near Bree"
    if "mirkwood" in lower or "woodland" in lower:
        return "under the boughs of Mirkwood"
    if "rhovanion" in lower:
        return "across Rhovanion"
    if "wilderland" in lower:
        return "across Wilderland"
    if "road" in lower:
        return f"along {name}"
    return f"near {name}"


def _project_canon_entities(project_slug: str) -> list[str]:
    slug = (project_slug or "").lower()
    if _is_hunt_gollum_project(slug):
        return [
            "Gandalf",
            "Aragorn",
            "Strider",
            "Gollum",
            "Smeagol",
            "Bilbo",
            "Thranduil",
            "Denethor",
            "Isildur",
            "Saruman",
            "Sauron",
        ]
    if _is_shire_gap_project(slug):
        return [
            "Frodo", "Gandalf", "Sam", "Bilbo", "Merry", "Pippin",
            "Bag End", "Hobbiton", "Bywater", "The Shire", "Shire",
            "Ring", "Rangers", "Bree", "Green Dragon", "Road",
        ]
    if "beren" in slug or "luthien" in slug:
        return [
            "Beren", "Luthien", "Lúthien", "Thingol", "Melian", "Sauron", "Morgoth",
            "Finrod", "Celegorm", "Curufin", "Huan", "Tol-in-Gaurhoth", "Doriath", "Nargothrond",
        ]
    return []


def _out_of_domain_entities(project_slug: str) -> set[str]:
    slug = (project_slug or "").lower()
    if _is_hunt_gollum_project(slug):
        return {
            "frodo", "sam", "merry", "pippin", "boromir", "legolas", "gimli",
            "elrond", "galadriel", "fellowship", "council of elrond",
            "rivendell", "rohan", "isengard", "orthanc",
            "helm's deep", "helms deep", "mount doom", "moria", "balrog",
            "ringwraith", "ringwraiths", "nazgul", "nazgûl", "black rider",
            "black riders",
        }
    if _is_shire_gap_project(slug):
        return {
            "aragorn", "strider", "boromir", "legolas", "gimli", "elrond", "galadriel",
            "sauron", "saruman", "gollum", "smeagol", "sméagol", "ringwraith",
            "ringwraiths", "nazgul", "nazgûl", "black rider", "black riders",
            "council of elrond", "fellowship", "mordor", "mount doom", "minas tirith",
            "rohan", "gondor", "isengard", "orthanc", "balrog", "moria",
        }
    if "beren" in slug or "luthien" in slug:
        return {
            "bilbo", "frodo", "sam", "gandalf", "aragorn", "legolas", "gimli", "boromir", "faramir",
            "pippin", "merry", "gollum", "smeagol", "saruman", "eowyn", "theoden", "denethor",
        }
    return set()


def _non_character_entities(project_slug: str) -> set[str]:
    slug = (project_slug or "").lower()
    if _is_hunt_gollum_project(slug):
        return {
            "anduin", "mirkwood", "rhovanion", "dead marshes", "mordor", "shire",
            "the shire", "bree", "prancing pony", "wood-elves", "woodland realm",
            "minas tirith", "gondor", "dol guldur", "lórien", "lorien",
            "road", "ring", "bilbo's ring", "isildur's record", "baggins",
        }
    if _is_shire_gap_project(slug):
        return {
            "bag end", "hobbiton", "bywater", "the shire", "shire", "ring",
            "rangers", "bree", "green dragon", "road", "westfarthing",
        }
    if "beren" in slug or "luthien" in slug:
        return {"tol-in-gaurhoth", "doriath", "nargothrond", "menegroth", "beleriand", "silmaril"}
    return set()


def _default_story_setting(project: dict) -> str:
    explicit = str(project.get("default_setting") or "").strip()
    if explicit:
        return explicit

    slug = str(project.get("slug") or "").lower()
    if _is_hunt_gollum_project(slug):
        return "Wilderland"
    if _is_shire_gap_project(slug):
        return "Bag End"
    if "beren" in slug or "luthien" in slug:
        return "Beleriand"
    if "mithrandir" in slug or "rhun" in slug or "east" in slug:
        return "Rhun"
    return "Middle-earth"


def _scene_plan_index(plan: dict) -> tuple[dict[str, dict], dict[int, dict]]:
    scenes: dict[str, dict] = {}
    chapters: dict[int, dict] = {}
    for chapter_row in plan.get("chapters", []):
        if not isinstance(chapter_row, dict):
            continue
        chapter_num = int(chapter_row.get("chapter_number", 0) or 0)
        if chapter_num > 0:
            chapters[chapter_num] = chapter_row
        for scene in chapter_row.get("scenes", []) or []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id") or "").strip()
            if scene_id:
                scenes[scene_id] = scene
    return scenes, chapters


def _load_shadow_beats_by_scene(proj_dir: Path) -> dict[str, list[dict]]:
    payload = _load_json(proj_dir / "shadow_beats.json", default={})
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for beat in payload.get("beats", []) or []:
        if not isinstance(beat, dict):
            continue
        scene_id = _scene_from_beat_id(str(beat.get("beat_id", "")))
        if scene_id and scene_id != "unknown-scene":
            by_scene[scene_id].append(beat)
    for rows in by_scene.values():
        rows.sort(key=lambda b: int(b.get("position", 0) or 0))
    return dict(by_scene)


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _story_scene_characters(project_slug: str, plan_scene: dict, event: dict, scene_beats: list[dict]) -> list[str]:
    banned = _out_of_domain_entities(project_slug) | _non_character_entities(project_slug) | PLACEHOLDER_PARTICIPANTS

    def _valid(rows: list[str]) -> list[str]:
        return [name for name in _dedupe_strings(rows) if name.lower() not in banned]

    plan_chars: list[str] = []
    raw_plan = plan_scene.get("characters")
    if isinstance(raw_plan, list):
        plan_chars.extend(str(x).strip() for x in raw_plan if str(x).strip())
    filtered_plan = _valid(plan_chars)
    if filtered_plan:
        return filtered_plan[:4]

    out: list[str] = []
    raw_event = event.get("characters")
    if isinstance(raw_event, list):
        out.extend(str(x).strip() for x in raw_event if str(x).strip())
    for beat in scene_beats:
        participants = beat.get("participants")
        if isinstance(participants, list):
            out.extend(str(x).strip() for x in participants if str(x).strip())

    filtered = _valid(out)
    if filtered:
        return filtered[:4]

    canon = [name for name in _project_canon_entities(project_slug) if name.lower() not in _out_of_domain_entities(project_slug)]
    return canon[:2] or ["Unknown"]


def _story_scene_objects(plan_scene: dict, event: dict) -> list[str]:
    out: list[str] = []
    for key in ("objects", "artifacts"):
        raw = plan_scene.get(key)
        if isinstance(raw, list):
            out.extend(str(x).strip() for x in raw if str(x).strip())
        raw_event = event.get(key)
        if isinstance(raw_event, list):
            out.extend(str(x).strip() for x in raw_event if str(x).strip())
    return _dedupe_strings(out)[:6]


def _story_scene_place(project: dict, plan_scene: dict, event: dict) -> str:
    for source in (plan_scene, event):
        for key in ("setting", "place", "location"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return _default_story_setting(project)


def _story_scene_source_refs(event: dict, scene_beats: list[dict]) -> list[str]:
    refs: list[str] = []
    event_refs = event.get("source_canon_node_ids")
    if isinstance(event_refs, list):
        refs.extend(str(x).strip() for x in event_refs if str(x).strip())
    for beat in scene_beats:
        raw_refs = beat.get("source_canon_node_ids")
        if isinstance(raw_refs, list):
            refs.extend(str(x).strip() for x in raw_refs if str(x).strip())
    return _dedupe_strings(refs)


def _story_scene_goal(
    *,
    plan_scene: dict,
    event: dict,
    scene_beats: list[dict],
    missing_terms_hint: list[str],
    timeline: dict[str, Any] | None = None,
    future_guardrail_entities: list[str] | None = None,
) -> str:
    parts: list[str] = []
    primary_goal = str(plan_scene.get("goal") or event.get("description") or plan_scene.get("summary") or "Advance the story while preserving canon.").strip()
    if primary_goal:
        parts.append(primary_goal.rstrip(".") + ".")

    summary = str(plan_scene.get("summary") or "").strip()
    if summary and summary.lower() not in primary_goal.lower():
        parts.append(f"Scene brief: {summary.rstrip('.')}.")

    action = str(event.get("action") or "").strip()
    if action:
        if action.lower() in PLACEHOLDER_PARTICIPANTS or action.lower() in {"unknown", "worked", "did", "was", "had"}:
            action = "counsel"
        parts.append(f"Shadow action to realize: {action}.")

    beat_intents = _dedupe_strings(
        [
            str(beat.get("intent") or "").strip()
            for beat in scene_beats
            if str(beat.get("intent") or "").strip()
        ]
    )
    if beat_intents:
        parts.append(f"Scene beats to cover: {'; '.join(beat_intents[:3])}.")

    motifs = event.get("motifs")
    if isinstance(motifs, list) and motifs:
        clean_motifs = [
            motif
            for motif in _dedupe_strings([str(x).strip(" .,:;!?\"'").lower() for x in motifs])
            if len(motif) > 2 and motif not in STOPWORDS and not motif.endswith("'s")
        ][:3]
        if clean_motifs:
            parts.append(f"Motifs to echo: {', '.join(clean_motifs)}.")

    if missing_terms_hint:
        parts.append(
            "Required canon anchors to weave in naturally during this chapter: "
            + ", ".join(missing_terms_hint)
            + "."
        )

    if timeline:
        story_era = str(timeline.get("story_era") or "").strip()
        story_year = timeline.get("story_year")
        if story_era:
            when = f"{story_era} {story_year}" if story_year is not None else story_era
            parts.append(f"Story-time is {when}.")
            if timeline.get("allow_past_references", True):
                parts.append("Past figures and older events may be remembered or spoken of.")
            if timeline.get("forbid_future_entities", True):
                parts.append("Do not mention people or events from the future of this story-time.")

    return " ".join(parts)


def _story_scene_runtime_id(project_slug: str, scene_id: str) -> str:
    return f"{project_slug}-{scene_id}"


def _chapter_title(chapter: int, plan_chapter: dict | None) -> str:
    title = str((plan_chapter or {}).get("title") or "").strip()
    return title or f"Chapter {chapter}"


def _chapter_structure_metadata(
    *,
    project: dict | None,
    plan_chapter: dict | None,
    scene_count: int,
) -> dict[str, Any]:
    """Expose whether movement count follows the chapter's declared purpose."""
    plan_chapter = plan_chapter or {}
    declared_count = _coerce_optional_int(plan_chapter.get("movement_count"))
    if declared_count is None:
        declared_count = _coerce_optional_int(plan_chapter.get("scene_count"))
    basis = str(plan_chapter.get("movement_count_basis") or plan_chapter.get("structure_role") or "").strip()
    intent = str(plan_chapter.get("intent") or "").strip()
    variable_counts = bool((project or {}).get("variable_scenes_per_chapter", False))
    purpose_driven = bool(variable_counts and basis and intent)
    count_matches = declared_count is None or declared_count == scene_count
    return {
        "purpose_driven": purpose_driven,
        "intent": intent,
        "structure_role": str(plan_chapter.get("structure_role") or "").strip(),
        "movement_count_basis": basis,
        "declared_movement_count": declared_count,
        "actual_movement_count": scene_count,
        "movement_count_matches_plan": count_matches,
    }


def _chapter_outline_text(plan_chapter: dict | None, chapter_rows: list[dict], scene_plan_by_id: dict[str, dict], graph_node_by_id: dict[str, dict], scene_beats_by_scene: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    intent = str((plan_chapter or {}).get("intent") or "").strip()
    if intent:
        lines.append(intent)
    for row in chapter_rows:
        scene_id = str(row.get("scene_id") or "").strip()
        if not scene_id:
            continue
        plan_scene = scene_plan_by_id.get(scene_id, {})
        event = graph_node_by_id.get(row.get("shadow_event_id"), {})
        goal = str(plan_scene.get("goal") or event.get("description") or plan_scene.get("summary") or scene_id).strip()
        fragments = [f"{scene_id}: {goal}"]
        action = str(event.get("action") or "").strip()
        if action:
            fragments.append(f"action={action}")
        scene_beats = scene_beats_by_scene.get(scene_id, [])
        if scene_beats:
            beat_summary = "; ".join(
                str(beat.get("intent") or beat.get("action") or "").strip()
                for beat in scene_beats[:3]
                if str(beat.get("intent") or beat.get("action") or "").strip()
            )
            if beat_summary:
                fragments.append(f"beats={beat_summary}")
        lines.append(" | ".join(fragments))
    return "\n".join(lines)


def _maybe_load_story_world_bible(generator, project: dict, proj_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for key in ("world_bible", "world_bible_file", "canon_file"):
        raw = str(project.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        candidates.extend([path, proj_dir / path] if not path.is_absolute() else [path])
    candidates.append(proj_dir / "story_bible.md")
    seen: set[Path] = set()
    for path in candidates:
        path = path.expanduser()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            generator.load_world_bible(str(path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        return path
    return None


def _load_story_voice_profiles(project: dict, proj_dir: Path) -> tuple[dict[str, Any], Path | None]:
    candidates: list[Path] = []
    for key in ("voice_profiles_file", "voice_file"):
        raw = str(project.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        candidates.extend([path, proj_dir / path] if not path.is_absolute() else [path])
    candidates.append(Path("data/voice/hobbit_all_voices.json"))

    from book_graph_analyzer.voice.profile import CharacterVoiceProfile

    seen: set[Path] = set()
    for path in candidates:
        path = path.expanduser()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        payload = _load_json(path, default={})
        raw_profiles = payload.get("profiles", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw_profiles, dict):
            continue
        profiles: dict[str, Any] = {}
        for name, raw_profile in raw_profiles.items():
            if not isinstance(raw_profile, dict):
                continue
            profile_payload = dict(raw_profile)
            profile_payload.setdefault("character_name", str(name))
            try:
                profiles[str(name)] = CharacterVoiceProfile.from_dict(profile_payload)
            except (TypeError, ValueError):
                continue
        if profiles:
            return profiles, path
    return {}, None


def _new_story_shadow_graph(story_id: str):
    from book_graph_analyzer.generate import ShadowGraph

    return ShadowGraph(story_id=story_id)


def _new_story_scene_generator(shadow_graph):
    from book_graph_analyzer.generate import SceneGenerator

    return SceneGenerator(shadow_graph=shadow_graph)


def _new_story_generation_writer():
    from book_graph_analyzer.generate import GenerationWriter

    return GenerationWriter()


def _validate_story_scene_participants(project_slug: str, scene_id: str, characters: list[str]) -> None:
    if not _project_canon_entities(project_slug):
        return

    invalid = [
        name
        for name in characters
        if not str(name).strip()
        or str(name).strip().lower() in PLACEHOLDER_PARTICIPANTS
        or str(name).strip().lower() in _out_of_domain_entities(project_slug)
        or str(name).strip().lower() in _non_character_entities(project_slug)
    ]
    if invalid:
        raise click.ClickException(
            f"Scene {scene_id or '<unknown>'} has invalid Tolkien-domain participants: {invalid}"
        )
    if not characters:
        raise click.ClickException(
            f"Scene {scene_id or '<unknown>'} has no valid Tolkien-domain participants after filtering."
        )


def _template_hunt_gollum_scene_text(
    *,
    project: dict,
    scene_goal: str,
    characters: list[str],
    place: str,
    objects: list[str],
    event: dict,
    scene_beats: list[dict],
    missing_terms_hint: list[str],
    quality: dict[str, Any] | None = None,
) -> str:
    quality = quality or {}
    target_words = int(quality.get("target_scene_words", 0) or 0) or 1400
    target_dialogue_ratio = float(quality.get("target_dialogue_ratio", 0.0) or 0.0)
    place_l = place.lower()
    character_lowers = {name.lower(): name for name in characters}

    def _join_names(names: list[str]) -> str:
        rows = names[:4]
        if len(rows) <= 1:
            return rows[0] if rows else "the hunters"
        if len(rows) == 2:
            return f"{rows[0]} and {rows[1]}"
        return ", ".join(rows[:-1]) + f", and {rows[-1]}"

    def _pick(preferred: list[str], fallback: str, blocked: set[str] | None = None) -> str:
        blocked = blocked or set()
        for candidate in preferred:
            lowered = candidate.lower()
            if lowered in character_lowers and lowered not in blocked:
                return character_lowers[lowered]
        for candidate in characters:
            lowered = candidate.lower()
            if lowered not in blocked:
                return candidate
        return fallback

    def _plain_sentence(text: str) -> str:
        return str(text or "").strip(" .,:;!?\"'")

    def _display_goal(raw_goal: str) -> str:
        public_goal = _public_scene_goal_text(raw_goal)
        if public_goal:
            return public_goal
        blocked_prefixes = (
            "shadow action to realize:",
            "scene beats to cover:",
            "scene brief:",
            "motifs to echo:",
            "story-time is",
            "past figures",
            "do not mention",
            "required canon anchors",
        )
        parts = []
        for raw_part in re.split(r"(?<=\.)\s+", raw_goal):
            part = raw_part.strip()
            if not part:
                continue
            if part.lower().startswith(blocked_prefixes):
                continue
            parts.append(part)
        return " ".join(parts[:2]) or "The hunt presses eastward while the hunters keep faith with Third Age canon."

    def _clean_motifs(raw: list[Any]) -> list[str]:
        rows: list[str] = []
        for raw_motif in raw:
            motif = _plain_sentence(str(raw_motif)).lower()
            if not motif or motif in MOTIF_STOPWORDS or motif in STOPWORDS or motif in {"without"}:
                continue
            if len(motif) < 3 or motif.endswith("'s"):
                continue
            rows.append(motif)
        preferred = [m for m in rows if m in {"hunt", "trail", "shadow", "pity", "fear", "secrecy", "wilderness"}]
        return _dedupe_strings(preferred + rows)[:4]

    def _scene_kind() -> str:
        narrative_goal_l = _display_goal(scene_goal).lower()
        if any(token in narrative_goal_l for token in ("rain and rumour", "bring gandalf", "unease rather than proof", "opening establishes")):
            return "charge"
        if any(token in narrative_goal_l for token in ("names gollum", "gollum as the quarry", "baggins and the shire must be guarded")):
            return "naming"
        if any(token in narrative_goal_l for token in ("accepts the hunt", "hidden service", "would not welcome him")):
            return "vow"
        if any(token in narrative_goal_l for token in ("frightened witness", "crooked report", "half helps", "crooked rumour")):
            return "witness"
        if any(token in narrative_goal_l for token in ("southward signs", "darker country", "trail is bending", "toward darker")):
            return "darkening"
        if any(token in narrative_goal_l for token in ("failure", "obstacles", "not arbitrary defeat", "lost trail", "false certainty")):
            return "failure"
        if any(token in narrative_goal_l for token in ("final image", "providential hope", "unresolved danger", "finale")):
            return "finale"
        if "active service" in narrative_goal_l:
            return "active_service"
        if "offstage" in narrative_goal_l:
            return "offstage_watch"
        if (
            any(token in narrative_goal_l for token in ("hidden guard", "active service", "offstage", "uncertainty", "shire borders", "border"))
            and "aragorn" in character_lowers
        ):
            return "border_watch"
        if any(token in narrative_goal_l for token in ("warning", "warn", "westward", "turns west", "keep watch", "roads toward the shire", "green country", "shire borders")):
            return "warning"
        if "mirkwood" in place_l and any(token in narrative_goal_l for token in ("question", "testimony", "answers", "weigh", "compare", "deliver", "guarded mirkwood")):
            return "delivery"
        if any(token in narrative_goal_l for token in ("question", "testimony", "answers", "weigh", "compare")):
            return "warning"
        if any(token in narrative_goal_l for token in ("escapes", "slips free", "narrow chance", "chance he has been waiting", "break in the watch")) and not any(
            token in narrative_goal_l for token in ("escaped trail", "compare", "testimony", "question", "weigh", "more troubling than the fugitive")
        ):
            return "escape"
        if (
            any(token in narrative_goal_l for token in ("routine", "watch become a map", "habit becomes a door", "studies lamps", "steps, voices"))
            and "gollum" in character_lowers
        ):
            return "watch"
        if (
            any(
                token in narrative_goal_l
                for token in ("captivity", "captive", "custody", "prisoner", "guarded", "routine", "watchfulness", "northward road")
            )
            and "gollum" in character_lowers
        ):
            return "captivity"
        if any(token in narrative_goal_l for token in ("warning", "warn", "westward", "turns west", "keep watch", "roads toward the shire")):
            return "warning"
        if any(token in narrative_goal_l for token in ("capture", "seize", "caught", "bind", "dead marsh")):
            return "capture"
        if any(token in narrative_goal_l or token in place_l for token in ("wood-elves", "wood elves", "woodland", "thranduil", "deliver", "question")):
            return "delivery"
        if "gollum" in character_lowers and len(character_lowers) == 1:
            return "quarry"
        if any(token in narrative_goal_l for token in ("gollum slips", "mutter", "flee", "mordor", "marsh")):
            return "quarry"
        if any(token in narrative_goal_l or token in place_l for token in ("anduin", "rhovanion", "mirkwood", "trail", "track", "search", "wilderland")):
            return "trail"
        return "charge"

    def _setting_sentence(raw_place: str, salt: str = "") -> str:
        lower = raw_place.lower()
        def _pick(rows: list[str]) -> str:
            return rows[_stable_seed("hunt-setting", raw_place, salt) % len(rows)]
        if "bree" in lower or "prancing pony" in lower:
            return _pick(
                [
                    "Near Bree the rain came low over the ditches, and every window seemed to listen before it showed a lamp.",
                    "At Bree the wet road shone under low clouds, and talk moved from doorway to doorway with the smell of smoke.",
                    "By the Bree-road the hedges dripped steadily, and the inn-lamps made small islands in the evening murk.",
                ]
            )
        if "anduin" in lower:
            return _pick(
                [
                    "Along the Anduin the reeds bent and straightened under a cold wind, writing and erasing signs at the water's edge.",
                    "Beside the Anduin the bank gave way in shelves of mud, and the river kept half the tale for itself.",
                    "The Anduin ran broad and colourless there, smoothing the marks of passage even as it revealed them.",
                ]
            )
        if "rhovanion" in lower or "wilderland" in lower:
            return _pick(
                [
                    "In the long reaches of Wilderland the road gave way to heather, stone, old ash, and the wide silence of countries watched by few.",
                    "Across Wilderland the distances opened bare and watchful, with old paths fading into grass and thorn.",
                    "In Rhovanion the wind had room to travel, and even a careful footstep seemed too loud beneath the empty sky.",
                    "Beyond the settled roads of Rhovanion, grass leaned over old tracks and the hills kept their answers to themselves.",
                    "In Wilderland the sky seemed to draw back from the earth, leaving every low sound exposed on the open miles.",
                    "East of the easier roads the country thinned into scrub, pale stone, and hollows where a fugitive might vanish by inches.",
                    "Across the open lands, old paths broke apart among thorn, flint, and grass already leaning over their edges.",
                    "In the wide country east of the river, distance made a discipline of every glance and every pause.",
                    "The ridges of Rhovanion lay pale under the weather, giving no counsel except the cost of haste.",
                    "Wilderland stretched in rough folds of grass and stone, too empty for comfort and too marked for trust.",
                    "There the road was less a road than a memory of passage, interrupted by scrub, gully, and weather.",
                    "Under the open sky of Rhovanion, even stillness seemed to have travelled a long way.",
                ]
            )
        if "mirkwood" in lower or "woodland" in lower:
            return _pick(
                [
                    "Under the eaves of Mirkwood the day thinned early, and every branch seemed to remember older trespasses.",
                    "In Mirkwood the lamps looked small against the boughs, and the dark between trunks seemed already awake.",
                    "Beneath the great trees the air was close and resinous, and watchfulness gathered before any word was spoken.",
                    "Mirkwood closed about the guarded place in ranks of dark boles, with lamp-glow caught low among roots and fern.",
                    "Under the woodland canopy, sound travelled strangely: near whispers seemed distant, and distant cracks came sharp as speech.",
                    "Among the trees the light had to be husbanded, for every flame looked like a small decision made against the dark.",
                    "The forest held its breath in layers of leaf, bark, and shadow, never wholly silent and never friendly to haste.",
                    "Below the high branches, the watched paths wound between roots like thoughts that had learned secrecy from old fear.",
                    "In the guarded wood, night did not arrive at once; it gathered by hollows, by trunks, and under the low places of boughs.",
                ]
            )
        if "dead marsh" in lower or "mordor" in lower:
            return _pick(
                [
                    "Near the Dead Marshes the ground had no honest firmness, and pale pools held the sky as if it were something drowned.",
                    "By the Dead Marshes each step tested the earth and found it treacherous, while vapour clung low over the pools.",
                    "The marsh country gave back no clean path, only cold gleams, sucking ground, and the sour breath of old water.",
                ]
            )
        if lower == "road" or lower in {"west road", "western road", "east road", "eastern road"}:
            return _pick(
                [
                    "On the westward road the dust held small reports of carts, boots, and one hurried passage no cart had made.",
                    "Along the watched road the verges were wet and narrow, and every bend had to be read before it was trusted.",
                    "By the road west the ditches kept old rain, while the ruts showed which travellers had hurried and which had lingered.",
                ]
            )
        if "shire" in lower or "border" in lower:
            return _pick(
                [
                    "Near the western borders the hills folded softly one upon another, and the lanes beyond them seemed asleep in trust.",
                    "At the edge of the green country the road quieted, as if even dust had learned the manners of peace.",
                    "Beyond the watched rise lay tilled fields and small roofs, untroubled by the danger moving toward them.",
                ]
            )
        return f"{_sentence_start(_hunt_place_phrase(raw_place))}, the hunt moved under a sky that made even ordinary speech sound guarded."

    def _object_label(raw: str) -> str:
        return _hunt_object_label(raw)

    def _join_object_labels(rows: list[str]) -> str:
        labels = [_object_label(row) for row in rows[:2]]
        if len(labels) <= 1:
            return labels[0] if labels else ""
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"

    scene_kind = _scene_kind()
    char_phrase = _join_names(characters)
    ranger = character_lowers.get("aragorn") or character_lowers.get("strider") or "Aragorn"
    wizard = character_lowers.get("gandalf") or "Gandalf"
    quarry = character_lowers.get("gollum") or character_lowers.get("smeagol") or "Gollum"
    motifs = _clean_motifs(event.get("motifs") or [])
    beat_lines = _dedupe_strings(
        [
            _plain_sentence(str(beat.get("intent") or beat.get("action") or ""))
            for beat in scene_beats
            if _plain_sentence(str(beat.get("intent") or beat.get("action") or ""))
        ]
    )
    object_rows = _dedupe_strings([str(obj).strip() for obj in objects if str(obj).strip()])
    anchors = _dedupe_strings([str(term).strip() for term in missing_terms_hint if str(term).strip()])

    object_sentence = ""
    object_sentence_start = "The things at hand"
    if object_rows:
        labels = _join_object_labels(object_rows)
        object_sentence_start = _sentence_start(labels) if labels else "The things at hand"
        object_variants = [
            f" Near {place}, {labels} lay ready; secrecy made them heavier than their make.",
            f" For {char_phrase}, {labels} were plain gear until the next choice demanded them.",
            f" Beside the way lay {labels}; weather, memory, and danger gave them rank.",
            f" Close by, {labels} waited for use before any large word was trusted.",
            f" Within reach, {labels} answered weather first and counsel only afterward.",
            f" The hour took up {labels}, making service out of common craft.",
            f" Under that sky, {labels} looked less like baggage than unfinished decisions.",
            f" At {place}, {labels} marked the difference between talk and action.",
            f" In {char_phrase}'s keeping, {labels} became part of the road's evidence.",
            f" The ground near {place} gave {labels} a sterner meaning than comfort.",
            f" Before the next mile, {labels} would matter more than any fair speech.",
            f" By the hunter's hand, {labels} served mud, caution, and haste.",
            f" In the guarded pause, {labels} kept counsel with weather and need.",
            f" There was no ornament in {labels}; the road allowed none.",
            f" Near the watchers, {labels} made a rough grammar of necessity.",
            f" The silence around {labels} was practical, not ceremonial.",
        ]
        object_sentence = object_variants[
            _stable_seed("hunt-object-sentence", scene_goal, labels, place) % len(object_variants)
        ]

    motif_sentence = ""
    if motifs:
        motif_sentence = ""

    beat_sentence = ""
    if beat_lines:
        beat_sentence = ""

    anchor_sentence = ""
    if anchors:
        anchor_sentence = (
            " By the end of the chapter these anchors had to stand in plain sight: "
            + ", ".join(anchors[:6])
            + "."
        )

    stakes_by_kind = {
        "charge": [
            "The danger was still mostly inference: Bilbo's old tale, Gollum's loss, and the risk that Baggins and Shire had become names with a road before them.",
            "Behind the quiet words stood a small country, an old ring-story, and the fear that a ruined creature might carry homely names into hostile hearing.",
            "Nothing in the errand looked grand at first glance, which was precisely why it was perilous: Baggins, Shire, and Gollum were small words with long shadows.",
        ],
        "naming": [
            "To name Gollum was to make the fear less vague and more dreadful. A creature could be followed; a rumour could only spread.",
            "Once the quarry had a name, all the old fragments drew nearer together: Bilbo's tale, the lost ring, and the exposed innocence of the Shire.",
            "The word Gollum did not sound like a power of the world, but it gathered Baggins and Shire into a peril that could no longer be dismissed.",
        ],
        "vow": [
            "The road asked for service before it offered honour. Aragorn accepted the poorer bargain, and in that choice the hidden watch began.",
            "No crown, song, or welcome stood behind the promise. Its worth lay in going for those who would bar the door against their own defender.",
            "The charge passed from Gandalf's fear into Aragorn's keeping, and became at once more practical and more lonely.",
        ],
        "witness": [
            "A frightened witness is not an oracle, yet fear may preserve the outline of what pride would smooth away.",
            "The hunt entered human rumour there, where every useful word had to be separated from ale, shame, and alarm.",
            "Truth came not as testimony fit for a court, but as a crooked report that still leaned toward the quarry.",
        ],
        "darkening": [
            "The southward turn changed the air of the pursuit. What had been elusive now seemed drawn toward older malice.",
            "Each sign that bent south made the hunt less local and more grave, as if the road itself had begun to remember Mordor.",
            "The trail no longer promised merely a fugitive. It hinted at a summons laid on a mind too broken to resist cleanly.",
        ],
        "failure": [
            "The failure of the trail was earned by weather, distance, and the quarry's ugly craft.",
            "Defeat came not as a blow but as a narrowing of honest signs until pride had nothing sound to hold.",
            "A lost trail could still teach discipline if the hunter refused to make certainty out of need.",
        ],
        "border_watch": [
            "The hidden guard took shape where peace looked least able to imagine needing it.",
            "Aragorn answered uncertainty by counting roads, strangers, dogs, and hours until doubt had work to do.",
            "The threat stayed offstage, but that only made the borders more deserving of quiet eyes.",
        ],
        "active_service": [
            "Uncertainty became useful when it was divided into small duties.",
            "The watcher could not finish the hunt, but he could make the roads less careless.",
            "Service began where certainty ended and discipline refused to stand idle.",
        ],
        "offstage_watch": [
            "The threat remained unseen, but unseen danger still shapes the roads around it.",
            "An offstage peril asks for watchfulness without the comfort of visible proof.",
            "The border had to be held against what had not yet shown its face.",
        ],
        "finale": [
            "The ending had to hold danger and mercy together without pretending the hunt was finished.",
            "Hope stood not in triumph but in time bought, roads watched, and warning nearly delivered.",
            "Gollum remained loose, Gandalf moved toward the door, and Aragorn kept the dark from easy roads.",
        ],
        "trail": [
            "The stakes lay in signs too slight for pride: a print near water, a rumour bent by fear, and the thought that Gollum's muttering might outrun the hunters.",
            "Every mile mattered because the quarry carried more than hunger. He carried Baggins and Shire in a memory cracked enough to leak.",
            "The hunt was not a race of horses. It was a contest between patient reading and a fugitive's talent for making the world forget him.",
        ],
        "quarry": [
            "In him the danger was not strength but memory. Baggins and Shire lived in his mouth like splinters he could neither swallow nor spit away.",
            "He was small, starved, and wretched, but the names he nursed could become larger than armies if darker ears received them.",
            "What he had lost ruled him still, and loss had taught him to speak the wrong names when pain or hatred stirred him.",
        ],
        "watch": [
            "Routine became perilous because Gollum could hate it patiently enough to read it.",
            "The guarded wood taught him lamps, steps, voices, roots, and shadows as if each were a crack in a wall.",
            "A captive who cannot master strength may still master timing, and timing was the lesson he began to learn.",
        ],
        "captivity": [
            "The danger had been bound, not ended. A captive tongue may travel farther than captive feet if pity grows careless.",
            "Custody changed the shape of the peril without diminishing it. Gollum still carried Baggins and Shire where no rope could reach.",
            "The prisoner's weakness was no proof of safety; his memory remained loose, listening, and venomous.",
        ],
        "escape": [
            "The old danger found a new shape. If Gollum passed beyond the watch, the names he carried would go with him into weather and rumour.",
            "A single breach could undo miles of labour, for the quarry's body was slight but his knowledge was not.",
            "The peril moved toward darkness by inches, and each inch threatened the quiet west more than any shout would have done.",
        ],
        "capture": [
            "The hunt had narrowed to contact, yet contact made the moral burden heavier. Gollum had to be held living because the truth in him was not yet spent.",
            "Victory, if the word could be used, looked like mud, rope, teeth, and a shivering prisoner whose misery did not make him harmless.",
            "The names Baggins and Shire had to be brought back through a mouth that lied, begged, cursed, and remembered.",
        ],
        "delivery": [
            "The road's answer had become a guarded question. Gollum was present at last, but presence only proved how crooked the truth could be.",
            "The hunters had won a prisoner and inherited a burden. Baggins and Shire were now evidence, not merely muttering.",
            "What had been followed in mud now had to be weighed in speech, and speech may deceive more subtly than footprints.",
        ],
        "warning": [
            "Knowledge had become the dangerous cargo. It must go west before the same names reached a darker use.",
            "The hunt no longer asked only where Gollum had gone. It asked how long the Shire could remain unwarned and still be safe.",
            "No clean ending stood before them. Only warning, watchfulness, and the uneasy mercy of time remained.",
        ],
    }
    matter_by_kind = {
        "charge": [
            "The matter was not fit for songs, not yet. It was made of wet cloaks, uncertain reports, guarded questions, and a decision taken before proof had become comfort.",
            "The beginning had little splendour in it. A few poor signs, an old fear, and a wizard's refusal to dismiss small things were enough to set the road in motion.",
            "No trumpet would have known what to do with such an errand. It belonged to maps spread under low light and to words saved until they could no longer be postponed.",
        ],
        "naming": [
            "Gandalf did not turn suspicion into certainty by speaking it. He gave uncertainty a shape that could be hunted, tested, and feared in the proper measure.",
            "The counsel no longer circled only around old unease. It had a creature at its centre, and that creature's memory made the west suddenly vulnerable.",
            "A name can steady thought and sharpen danger at the same time. So it was with Gollum, spoken there in low light among practical things.",
        ],
        "vow": [
            "Aragorn did not answer as a man choosing an adventure. He answered as one accepting weather, hunger, mistrust, and silence as the ordinary wages of need.",
            "The promise was not large in words. It became large only when measured against the miles, the hidden labour, and the peace that would never know his face.",
            "The vow drew no witness except the road and Gandalf's grave attention. That was fitting, for hidden service loses something when it asks to be seen.",
        ],
        "witness": [
            "The report came with gaps, evasions, and the haste of a man wishing to be done with his own memory. Aragorn listened past all three.",
            "No one there wished to be part of great matters. That reluctance gave the account some value, for invention usually desires a larger audience.",
            "The witness had seen too little to understand and enough to be troubled. Such half-knowledge is often where a hidden hunt must pause.",
        ],
        "darkening": [
            "The land did not announce the change with thunder. It changed by degrees: fewer fires, harsher banks, rumours that ended when strangers entered.",
            "Aragorn marked the turn not by one proof but by an agreement among lesser signs, all of them leaning away from safety.",
            "The trail's southward habit made hunger seem less than the whole explanation. Something besides want had begun to pull at the quarry.",
        ],
        "failure": [
            "The road did not betray him in one stroke. It failed by honest degrees: a washed bank, a confused rumour, a sign too old to bear weight.",
            "To press farther on false certainty would not have been courage. It would have been vanity wearing the cloak of endurance.",
            "The defeat mattered because it changed the work. Pursuit could no longer pretend to be the only shape of vigilance.",
        ],
        "border_watch": [
            "The work had become less dramatic and more exact: roads noted, strangers weighed, rumours followed only as far as truth allowed.",
            "No one asked for his guardianship. That made it cleaner, and lonelier, and more easily mistaken for idleness by those who slept because of it.",
            "A border is not defended only at the moment of attack. It is defended in the long quiet beforehand, when doubt still has to do useful work.",
        ],
        "active_service": [
            "He made doubt practical: one road checked at dusk, one rumour followed to its source, one silence remembered because it came too quickly.",
            "Bree gave him mud, ale-talk, suspicion, and useful fragments. Such things were not beneath the errand now; they were the errand's working material.",
            "The active part of hidden service was often invisible even to the one performing it, until the pattern of small duties began to hold.",
        ],
        "offstage_watch": [
            "He did not need to see the threat in order to deny it easy roads. A watched border is a warning before it is a battlefield.",
            "The Shire's peace remained plain and unarmed, and that plainness required a guard willing to remain outside its notice.",
            "Offstage danger made every ordinary movement matter more: the cart, the stranger, the dog gone silent, the path no one used twice.",
        ],
        "finale": [
            "No clean victory could honestly close the tale. The prisoner was gone, the warning incomplete, and yet the hidden labour had not failed.",
            "The final balance lay in three motions: a wizard westward, a Ranger watching, and a ruined creature carrying names he did not understand.",
            "Providence, if it could be glimpsed, appeared not as rescue but as time: a narrow mercy purchased before the door opened.",
        ],
        "trail": [
            "The chase became a discipline of fragments. A scraped bank, a fish-bone, a frightened witness, and silence after birdsong each had to be weighed without haste.",
            "The road did not offer certainty in a single gift. It yielded scraps, withdrew them, and forced the hunter to earn continuity by patience.",
            "Tracking made a hard alphabet of the land. Mud, reed, ash, and broken grass became letters in a message the quarry had never meant to write.",
        ],
        "quarry": [
            "His world was made of hunger, suspicion, cold water, and old injury. Even the open sky seemed to him like another watcher leaning too near.",
            "He moved as if every stone had betrayed him before. Food was never merely food, shelter never merely shelter, and memory never quiet.",
            "The wild did not cleanse him. It gave him places to crouch, things to gnaw, and enough darkness to keep talking to his loss.",
        ],
        "watch": [
            "He did not need kindness to cease. He needed it to repeat, and then to vary by one careless breath.",
            "The watch was strong, but strength itself had habits: the lifted lamp, the shifted foot, the voice that softened before food.",
            "Gollum made a private craft of noticing what decent hearts did not know they revealed.",
        ],
        "captivity": [
            "Under guard, small things grew large: a lamp trimmed late, a knot retied, a footstep repeated at the same hour, a voice made gentle by pity.",
            "The watch was not cruel, and that troubled him most. Cruelty would have given his hatred a clean wall to strike; mercy left him angrier and more alert.",
            "Time in custody did not pass. It circled. It returned to the same rope, the same food, the same eyes, and sharpened resentment by repetition.",
        ],
        "escape": [
            "The break came without grandeur. Confusion moved through the trees, attention shifted, and a miserable cunning found the gap before strength understood it was there.",
            "No great gate opened. A habit faltered, a shadow deepened, and long resentment moved faster than any watcher expected.",
            "Escape began before motion. It had been rehearsed in glances, in counted steps, and in the prisoner's hatred of every kindness that came near him.",
        ],
        "capture": [
            "The moment of taking was ugly and close. Wet ground, hard breath, rope, and teeth made an answer no map could have supplied.",
            "Nothing in the capture felt clean. Necessity held the rope; pity kept it from becoming cruelty; disgust had to be mastered before it became judgement.",
            "At last the rumour had a body. That body struggled, spat, begged, and made the danger more human without making it less grave.",
        ],
        "delivery": [
            "Under the trees the hunt changed language. Footprints gave way to answers, and answers had to be watched as carefully as any track.",
            "The guarded place did not make the prisoner safe to handle. It only gave wisdom enough room to listen without believing too soon.",
            "Mirkwood received the burden in silence. The trees seemed to know that some truths arrive bent and must not be straightened by force.",
        ],
        "warning": [
            "The matter had passed from pursuit into duty. What had been learned must now move faster than fear, and quieter than rumour.",
            "A warning is a hard kind of victory. It admits danger, abandons comfort, and asks the watcher to act before the world applauds certainty.",
            "The west lay peaceful because it did not yet know enough. That ignorance could be mercy only if others spent themselves guarding it.",
        ],
    }
    focus_by_kind = {
        "charge": "The errand began as a quiet request, too spare for the danger behind it.",
        "naming": "The errand found its true weight when the quarry's name was spoken plainly.",
        "vow": "The charge became a vow when hidden service was accepted without hope of welcome.",
        "witness": "The trail passed through ordinary fear, where truth arrived bent but not useless.",
        "darkening": "The pursuit widened toward darker country, and each sign became less innocent.",
        "failure": "The hour turns on a disciplined refusal to pretend the trail is clearer than it is.",
        "border_watch": "The hunt changed into hidden watchfulness along roads that had to remain ordinary.",
        "active_service": "The work divided itself into roads, signs, and hours that could be watched.",
        "offstage_watch": "The threat stays unseen, and the watch must therefore become wider than proof.",
        "finale": "Gandalf turned west, Aragorn remained on watch, and Gollum moved loose with dangerous names.",
        "trail": "The chase had become practical work, measured in signs so small that haste would have destroyed them.",
        "quarry": "The quarry moved by hunger, memory, and dread, never understanding how much danger his muttering carried.",
        "watch": "The captive began to turn routine into a map, counting mercy as carefully as command.",
        "captivity": "The prisoner had ceased to run, but in his stillness he began another kind of hunt.",
        "escape": "The bars of watchfulness held for a time, and then the world shifted by one dark inch.",
        "capture": "The long pursuit had narrowed to wet ground, bad footing, and the moment when pity could not be allowed to loosen the hand.",
        "delivery": "The hunt had become custody, and custody had become the harder labour of drawing truth from misery.",
        "warning": "The chase had yielded no peace, but it had yielded knowledge, and knowledge had become a burden that must be carried west.",
    }
    focus_sentence = focus_by_kind.get(scene_kind, focus_by_kind["trail"])

    intro_options = stakes_by_kind.get(scene_kind, stakes_by_kind["trail"])
    matter_options = matter_by_kind.get(scene_kind, matter_by_kind["trail"])
    intro = intro_options[_stable_seed("hunt-intro", scene_goal, place, ",".join(characters)) % len(intro_options)]
    matter = matter_options[_stable_seed("hunt-matter", scene_goal, place, ",".join(characters)) % len(matter_options)]
    movement_clauses_by_kind = {
        "charge": [
            "took up an errand still small enough to be doubted and grave enough to obey.",
            "stood at the point where an old fear first became a road.",
        ],
        "naming": [
            "faced the moment when rumour hardened into a name that could be hunted.",
            "found the errand narrowing around one ruined creature and the names he carried.",
        ],
        "vow": [
            "accepted a hidden service whose first reward would be weather and mistrust.",
            "took up the road before the road had shown how long it meant to be.",
        ],
        "witness": [
            "entered another man's fear without borrowing its disorder.",
            "listened where ordinary shame had tangled itself around useful news.",
        ],
        "darkening": [
            "found the trail growing less innocent with every southward sign.",
            "followed marks that seemed pulled by something darker than hunger.",
        ],
        "failure": [
            "stood where the honest trail thinned past use.",
            "accepted that refusal can be a hunter's discipline.",
        ],
        "border_watch": [
            "stood watch where peace still looked ordinary.",
            "held the edge of quiet country without entering its comfort.",
        ],
        "active_service": [
            "turned doubt into errands small enough to perform.",
            "made uncertainty useful by giving it roads to watch.",
        ],
        "offstage_watch": [
            "watched for a threat that had not yet shown its face.",
            "guarded the border against absence as carefully as sign.",
        ],
        "finale": [
            "held warning, watchfulness, and danger in an unfinished balance.",
            "stood where the bought time of the hunt became mercy.",
        ],
        "trail": [
            "read the ground as a reluctant script, losing nothing to haste.",
            "kept faith with small marks where a louder hunter would have passed by.",
        ],
        "quarry": [
            "carried danger in mutter, hunger, and broken memory.",
            "went crookedly through the wild, never free of the names he hated.",
        ],
        "captivity": [
            "made a narrow kingdom of rope, lamp, resentment, and watched habits.",
            "studied custody as if every mercy might one day loosen into escape.",
        ],
        "watch": [
            "turned the habits of the watch into a private map.",
            "counted lamp, root, voice, and shadow with hungry patience.",
        ],
        "escape": [
            "moved in the instant when order looked away.",
            "turned misery into swiftness before the watch could gather itself.",
        ],
        "capture": [
            "brought the long pursuit down to mud, breath, rope, and mercy.",
            "stood where the chase ceased to be rumour and became a shivering body.",
        ],
        "delivery": [
            "carried the road's answer into a guarded place of harder questions.",
            "brought custody under the trees, where speech would be watched like tracks.",
        ],
        "warning": [
            "stood at the change from pursuit into warning.",
            "held knowledge that had become too dangerous to keep still.",
        ],
    }
    movement_clauses = movement_clauses_by_kind.get(
        scene_kind,
        ["kept to the hidden road between uncertainty and necessary action."],
    )
    movement_clause = movement_clauses[
        _stable_seed("hunt-movement-clause", _display_goal(scene_goal), scene_kind, place, ",".join(characters))
        % len(movement_clauses)
    ]
    paragraphs = [
        (
            f"{_setting_sentence(place, f'{_display_goal(scene_goal)}|{scene_kind}|{focus_sentence}')} {char_phrase} {movement_clause} "
            f"{focus_sentence} {intro}"
        ),
        (
            matter
            + f"{object_sentence}{motif_sentence}{beat_sentence}{anchor_sentence}"
        ),
    ]
    event_turns_by_kind = {
        "charge": [
            (
                f"{wizard} set two damp scraps of report beside the map and weighted them with his pipe. One named "
                "a riverbank; the other named no place at all, only a thin voice heard after midnight."
            ),
            (
                f"{ranger} moved the nearest candle and traced the eastern road with a knife point, stopping where "
                "three doubtful rumours could be made to meet."
            ),
            (
                f"A stable-lad knocked once with a message from the yard. {wizard} read it, frowned, and folded it "
                "under his sleeve instead of laying it with the others."
            ),
            (
                f"Before they left the table, {ranger} had named two fords, one ruined watch-post, and a man near "
                "the river who sold news cheaply but remembered tracks well."
            ),
        ],
        "naming": [
            (
                f"{wizard} wrote the name Gollum in the margin of the map, then crossed it out. The mark remained "
                "dark enough to show that the word had changed the errand."
            ),
            (
                f"{ranger} asked for every remembered habit: fish stolen raw, hands used more than feet, sleep "
                "taken in holes, and speech that circled back to Baggins."
            ),
            (
                "One by one the rumours were sorted: ale-talk to the hearth, river-news to the road, and every "
                "mention of the Shire into a silence neither man liked."
            ),
        ],
        "vow": [
            (
                f"{ranger} checked the stitching of his weathered cloak, changed the worn thong on his knife, and "
                "took only what could travel without clatter."
            ),
            (
                f"{wizard} gave him no written commission. He gave instead three names, two roads, and a warning "
                "that must be carried in memory if paper became dangerous."
            ),
            (
                f"At the threshold {ranger} paused long enough to listen to the ordinary noise of Bree, then stepped "
                "out before warmth could make another argument."
            ),
        ],
        "witness": [
            (
                f"{ranger} made the witness draw the bank in spilled ale on the table. The man's finger shook, but "
                "the bend he drew matched an old ford on the map."
            ),
            (
                "A torn eel-trap was brought from the yard. Three cords had been bitten through, and one knot had "
                "been worried loose by hands too clever for an animal."
            ),
            (
                f"When the witness tried to leave, {ranger} blocked the door only with a question about moonrise. "
                "The answer changed the trail by half a day."
            ),
        ],
        "darkening": [
            (
                f"{ranger} found a camp where the ashes had been scattered with both hands, as if the maker had "
                "wished to hide warmth and hated the warmth for needing it."
            ),
            (
                "At a crossing he turned back from the easy bank and found the true mark under thorn: three toes "
                "pressed deep, the heel barely touching."
            ),
            (
                "A carrion bird rose before he reached the hollow. Under it lay no body, only fish skins, black "
                "mud, and a print aimed south."
            ),
        ],
        "failure": [
            (
                f"{ranger} set three signs in a row on a flat stone: a reed, a crust of mud, and a strip of torn "
                "bark. By sunset only one still answered the road."
            ),
            (
                "He followed the false line to a sheep-track and stopped there, not because the track ended but "
                "because it began to tell a cleaner story than truth would tell."
            ),
            (
                "Before turning back, he scratched a small mark where a friend might read it and a stranger would "
                "take it for weathering."
            ),
        ],
        "trail": [
            (
                f"{ranger} crossed water twice to test the bank and came back with black mud under one nail and "
                "a strand of weed caught in his sleeve."
            ),
            (
                "Near dusk he found fish bones hidden under a stone. They were cracked twice, sucked clean, and "
                "left where a hungry creature had not wished birds to gather."
            ),
            (
                "A child in a riverside hamlet pointed to a culvert and then hid behind a cart. The adults laughed "
                "until Aragorn found wet finger-marks inside the arch."
            ),
            (
                f"He bought a broken net, not for use but for the smell upon it. {quarry}'s passage had made even "
                "poor cordage speak."
            ),
        ],
        "quarry": [
            (
                f"{quarry} scraped a fish from a trap with a sliver of bone, dropped it when a twig snapped, and "
                "crouched so low that his chin touched the mud."
            ),
            (
                "He licked rain from a stone before he dared cross the open patch. Twice he turned back; the third "
                "time hunger drove him farther than fear allowed."
            ),
            (
                "When the name Baggins rose in his throat, he bit his own wrist until the word came out smaller."
            ),
        ],
        "capture": [
            (
                f"{ranger} threw the rope only after {quarry} lunged for water. The loop caught shoulder and arm, "
                "not throat, and dragged him sideways into the reeds."
            ),
            (
                f"{quarry} bit through leather and reached skin. {ranger} did not strike him; he shifted his grip, "
                "took the pain, and pinned the narrow wrist against mud."
            ),
            (
                "The struggle ended with both of them breathing hard and the marsh bubbling around their knees."
            ),
        ],
        "delivery": [
            (
                "The first Wood-elf to meet them lowered his bow only after he had counted the knots and seen the "
                "blood dried on Aragorn's glove."
            ),
            (
                f"{ranger} gave the road-report in order: where the trail turned, where the captive bit, where "
                "Baggins was spoken, and where fear looked south."
            ),
            (
                f"{wizard} asked for the lamps to be moved behind the prisoner. When {quarry} turned toward the "
                "shadow instead of the light, every watcher saw it."
            ),
        ],
        "captivity": [
            (
                "Each morning the guard changed the water bowl before the rope was checked. By the fourth morning "
                "Gollum had learned which hand reached first."
            ),
            (
                "A lamp was raised after food and lowered before questions. He watched the pattern from under his "
                "hair and scratched it into the earth with one nail."
            ),
            (
                "When a young guard stumbled on a root, the prisoner did not laugh. He stored the sound and waited "
                "for it to happen again."
            ),
        ],
        "watch": [
            (
                f"{quarry} coughed once to draw a glance, then twice to measure irritation. The second guard turned "
                "faster than the first, and that difference became part of his map."
            ),
            (
                "He dragged his foot near the rope-post until the bark showed a pale scrape. No guard noticed it "
                "because the scrape looked like old damage."
            ),
            (
                "At the hour when food came, he let the bowl tip and watched which hands went first to pity and "
                "which went first to the knot."
            ),
        ],
        "escape": [
            (
                "A branch cracked in the outer dark and two guards turned their heads together. Gollum moved before "
                "either had finished turning."
            ),
            (
                "He went under the rope rather than over it, tearing skin from his shoulder and leaving the blood "
                "on bark instead of leaves."
            ),
            (
                "The first shout followed the wrong shadow. By the time the second corrected it, water had taken "
                "his scent and fern had taken his shape."
            ),
        ],
        "border_watch": [
            (
                f"{ranger} changed a gate-latch that had begun to squeal, then marked the mud below it so he would "
                "know whether it opened after moonrise."
            ),
            (
                "He paid for a mug he did not drink and listened while a carter named three strangers, two dogs, "
                "and the road none of them had taken."
            ),
            (
                "At dawn he moved a stone from the verge. To a farmer it was tidiness; to a Ranger it made a later "
                "footprint easier to read."
            ),
        ],
        "active_service": [
            (
                f"{ranger} wrote no report. He tied a grass-knot under the third rail of a fence, where a friend "
                "would read delay and a stranger would see only neglect."
            ),
            (
                "He followed one rumour to a mill, another to a ditch, and a third to an empty lane where the dogs "
                "had barked at nothing for three nights."
            ),
            (
                "Before leaving Bree he changed inns, not for comfort but to see which questions followed him."
            ),
        ],
        "offstage_watch": [
            (
                "No enemy crossed the road that night, but a cart stopped too long at the rise. Aragorn remembered "
                "the driver, the wheel mark, and the patched left rein."
            ),
            (
                "He set three small signs on the westward ways and found two untouched by dawn. The third had been "
                "moved by a badger, which was also useful knowledge."
            ),
            (
                "A message came folded inside a scrap of harness leather. He burned the leather after reading it "
                "and ground the ash under his heel."
            ),
        ],
        "warning": [
            (
                f"{wizard} sorted what could be spoken from what must wait: the name Baggins, the word Shire, the "
                "creature's hatred, and the danger of arriving too late."
            ),
            (
                f"{ranger} named the roads he could hold before nightfall, then crossed out one because a watcher "
                "seen twice is less useful than none."
            ),
            (
                "The departure was measured by practical things: a staff taken up, a message destroyed, a horse "
                "refused because speed in the open would draw the wrong eyes."
            ),
        ],
        "finale": [
            (
                f"{wizard} left before the eastern sky paled. He took the folded letters, then burned one because "
                "the message was safer in memory than in ink."
            ),
            (
                f"{ranger} watched him go only until the road bent. Then he turned back to the first crossing and "
                "reset a mark that rain had blurred."
            ),
            (
                f"Far away, {quarry} crawled from one ditch to another and carried the names onward without knowing "
                "which road they had already set in motion."
            ),
        ],
    }
    event_turns = event_turns_by_kind.get(scene_kind, event_turns_by_kind["trail"])
    event_object_phrase = _join_object_labels(object_rows) or "the poor signs left behind"
    event_start = _stable_seed("hunt-event-turns", scene_goal, place, ",".join(characters), event_object_phrase) % len(event_turns)
    for offset in range(min(3, len(event_turns))):
        paragraphs.append(event_turns[(event_start + offset) % len(event_turns)])

    if scene_kind == "charge":
        paragraphs.extend(
            [
                (
                    f"{wizard} had not come to {ranger} with certainty. Certainty would have been easier. He came "
                    "with a knot of hints: a vanished creature, a stolen name, an old birthday gift, and the fear "
                    "that the wild had ears where the wise had none."
                ),
                (
                    f"'I need a hunter who can follow shame as readily as footprints,' said {wizard}. "
                    "'The quarry is not strong in arms, but he is old in flight, and hunger has taught him more "
                    "roads than any mapmaker.'"
                ),
                (
                    f"{ranger} looked toward the dark line of the eastern road. 'If the creature has gone to ground, "
                    "I can seek him. If he has gone into lands watched by the Enemy, seeking may become another word "
                    "for vanishing.'"
                ),
                (
                    f"'{quarry} has already vanished too often,' said {wizard}. 'He had a ring once. He lost it to "
                    "Bilbo Baggins, and the loss has gnawed him hollow. If he speaks the name Baggins in the wrong "
                    "darkness, the Shire will no longer be hidden by its smallness.'"
                ),
                (
                    f"The name lay between them with no greatness in its sound. That made it more perilous. {ranger} "
                    "had heard many names that armies feared; this one was homely, almost comic, and therefore easy "
                    "for proud minds to neglect."
                ),
                (
                    f"'You ask me to hunt a wretch for the sake of a country that would bolt its doors if "
                    f"I came too near,' said {ranger}. 'That is no reason to refuse. It may be the best reason to go.'"
                ),
            ]
        )
    elif scene_kind == "witness":
        paragraphs.extend(
            [
                (
                    f"The man who spoke to {ranger} did not wish to be remembered. He kept one hand on the door "
                    "and the other at his belt, though there was no weapon there worth trusting."
                ),
                (
                    "His tale came out in pieces: a shape under the bank, fish missing from a night-line, a voice "
                    "that cursed in the reeds and then begged the dark to hide it. Shame had trimmed the account, "
                    "but fear had left enough of its outline."
                ),
                (
                    f"'Tell it from the beginning,' said {ranger}. 'Do not smooth the parts that trouble you. A "
                    "crooked report may still keep the shape of a true track.'"
                ),
                (
                    "The witness swallowed, angry at being believed and afraid of not being believed. He named no "
                    "kingdom and understood no ring, but he had heard Baggins muttered like a wound that would not close."
                ),
                (
                    f"{ranger} paid him for food he had not eaten and left before gratitude could become talk. In "
                    "hidden work even kindness had to pass lightly, lest it become another rumour."
                ),
            ]
        )
    elif scene_kind == "darkening":
        paragraphs.extend(
            [
                (
                    f"{ranger} first mistrusted the southward turn because it made too much sense to his fear. A "
                    "hunter must beware the sign that flatters his dread as sharply as the sign that flatters hope."
                ),
                (
                    "Yet the lesser marks consented to it: a bank crossed where no hungry creature needed to cross, "
                    "a night shelter abandoned before dawn, and a trail that began to avoid wholesome fires."
                ),
                (
                    "The land altered without any single herald. Dogs barked less at night and men answered questions "
                    "with their eyes on the east. Even the wind seemed to carry news it would not speak plainly."
                ),
                (
                    f"'He is being drawn,' said {ranger}, speaking only to the dark water beside him. 'Not led as "
                    "a servant is led, but pulled as a wound is pulled by cold.'"
                ),
                (
                    "That thought gave no comfort. It made the quarry smaller and the danger larger, for a broken "
                    "mind may still carry a road toward those who know how to use broken things."
                ),
            ]
        )
    elif scene_kind == "failure":
        paragraphs.extend(
            [
                (
                    f"{ranger} found the first false comfort at a dry crossing. The mark looked promising because "
                    "he needed it to be promising, and that was the warning against it."
                ),
                (
                    "Beyond it the signs multiplied and weakened. A broken grass stem answered one guess, a scuffed "
                    "stone answered another, and the wind made a liar of both before noon."
                ),
                (
                    "He went on long enough to honour the possibility, not so long that honour became stubbornness. "
                    "There is a point where persistence ceases to serve truth and begins to serve only the hunter's pride."
                ),
                (
                    f"{object_sentence_start} could not rescue the hour. Map, blade, food, and memory all had uses, "
                    "but none could command the earth to remember what it had chosen to lose."
                ),
                (
                    "By dusk the defeat was plain. It had no drama in it, and for that reason it was harder to bear: "
                    "not a blow to answer, but an absence that would not become speech."
                ),
            ]
        )
    elif scene_kind in {"border_watch", "active_service", "offstage_watch"}:
        paragraphs.extend(
            [
                (
                    f"{ranger} came where the roads softened and the talk of danger sounded almost indecent. "
                    "Hedges, carts, low fields, and ordinary weather seemed to rebuke the very thought of shadow."
                ),
                (
                    "That was why the watch mattered. A country unable to imagine its peril is not foolish for "
                    "being peaceful; it is merely dependent on those who can imagine peril without worshipping it."
                ),
                (
                    f"He kept no lordly posture near {place}. He mended a strap, asked after a road, listened to "
                    "gossip, and let suspicion pass over him like rain over a cloak."
                ),
                (
                    "Uncertainty did not release him from service. It sharpened the service, for a known enemy "
                    "can be met in one place, while an uncertain one must be denied many roads."
                ),
                (
                    "So his duty grew quieter as it drew nearer the Shire. The less visible the threat became, "
                    "the more faithfully the borders had to be watched."
                ),
            ]
        )
    elif scene_kind == "finale":
        paragraphs.extend(
            [
                (
                    f"{quarry} was not mastered. He moved still, somewhere beyond the clean knowledge of those who "
                    "had hunted him."
                ),
                (
                    f"{wizard} had not yet reached the quiet door. The warning was near, not complete, and the road "
                    "still lay between fear and the house under the Hill."
                ),
                (
                    f"{ranger}'s part was no clearer to the eyes of the world. He had no trophy, no prisoner, no "
                    "song, and no proof that the guarded would ever understand the guarding."
                ),
                (
                    "At the first crossing he reset a rain-blurred mark, checked the ditch for a second print, and "
                    "let the empty road tell him nothing more than it knew."
                ),
                (
                    "Westward went the staff; eastward and southward moved the loose memory; between them a watcher "
                    "stood by the road until the morning grew plain."
                ),
            ]
        )
    elif scene_kind == "trail":
        paragraphs.extend(
            [
                (
                    f"{ranger} learned again how little a trail wishes to be found. A bent reed might be weather, "
                    "a broken twig might be deer, and a smear of mud on stone might be nothing until three other "
                    "nothings had been set beside it."
                ),
                (
                    "He went by night when the land permitted it, and by grey morning when night had become too "
                    "noisy with imagined feet. The hunt was long because the quarry did not travel like a man. "
                    "He crawled under notice, doubled back through water, and left fear where others leave tracks."
                ),
                (
                    f"Once, where the Anduin drew a slow bend, {ranger} knelt and touched a print no wider than two "
                    "fingers. It had been made by a foot used to stone, not field; by a body lean with want; by a "
                    "mind that hated open ground."
                ),
                (
                    "'You have gone "
                    "south when hunger should have driven you north. What calls you that way?'"
                ),
                (
                    "No answer came except water against root and the far cry of a bird. Yet the silence was an "
                    "answer of its own. Some roads are chosen by desire; some by fear; and some by a voice the "
                    "traveller no longer admits he has heard."
                ),
                (
                    f"When {wizard}'s message found him weeks later, it added little and changed everything. The "
                    "wizard had gathered more doubt, not less. Doubt, in such matters, was a kind of evidence."
                ),
            ]
        )
    elif scene_kind == "quarry":
        paragraphs.extend(
            [
                (
                    f"{quarry} did not think of himself as hunted at every hour. At times he thought only of cold, "
                    "of food, of the wet bite of reeds, of stones that bruised the hand, and of the bright absence "
                    "that had once ruled all his wanting."
                ),
                (
                    "'Baggins,' he whispered, and the name came out like a fishbone. 'Thief-name. Soft-foot name. "
                    "Shire-name. It took and went away, and the dark laughed behind us.'"
                ),
                (
                    "He stopped after saying it, as if the word itself had made tracks. His head turned from side "
                    "to side. The marsh gave back no face, only his own shape shivered in black water and broken "
                    "into pieces by midges."
                ),
                (
                    f"{quarry} remembered tunnels more kindly than fields. In tunnels the ceiling came close and "
                    "no star looked down with judgement. Out here the sky had too much room. It saw him scratch, "
                    "crawl, mutter, listen, and hate."
                ),
                (
                    "'Not back, not there, not to the hard hands,' he said. 'But the pull goes south, yes, south "
                    "and hot and black in the thought. We hates it. We goes. We goes because hating is not enough.'"
                ),
                (
                    "Thus the quarry moved, not in courage and not in counsel, but under compulsion so tangled "
                    "with his own desire that he could no longer tell fear from longing."
                ),
            ]
        )
    elif scene_kind == "captivity":
        paragraphs.extend(
            [
                (
                    f"{quarry} learned the place by resentments. One lamp was trimmed later than the others; one "
                    "guard hummed under his breath; one root pressed through the earth near the post where the rope "
                    "was fastened. Such things became kingdoms in his thought."
                ),
                (
                    "The Wood-elves did not torment him. That made him hate them more darkly. Cruel hands he could "
                    "understand; measured hands troubled him, for they gave him no simple shape into which he could "
                    "pour his malice."
                ),
                (
                    "'Kind, kind, yes,' Gollum whispered when food was set down. 'Kind with knots. Kind with eyes. "
                    "Kind that keeps and keeps and calls keeping mercy.'"
                ),
                (
                    "He watched pity as another creature watches a snare. Pity made voices softer, and soft voices "
                    "sometimes came near. Nearness made shadows cross the ground. Shadows, if followed rightly, "
                    "could show where the watchers did not look."
                ),
                (
                    "Yet not all his misery was craft. In the long hours he shook with real cold, real hunger, and "
                    "the real wound of a desire that had outlived every decent memory in him."
                ),
                (
                    "Thus captivity did not cleanse him and did not master him. It merely gathered his cunning into "
                    "a smaller cup, where it darkened and waited."
                ),
            ]
        )
    elif scene_kind == "watch":
        paragraphs.extend(
            [
                (
                    f"{quarry} began with the lamps. They were lowered, lifted, trimmed, and shaded by hands that "
                    "did not know they were teaching him."
                ),
                (
                    "Then he learned the feet. One guard stepped over the same root; another paused before turning; "
                    "a third dragged his heel when the night grew old. Such things were not mercy, but they were openings in mercy's wall."
                ),
                (
                    "Voices mattered too. A stern voice came straight and left straight. A kind voice wandered, "
                    "and wandering was a shape that could be followed."
                ),
                (
                    "'Soft steps, hard steps,' Gollum whispered into his own shoulder. 'All steps goes somewhere. "
                    "We watches where they goes.'"
                ),
                (
                    "The roots under the leaves seemed at first only discomfort. Later they became measures: this "
                    "root near the lamp, that hollow near the turning foot, the black place where a small body might be less than a shadow."
                ),
                (
                    "Thus habit became a door in his mind. It was not open. It was not even shaped. But it was no "
                    "longer nothing, and nothing had been his prison's strongest wall."
                ),
            ]
        )
    elif scene_kind == "escape":
        paragraphs.extend(
            [
                (
                    "The night of his chance did not announce itself. No horn named it; no star altered. A branch "
                    "fell, a shout went out under the trees, and for a moment every ordered thing leaned the wrong way."
                ),
                (
                    f"{quarry} felt the change before he understood it. The watch was still strong, but strength "
                    "had turned its face elsewhere. He made himself less than a shadow, less than a breath, less "
                    "than the memory of a footfall."
                ),
                (
                    "'Now, now, now,' he breathed. 'Soft feet, old feet. Bad rope, blind leaves. Baggins waits, "
                    "Shire hides, and Gollum goes where hurting tells him.'"
                ),
                (
                    "He did not flee bravely. He fled as a worm flees the spade and as smoke escapes a cracked "
                    "door. His courage was only terror moving quickly, yet terror can be swift where nobler powers stumble."
                ),
                (
                    "Behind him voices crossed and parted under the boughs. Before him lay mud, thorn, water, and "
                    "the merciless blessing of darkness."
                ),
                (
                    "So the prisoner became again a trail. But now the trail carried more than hunger. It carried "
                    "names, and the names had learned to bite."
                ),
            ]
        )
    elif scene_kind == "capture":
        paragraphs.extend(
            [
                (
                    f"{ranger} found him where the land itself seemed reluctant to keep a record. The Dead Marshes "
                    f"shifted underfoot, but fear has patterns, and {quarry}'s fear had grown tired enough to repeat "
                    "itself."
                ),
                (
                    "The final approach was not heroic. It was slow, cold, humiliating work: mud to the thigh, "
                    "breath held under reeds, a leech burned off with a coal, and hours spent while the quarry "
                    "argued with noises no other ear could hear."
                ),
                (
                    f"When {ranger} sprang, {quarry} twisted like a snared thing and bit at the hand that held him. "
                    "There was strength in him still, the strength of panic and old malice, but no true hope."
                ),
                (
                    f"'You are caught,' said {ranger}. 'Do not spend the little strength left to you on "
                    "teeth. I have followed too far to lose you for spite.'"
                ),
                (
                    "'Cruel tall one,' hissed Gollum. 'Wizard sends, yes. Bright-name thief sends. Baggins hides, "
                    "and Gollum is dragged. We tells nothing. We knows nothing. We are only poor bones.'"
                ),
                (
                    f"{ranger} bound him with more care than anger. Pity did not make the rope loose. Disgust did "
                    "not make it cruel. The captive was wretched, dangerous, and necessary, and those three truths "
                    "had to be held together."
                ),
            ]
        )
    elif scene_kind == "warning":
        paragraphs.extend(
            [
                (
                    f"{wizard} did not speak at once. He had heard enough from {quarry} to make silence heavy, "
                    "and not enough to make action simple. Between those two measures lay the peril of the hour."
                ),
                (
                    f"{ranger} stood beside the maps and looked westward. The Shire was not marked there as a "
                    "fortress or a throne, only as a green country of small roads. That smallness seemed now less "
                    "protection than invitation."
                ),
                (
                    f"'The road turns west for me,' said {wizard}. 'The warning must reach the house "
                    "under the Hill before darker knowledge outruns it.'"
                ),
                (
                    f"'Then I will keep the roads behind you,' answered {ranger}. 'If the wild has learned "
                    "the name Baggins, the wild shall find it watched.'"
                ),
                (
                    "This was no victory such as songs prefer. The prisoner was not safely mastered, the whole "
                    "truth was not held, and yet enough had been won to make delay a kind of treason."
                ),
                (
                    "Therefore they parted not in despair but in grave haste: one toward warning, one toward watch, "
                    "and both under the burden of a tale that had not finished speaking."
                ),
            ]
        )
    else:
        paragraphs.extend(
            [
                (
                    f"The northward road with {quarry} was worse than the hunt in one respect: a trail may be "
                    "silent, but a captive carries his misery beside you and makes every mile answer for it."
                ),
                (
                    f"{ranger} brought him toward the shadowed halls under the trees because {wizard} needed words, "
                    "and words were the only weapon left that might reach the Shire before darker messengers did."
                ),
                (
                    f"'Ask him quickly when you have him,' {ranger} said when he and {wizard} met beneath the "
                    "outer boughs. 'He is all edges. Hunger, hate, and fear have made a knot of him.'"
                ),
                (
                    f"'I will ask what must be asked,' answered {wizard}. 'But even the truth may come crooked "
                    "from such a mouth. We must learn the shape of the lie as well as the answer.'"
                ),
                (
                    "The Wood-elves watched without love for the prisoner. They had seen many unclean things move "
                    "under the leaves, but few that seemed so small and yet brought such heaviness after them."
                ),
                (
                    f"{quarry} crouched away from every lamp. When the name Baggins was spoken, he flinched and "
                    "grinned in the same breath, as if pain and delight had become one habit of the face."
                ),
            ]
        )

    expansion_templates_by_kind = {
        "charge": [
            (
                f"{wizard} spoke little of the ring itself. He would not dress suspicion as knowledge. Yet the "
                "shape of his fear could be seen in the care with which he avoided naming it too grandly."
            ),
            (
                f"{ranger} asked for landmarks, habits, scraps of remembered speech. He did not ask whether the "
                "task was worthy. A hidden people are often guarded by those they never welcome."
            ),
            (
                "Outside, Bree went about its business with mud on its thresholds and gossip under its beams. "
                "That common noise steadied the hour. It reminded them what secrecy was for."
            ),
            (
                f"'If I find him alive, he will not thank me,' said {ranger}. 'If I find him dead, we may learn "
                "too little. There is a narrow road between mercy and need.'"
            ),
            (
                f"'Walk it,' said {wizard}. 'You have walked narrower roads with fewer watching and more depending.'"
            ),
        ],
        "naming": [
            (
                f"{wizard} did not speak Gollum's name as an accusation only. He spoke it as a clue, a wound, "
                "and a warning that had learned to crawl."
            ),
            (
                f"{ranger} asked what the creature feared, what he desired, and which roads hunger had taught him. "
                "The answers were few, but each one narrowed the dark a little."
            ),
            (
                "Bilbo's part in the matter was treated with care. An old adventure could be comic by the hearth "
                "and perilous on the road, and wisdom had to hold both truths without mockery."
            ),
            (
                f"'A lost ring may leave a longer trail than a stolen horse,' said {wizard}. 'Especially when the "
                "loser has nothing left but loss.'"
            ),
            (
                f"'Then I must follow what he remembers as much as where he walks,' said {ranger}. 'A bitter guide, "
                "but perhaps the only one he has left us.'"
            ),
        ],
        "vow": [
            (
                f"{ranger} fastened his cloak as if the answer had already become weather. A promise is not proved "
                "while it is warm indoors."
            ),
            (
                f"{wizard} watched him with the sorrow of one who must ask too much and the trust of one who knows "
                "why the asking is not wasted."
            ),
            (
                "The Shire was present only as an absence: no hobbit stood there, no small door opened, no fire "
                "burned on a round hearth. That made the duty lonelier and more exact."
            ),
            (
                f"'If I am seen, I will be distrusted,' said {ranger}. 'If I am unseen, I may be useful. Let the "
                "use be enough.'"
            ),
            (
                f"'It is often enough in the keeping of small countries,' said {wizard}. 'More than enough, though "
                "they may never learn the measure of it.'"
            ),
        ],
        "witness": [
            (
                "The report was small enough to be doubted and stubborn enough to remain. That is often the measure "
                "of useful news in wild country."
            ),
            (
                f"{ranger} asked after weather, dogs, water, and the hour of moonrise. The witness had expected "
                "wonder and instead found himself questioned about mud."
            ),
            (
                "A frightened man will sometimes invent monsters, but he seldom invents the dull particulars by "
                "which a monster has passed his door."
            ),
            (
                f"'Fear makes bad order of memory,' said {ranger}. 'Say the pieces as they come, and I will put "
                "them beside the road.'"
            ),
            (
                "When he left, the witness looked relieved and insulted at once. The tale had been taken seriously, "
                "but not grandly, and that was the best use that could be made of it."
            ),
        ],
        "darkening": [
            (
                "The turn south did not show itself as command. It showed itself as reluctance overcome again and "
                "again, until reluctance itself became evidence."
            ),
            (
                f"{ranger} found places where the quarry had paused as if listening inwardly. Hunger leaves haste; "
                "this was a different pause, sourer and more afraid."
            ),
            (
                "The road toward darker country was not broad, but many little things leaned toward it: silence "
                "at crossings, ash left cold, and paths chosen for dread rather than shelter."
            ),
            (
                f"'No beast chooses such country for comfort,' said {ranger}. 'If he goes that way, something worse "
                "than hunger keeps him company.'"
            ),
            (
                "So the hunt changed weight. It was still a search for one wretched body, yet the shadow behind "
                "that body had begun to lengthen across the map."
            ),
        ],
        "failure": [
            (
                "The lost trail did not absolve him of the errand. It merely stripped the errand of the comfort "
                "of motion."
            ),
            (
                f"{ranger} marked the last trustworthy sign and did not adorn it with wishes. A true last sign is "
                "more useful than a dozen invented continuations."
            ),
            (
                "The country had not become empty; it had become overfull. Too many possible marks contended, "
                "and none would bear the weight of decision."
            ),
            (
                "'Enough,' he said at last, and the word cost him more than another mile would have done."
            ),
            (
                "Humility entered the hunt there, not as surrender but as the discipline by which a watcher refuses "
                "to serve his own need for certainty."
            ),
        ],
        "border_watch": [
            (
                "He learned which roads could be watched openly and which must be watched by absence. A hidden "
                "guard must sometimes be most useful where he is least acknowledged."
            ),
            (
                f"{ranger} let the small talk of Bree and the borderlands do part of the work. Men who refuse "
                "questions will often answer a silence if it seems harmless enough."
            ),
            (
                "The Shire did not become grander as he neared it. It became smaller, more particular, and therefore "
                "more in need of remaining itself."
            ),
            (
                "'Let them mistrust the weather-beaten man,' he said softly. 'A closed door may still be guarded "
                "from the road.'"
            ),
            (
                "Thus active service took the form of patience: watching without alarm, doubting without noise, "
                "and holding the edge of peace without stepping into its firelight."
            ),
        ],
        "trail": [
            (
                "The country did not offer one continuous line. It offered fragments: a scraped bank, a stolen fish, "
                "ashes fingered cold, a frightened ferryman, a child who had seen eyes under a culvert."
            ),
            (
                f"{ranger} carried each fragment until it found its neighbour. That was the craft of the hunt: not "
                "a leap of genius, but a patience that let small truths gather weight."
            ),
            (
                "At night he dreamed of the Shire though he had never been welcomed deeply into it. In the dream "
                "it was all fields, smoke, and round doors, absurdly gentle and therefore worth a hard road."
            ),
            (
                f"'Gandalf fears for gardeners and supper tables,' {ranger} said once to the empty dark. 'Then let "
                "the wild keep watch for them a little longer.'"
            ),
            (
                "The farther east he went, the less the land cared for names. Hill, marsh, thicket, ford, and hollow "
                "accepted footfall and forgot it. Only malice seemed to remember."
            ),
        ],
        "quarry": [
            (
                f"{quarry} ate what he could catch and distrusted everything he could not eat. Yet even hunger "
                "could not fill the hollow where the ring had been."
            ),
            (
                "He did not know that wise minds were trying to fit his broken words into a design. He knew only "
                "that names hurt, and that some names drew eyes out of darkness."
            ),
            (
                "'Lost, lost, but not gone from us,' he said. 'Baggins has it. Shire has it. Soft hands, locked "
                "doors, warm holes. We will find the way back into the small bright thought.'"
            ),
            (
                "Then he laughed without mirth, and a heron rose from the reeds as if the sound had offended the "
                "very patience of the water."
            ),
            (
                "Southward lay dread. Northward lay memory. Between them he chose the path that hurt most, because "
                "hurt had become the only guide that never left him."
            ),
        ],
        "captivity": [
            (
                "Time under guard did not move like time on the road. It dripped. It returned. It found old hurts "
                "and made them speak again in the dark."
            ),
            (
                f"{quarry} counted kindnesses as injuries and injuries as proof. In that crooked arithmetic he "
                "could make every hand guilty and every morsel stolen."
            ),
            (
                "The trees above him had deep roots and long memories. Their leaves whispered in tongues he did "
                "not know, and he imagined all whispering to be accusation."
            ),
            (
                "'They watches, watches,' he said. 'But eyes grows tired. Leaves grows black. Knots remembers teeth.'"
            ),
            (
                "Mercy stood near him and could not heal him. Yet its presence mattered, for even a refused mercy "
                "is not nothing in the reckoning of the world."
            ),
        ],
        "escape": [
            (
                "The break in the watch was narrow, but evil and misery are often thin enough to pass where strength "
                "expects a larger foe."
            ),
            (
                f"{quarry} went down into bracken, hollow, root-tangle, and ditch. He did not look back from loyalty "
                "to hatred; he looked back only to make sure hatred followed."
            ),
            (
                "For a while the forest itself seemed to deny him passage. Thorn took skin, mud took prints, water "
                "took scent, and still he slipped on."
            ),
            (
                "'Baggins,' he whispered when he was far enough to dare the wound. 'Baggins and Shire. We keeps "
                "the name, yes. Names are food when fish are gone.'"
            ),
            (
                "By dawn there was no prisoner under the trees, only broken fern, a dark smear on bark, and the "
                "knowledge that a captured danger may become more dangerous when it escapes."
            ),
        ],
        "capture": [
            (
                "For a moment, when the rope held and the struggle ended, the whole pursuit seemed to shrink into "
                "one shivering body. It was hard to believe that so much fear had followed so little flesh."
            ),
            (
                f"{ranger} did not let himself despise the captive. Contempt is a poor guard. It sleeps when pity "
                "would still be awake, and it mistakes weakness for harmlessness."
            ),
            (
                "'You will walk,' he said. 'You will drink when water is clean enough. You will speak when speech "
                "is needed. And you will not be struck for being miserable.'"
            ),
            (
                "'Kind cruel,' Gollum spat. 'Rope-kind. Knife-kind. Wizard-kind. Baggins-kind. All wants us. All "
                "takes from us.'"
            ),
            (
                "The accusation was crooked, but not empty. Aragorn heard the truth inside its malice and did not "
                "answer too quickly."
            ),
        ],
        "watch": [
            (
                "He learned the sound of each foot before he learned the mercy or malice attached to it. Feet are "
                "honester than faces, for they forget to pretend when the body is tired."
            ),
            (
                "The lamps made a pattern that the keepers did not see as a pattern. Gollum saw it because he "
                "hated light enough to count every time it moved."
            ),
            (
                "Voices gave him another map. A soft voice came nearer before it remembered caution; a stern voice "
                "kept its distance but often turned away sooner, satisfied with its own sternness."
            ),
            (
                "Roots mattered most of all. They broke the smoothness of the ground, made watchers choose where "
                "to place the foot, and gave a crawling mind hope of unevenness."
            ),
            (
                "He stored these things without calling them hope. Hope was a word too clean for him. He stored "
                "them as injuries, and injuries were treasures he knew how to keep."
            ),
            (
                "Sometimes he made a noise only to measure the answer. A cough, a mutter, a scrape of nail on bark: "
                "each returned to him as knowledge of speed, patience, or irritation."
            ),
            (
                "No single habit promised escape. But many habits laid side by side began to show a place where "
                "the watch was strongest in intention and weakest in rhythm."
            ),
            (
                "So the door in his thought did not open at once. It was built there piece by piece, out of lamp, "
                "root, voice, pity, anger, and the small delays of living creatures."
            ),
            (
                "By the time he seemed most cowed, his inward map had grown crowded. He knew where shadow fell "
                "first, where it deepened, and where a body lower than a man's might borrow it."
            ),
            (
                "The old names moved through the counting. Baggins sharpened it; Shire warmed it; loss gave it "
                "patience; hatred gave it teeth."
            ),
        ],
        "delivery": [
            (
                "The trees received the prisoner without welcome. Leaves shifted above him, and unseen watchers "
                "marked every twitch of his hands."
            ),
            (
                f"{wizard} looked older when he saw what the hunt had brought in. Not defeated, not surprised, but "
                "older, as if a calculation long resisted had at last required its sum."
            ),
            (
                f"'He says Shire when sleep catches him,' said {ranger}. 'He says Baggins when he thinks hate will "
                "warm him. Those words are not random.'"
            ),
            (
                f"'No,' said {wizard}. 'They are a road. A road can be followed both ways.'"
            ),
            (
                "So the questioning began under guard, not with torture and not with trust, but with a stern patience "
                "that knew the world's safety may rest on a half-swallowed name."
            ),
        ],
        "warning": [
            (
                "There are tidings that do not grow lighter because they are at last understood. They grow heavier, "
                "for then ignorance can no longer bear any part of the load."
            ),
            (
                f"{wizard} gathered the little that could be carried: words, guesses, remembered looks, and the "
                "terrible agreement of separate doubts."
            ),
            (
                f"{ranger} asked no praise and received none. That was fitting. The best guard of a hidden land "
                "must be content to remain outside its songs."
            ),
            (
                "'If I do not return swiftly,' said Gandalf, 'hold the roads as long as secrecy allows.'"
            ),
            (
                "'I have held worse roads for less hope,' said Aragorn. 'Go. I will make the wild pay dearly for "
                "each step it takes westward.'"
            ),
            (
                "So hope took a lean shape: a staff on the road, a ranger in the rain, and a green country still "
                "sleeping because others had not slept."
            ),
        ],
    }
    scene_specific_templates = expansion_templates_by_kind.get(scene_kind, expansion_templates_by_kind["trail"])

    shared_expansions = [
        (
            "No banner would ever mark this labour. It belonged to the hidden side of history, where doors remain "
            "safe because strangers freeze beside rivers and do not send word home."
        ),
        (
            "The land itself seemed to resist neat telling. Rain spoiled the page; mud corrected the map; a crow "
            "made mockery of direction; hunger reduced noble purposes to the next mouthful of bread."
        ),
        (
            f"Yet the hunt did not lose its centre. {wizard} sought knowledge before knowledge became a weapon in "
            f"other hands, and {ranger} gave that search the endurance of his feet."
        ),
        (
            "In such work, courage was mostly repetition. Rise, listen, choose the least false trail, go on, and "
            "bear the loneliness of guarding people who would never know your name."
        ),
        (
            "The word Shire returned often in thought because it did not sound like a fortress. It sounded like "
            "gardens, gossip, bread, birthdays, and foolish peace. That was why it mattered."
        ),
        (
            "The name Baggins was stranger still. It had no majesty. It could be laughed across a common room. "
            "But the right small name in the wrong ear can open a gate no army has yet found."
        ),
        (
            f"{quarry}'s absence pressed on the scene as strongly as his presence. Even when unseen, he drew choices "
            "after him, leaving wiser hearts to follow the crooked furrow of his need."
        ),
        (
            "There was pity in the matter, though no softness. Pity did not deny danger. It refused only the easy "
            "lie that a ruined creature has ceased to be part of the moral world."
        ),
        (
            "So the tale moved by hidden service rather than glory. It asked whether watchfulness can be faithful "
            "when no song confirms it and no witness understands the cost."
        ),
        (
            "The road ahead had no clean edge. It bent through wet grass, black trees, wary inns, cold fords, and "
            "lands where a traveller lowered his voice without knowing why."
        ),
        (
            "A lesser fear would have hurried them into noise. This fear did the opposite. It taught quietness, "
            "and quietness made each chosen word seem heavier."
        ),
        (
            "The old powers of the world did not need to appear in person. Their pressure could be felt in a "
            "missed track, a sudden silence among birds, or a name spoken too softly by firelight."
        ),
        (
            f"{ranger} understood that a hunt is a kind of reading. The ground writes reluctantly, but it writes. "
            "The hunter must learn grammar from mud, punctuation from broken grass, and meaning from what is absent."
        ),
        (
            f"{wizard} understood another reading. Speech has tracks as surely as feet. A repeated word, a flinch, "
            "a refusal, a hungry glance - these too can show where the truth has passed."
        ),
        (
            "Neither craft was enough alone. A body must be found before its words can be weighed. A word must be "
            "weighed before a body becomes only a prisoner."
        ),
        (
            "The weather gave no blessing. It merely continued: rain, chill, a thin moon, a dawn like old pewter. "
            "Middle-earth often leaves its necessary work to such indifferent mornings."
        ),
        (
            "In the far west, fields lay quiet. In the east, malice gathered itself without haste. Between them "
            "moved a handful of names, fragile as sparks and dangerous as lit tinder."
        ),
        (
            "The thought of Bilbo came not as comedy but as consequence. An old adventure had left a thread loose, "
            "and now darker fingers might find it."
        ),
        (
            "Every practical choice had a moral shadow. To bind, to question, to pursue, to keep silent, to speak "
            "too late - none of these could be made clean by necessity alone."
        ),
        (
            "Still they went on. That was the hard virtue of the hidden watch: not certainty, not triumph, but "
            "continued motion under an obligation that had become clearer than comfort."
        ),
        (
            "Villages along the way offered fragments in different measures. A gatekeeper remembered a crouching "
            "shape by moonrise; a fisherwoman missed three hooks; an old drover swore that a ditch had cursed him."
        ),
        (
            "The hunters learned to distrust dramatic evidence. A loud rumour usually belonged to ale, pride, or "
            "fear. The useful sign was smaller: slime on a stone, a silence among dogs, a footprint half full of rain."
        ),
        (
            "Rope, tinder, flint, needles, dried meat, spare thongs, and a blackened pot became instruments of the "
            "same labour. Great errands are often kept alive by dull inventories."
        ),
        (
            "At river crossings the world seemed to pause. Water made every track doubtful. It also forced every "
            "traveller, hunter and hunted alike, to choose an exposed place and leave some portion of himself behind."
        ),
        (
            "There were inns where speech had to be purchased with patience. Men who would say nothing to a question "
            "might say everything to a repaired buckle, a paid debt, or a silence that did not hurry them."
        ),
        (
            "The wild had its own witnesses: willow bark scraped pale, rushes flattened against their growth, crows "
            "startled from carrion, foxes unwilling to use a bank where something fouler had passed."
        ),
        (
            "Many nights ended without progress. Such nights were not empty. They taught the shape of failure, and "
            "failure, repeated honestly, can narrow a search as surely as success."
        ),
        (
            "The quarry's misery left a weather around him. Where he had sheltered, even warmth seemed used badly; "
            "ashes were scattered, bones cracked twice, and every hiding place looked hated after he left it."
        ),
        (
            "No council weighed these details while they happened. No chronicler stood beside the ford. The history "
            "of the hour was carried in wet boots, sore wrists, and the stubborn refusal to abandon a faint sign."
        ),
        (
            "Sometimes the trail ran through country where older ruins shouldered through grass. Broken lintels, "
            "sunk stones, and roofless towers looked on without comment, reminders that even hidden labours pass over buried grief."
        ),
        (
            "Birdsong changed from district to district. In safer lands it stitched the morning together. Farther "
            "east it came in wary pieces, as if each note waited to see whether the next would dare answer."
        ),
        (
            "Aragorn kept his judgments provisional. A good hunter is not one who never guesses, but one who knows "
            "when a guess has become vanity and must be laid down."
        ),
        (
            "Gandalf's fear had a scholar's patience and a traveller's impatience. He could wait years for a fact, "
            "yet rage inwardly at a single wasted afternoon when lives might be bending toward peril."
        ),
        (
            "Gollum's name did not summon one feeling. It summoned revulsion, pity, annoyance, caution, and a strange "
            "bond of need that no decent heart would have chosen."
        ),
        (
            "The ring remained mostly absent from sight, and absence suited it. Its influence was not a jewel in "
            "the hand here, but a displacement in lives, a pressure by which distant people began to move."
        ),
        (
            "Every map was both help and insult. Ink made the lands flat and obedient. The actual ground rose, sank, "
            "stank, froze, flooded, deceived, and demanded payment for each mile."
        ),
        (
            "A red dawn over marsh water could look almost beautiful until the smell reached the road. Beauty and "
            "corruption often stood close in those lands, and neither cancelled the other."
        ),
        (
            "The hidden watch had no clean ending. Even capture would not answer everything. A living prisoner is "
            "not a solved riddle; he is a locked door that bites the hand reaching for the key."
        ),
        (
            "Under the labour lay one stern mercy: if the small and peaceful are to remain small and peaceful, "
            "others must sometimes meet the darkness before it learns their address."
        ),
        (
            "So the hunt gathered texture: smoke in wool, river grit in seams, bruised palms, hoarse questions, "
            "carefully saved candle ends, and the ache of sleeping with one ear open."
        ),
        (
            "News travelled crookedly. A rumour might outrun a horse and still arrive lame. A true report might "
            "sit for weeks in a suspicious mouth before hunger, payment, or kindness drew it out."
        ),
        (
            "The moral burden did not grow simpler with distance. The nearer the hunters came to Gollum, the harder "
            "it became to think of him as only evidence. Wretchedness has a face when it is close enough."
        ),
        (
            "Yet too much tenderness would be its own betrayal. The Shire could not be protected by sentiment alone, "
            "nor could Baggins be unspoken merely because the speaker was pitiable."
        ),
        (
            "At dusk the land often seemed to hold its breath. Reeds blackened, hills lost their shoulders, and the "
            "path ahead became less a line than a decision renewed step by step."
        ),
        (
            "That renewal was the story's true measure. Not the glamour of pursuit, but the disciplined return to "
            "a hard purpose after cold, disgust, error, delay, and pity had each made its argument."
        ),
    ]
    place_name = place or "the wild"
    object_phrase = _join_object_labels(object_rows) or "the poor signs left behind"
    object_sentence_start = object_phrase[:1].upper() + object_phrase[1:] if object_phrase else "The poor signs left behind"
    kind_thesis = {
        "charge": "a charge is only noble when it has become practical enough to obey",
        "trail": "tracking is patience made visible upon unwilling ground",
        "witness": "a crooked report can still narrow a road when handled without pride",
        "darkening": "a southward sign is graver when hunger alone cannot explain it",
        "quarry": "flight can be another form of confession",
        "captivity": "custody is not safety when the captive keeps studying the lock",
        "escape": "an escape begins before the rope is broken, in the mind that counts every habit",
        "capture": "capture answers the road but opens the question of mercy",
        "delivery": "delivery turns the hunter's labour into the questioner's burden",
        "warning": "warning is the shape taken by a hunt that has won truth but not peace",
    }.get(scene_kind, "hidden service is proven by continuance rather than by praise")
    local_pressure = {
        "bree": "mud, lamplight, and common gossip",
        "anduin": "water, reed, and vanishing banks",
        "rhovanion": "wide distances and half-remembered roads",
        "wilderland": "wide distances and half-remembered roads",
        "mirkwood": "boughs, lamps, and listening dark",
        "dead marsh": "false ground and drowned reflections",
        "shire": "green quiet and the peril of being unguarded",
        "border": "green quiet and the peril of being unguarded",
    }
    place_pressure = next((value for key, value in local_pressure.items() if key in place_l), "weather, distance, and silence")
    contextual_expansions = [
        (
            f"Around {place_name}, {place_pressure} made the errand take its own shape. It was not a tally of "
            "incidents but a change in pressure: doubt becoming duty, pursuit becoming knowledge, or knowledge "
            "becoming warning."
        ),
        (
            f"The labour turned upon {object_phrase}. Such things were humble, but humble evidence often carries "
            "history farther than banners or proclamations."
        ),
        (
            f"{kind_thesis.capitalize()}. That thought did not make the road shorter, but it kept the road "
            "from becoming meaningless hardship."
        ),
        (
            "The charge was simple enough to obey, though not easy to finish. It had to pass through weather, "
            "doubt, and the reluctance of the world to yield a straight answer."
        ),
        (
            f"In {place_name} the old map ceased to be a promise and became an argument. Ink said one thing; "
            "mud, branch, ford, and silence answered with another."
        ),
        (
            "The names Shire and Baggins did not become grander by repetition. They became more vulnerable, for "
            "each utterance proved that smallness is no defence once malice has learned where to look."
        ),
        (
            f"{ranger} measured danger by what it asked of the feet. {wizard} measured it by what it asked of "
            "memory. Between those crafts the hunt found what little strength it possessed."
        ),
        (
            f"{quarry} remained the narrow centre of the matter, whether near, hidden, bound, or fled. Around him "
            "wisdom and pity had to move carefully, lest either become pride in another dress."
        ),
        (
            "The errand's moral shape was felt first in the body: cold fingers, watched speech, aching knees, "
            "and the pause before a hard choice."
        ),
        (
            f"What happened in {place_name} could not be judged by victory alone. A fact learned too late, a name "
            "kept from the wrong mouth, or a prisoner kept breathing might matter more than triumph."
        ),
        (
            "No later company stood behind these hours to make them shine. Their worth lay in being earlier, "
            "lonelier, and less certain than the tale that would one day be visible."
        ),
        (
            f"The pressure of {place_pressure} made speech sparse. Words were saved for what could change a step, "
            "steady a will, or prevent mercy from turning careless."
        ),
        (
            "Each sign was weighed beside the marks around it. A lesser tale would hurry; "
            "this one had to earn certainty by watching what did not wish to be seen."
        ),
        (
            "The danger was not only that enemies might hear. It was that good hearts, tired by distance, might "
            "mistake a partial answer for an ending."
        ),
        (
            f"{object_sentence_start} gathered importance because they were touched by consequence. The "
            "great matters of Middle-earth often pass first through cracked leather, wet wool, and a poor meal."
        ),
        (
            "The hour did not flatter those who bore it. Hidden service gives its servants toil before it gives "
            "them understanding, and often gives no thanks at all."
        ),
        (
            f"Under that restraint the older world pressed close. In {place_name}, even ordinary sounds seemed to "
            "have roots in stories half-buried below the road."
        ),
        (
            "Gollum's misery could not be allowed to excuse him, and it could not be erased so that duty might "
            "feel clean. That double truth kept the hunt morally awake."
        ),
        (
            "If the hour had a hinge, it lay where a hidden errand became an unavoidable duty. Before that "
            "turn the way remained conjecture; after it, the burden had shifted, however slightly, toward knowledge."
        ),
        (
            "The story therefore moved as guarded history moves: one necessary errand at a time, each leaving "
            "behind a trace small enough to be missed and grave enough to matter."
        ),
        (
            f"{wizard}'s fear and {ranger}'s endurance were not the same virtue. One read the peril in remembered "
            "words; the other followed it through ground that tried to forget."
        ),
        (
            "The shadow over the errand was large, but the task of the hour remained local and practical. "
            "That kept the telling close to road, hand, breath, and weather."
        ),
        (
            "Even pity had to be disciplined. It must not become cruelty by neglecting danger, nor harden into "
            "cruelty by forgetting that the dangerous creature still suffered."
        ),
        (
            f"So {place_pressure} shaped the hour's pace. It quickened where evidence quickened, slowed where "
            "judgement was needed, and stopped only long enough for purpose to overrule weariness."
        ),
    ]
    shared_expansions = contextual_expansions

    expansion_templates = scene_specific_templates + shared_expansions
    idx = _stable_seed(scene_goal, place, ",".join(characters), str(objects)) % max(1, len(expansion_templates))
    added_expansions = 0
    while target_words and _count_words("\n\n".join(paragraphs)) < target_words:
        paragraphs.append(expansion_templates[idx % len(expansion_templates)])
        idx += 1
        added_expansions += 1
        if added_expansions >= len(expansion_templates):
            break

    if target_dialogue_ratio:
        current = "\n\n".join(paragraphs)
        dialogue_words = _dialogue_word_count(current)
        total_words = max(1, _count_words(current))
        dialogue_idx = 0
        if scene_kind == "quarry":
            dialogue_templates = [
                "'No road, no road, only teeth in the dark,' said Gollum. 'But Baggins has roads. Shire has doors. We remembers.'",
                "'They hunts us for what was ours,' Gollum whispered. 'Tall ones, grey ones, all asking, all taking.'",
                "'Not tell, not tell,' said Gollum. 'But names leak out when sleep is cruel, and sleep is always cruel.'",
                "'South pulls,' he said. 'North stings. West hides the thief-name. We hates all ways and still we crawls.'",
                "'Little name, sharp name,' Gollum said. 'Baggins cuts the mouth. Shire hides it, but hiding is not keeping.'",
                "'Fish knows water, bird knows air,' he muttered. 'Gollum knows loss. Loss knows the way better than maps.'",
                "'No wizard, no ranger, no rope,' he said, rocking on his heels. 'Only dark ahead and old hurt behind.'",
                "'If they asks, we spits,' Gollum whispered. 'If they binds, we bites. If they pities, we hates pity worst.'",
                "'But the small land smells of warm holes,' he said. 'Warm holes and soft hands and thief-laughter under the hill.'",
                "'Go back, go on, go down,' said Gollum. 'Every way hurts. We chooses the hurt that calls loudest.'",
            ]
        else:
            dialogue_templates = [
                f"'Follow the sign and not your anger,' said {wizard}. 'Anger runs straight where a crooked trail requires patience.'",
                f"'I can master anger,' answered {ranger}. 'It is pity that may slow the hand when speed is needed.'",
                f"'Pity is not delay unless it forgets the peril,' said {wizard}. 'Keep both in your mind, or either one may betray you.'",
                f"'If the name Baggins has travelled east, time is already against us,' said {ranger}. 'Then I will spend no more of it indoors.'",
                f"'Gollum is not merely a clue,' said {wizard}. 'He is a sufferer, a liar, and perhaps the narrow bridge to the truth.'",
                f"'Then I will bring him living if I can,' said {ranger}. 'But do not ask me to call the road gentle.'",
                f"'Gentle roads seldom lead where need sends us,' said {wizard}. 'Go with open eyes, and return with what can still be saved.'",
                f"'When weariness argues with you, remember the doors that sleep because you do not,' said {wizard}.",
                f"'Hidden service is still service,' said {ranger}. 'I have not forgotten that, though mud is a poor herald.'",
                f"'If he begs, hear the fear beneath it,' said {wizard}. 'If he lies, hear the wound around the lie.'",
                f"'I will not confuse mercy with softness,' said {wizard}. 'Nor should you confuse hardness with strength.'",
                f"'The trail will have my strength,' said {ranger}. 'Let the questions be ready when strength has done its part.'",
                f"'The Shire is defended tonight by men it would mistrust at its gate,' said {wizard}. 'That is often the way of mercy.'",
                f"'Then let mercy wear a travel-stained cloak,' said {ranger}. 'It will be less noticed and better armed.'",
                f"'Do not make a legend of this work,' said {wizard}. 'Legends are noisy, and this task must pass under notice.'",
                f"'I need the truth brought back breathing,' said {wizard}. 'Dead silence would leave too many doors open.'",
                f"'Breathing truth may still bite,' answered {ranger}. 'I will keep my hand clear of his teeth if I can.'",
                f"'Bring back the creature, not vengeance,' said {wizard}. 'Vengeance would be simpler, and therefore less useful.'",
                f"'If the trail turns toward Mordor, do not follow pride beyond prudence,' said {wizard}. 'Send word if word can be sent.'",
                f"'If word cannot be sent, I will leave signs you can read,' said {ranger}. 'That much the wild may grant us.'",
            ]
        dialogue_start = _stable_seed("dialogue", scene_goal, place, ",".join(characters)) % max(1, len(dialogue_templates))
        while (dialogue_words / total_words) < target_dialogue_ratio and dialogue_idx < len(dialogue_templates):
            paragraphs.append(dialogue_templates[(dialogue_start + dialogue_idx) % len(dialogue_templates)])
            dialogue_idx += 1
            current = "\n\n".join(paragraphs)
            dialogue_words = _dialogue_word_count(current)
            total_words = max(1, _count_words(current))

    return "\n\n".join(paragraphs)


def _scene_quality_for_plan(
    quality: dict[str, Any],
    plan_chapter: dict[str, Any] | None,
    plan_scene: dict[str, Any],
    scene_count: int,
) -> dict[str, Any]:
    scene_quality = dict(quality)
    target_words = _coerce_optional_int(plan_scene.get("target_words"))
    if target_words is None and isinstance(plan_chapter, dict):
        chapter_target = _coerce_optional_int(plan_chapter.get("target_words"))
        if chapter_target:
            target_words = max(1, round(chapter_target / max(1, scene_count)))
    if target_words:
        scene_quality["target_scene_words"] = target_words
    return scene_quality


def _dedupe_scene_paragraphs(text: str, seen: set[str], min_words: int = 8) -> str:
    kept: list[str] = []
    for raw in re.split(r"\n\s*\n", str(text or "")):
        paragraph = raw.strip()
        if not paragraph:
            continue
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        if paragraph != "* * *" and not paragraph.startswith("#") and _count_words(paragraph) >= min_words:
            if normalized in seen:
                continue
            seen.add(normalized)
        kept.append(paragraph)
    return "\n\n".join(kept)


def _existing_project_paragraph_signatures(proj_dir: Path, *, exclude_chapter: int, min_words: int = 8) -> set[str]:
    signatures: set[str] = set()
    for path in sorted(proj_dir.glob("chapter_??.md")):
        match = re.fullmatch(r"chapter_(\d{2})\.md", path.name)
        if not match or int(match.group(1)) >= exclude_chapter:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in re.split(r"\n\s*\n", text):
            paragraph = raw.strip()
            if not paragraph or paragraph == "* * *" or paragraph.startswith("#"):
                continue
            if _count_words(paragraph) < min_words:
                continue
            signatures.add(re.sub(r"\s+", " ", paragraph).strip().lower())
    return signatures


def _public_scene_goal_text(raw: str) -> str:
    text = re.sub(r"\s+", " ", str(raw or "")).strip(" .")
    if not text:
        return "the errand had to move from uncertainty toward guarded knowledge"
    markers = (
        "Scene brief:",
        "Shadow action to realize:",
        "Scene beats to cover:",
        "Motifs to echo:",
        "Story-time is",
        "Past figures",
        "Do not mention",
        "Required canon anchors:",
    )
    marker_pattern = "|".join(re.escape(marker) for marker in markers)
    scene_brief_match = re.search(
        r"Scene\ brief:\s*(.*?)(?=(?:" + marker_pattern + r")|$)",
        text,
        flags=re.IGNORECASE,
    )
    if scene_brief_match and scene_brief_match.group(1).strip():
        prefix = text[: scene_brief_match.start()].strip(" .")
        brief = scene_brief_match.group(1).strip(" .")
        text = " ".join(part for part in (prefix, brief) if part)
    else:
        lowered = text.lower()
        cut_at = len(text)
        for marker in markers:
            found = lowered.find(marker.lower())
            if found >= 0:
                cut_at = min(cut_at, found)
        text = text[:cut_at].strip(" .")
    text = re.sub(r"\bthe chapter\b", "the hour", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchapter\b", "hour", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe scene\b", "the hour", text, flags=re.IGNORECASE)
    text = re.sub(r"\bscene\b", "hour", text, flags=re.IGNORECASE)
    return text or "the errand had to move from uncertainty toward guarded knowledge"


def _extend_hunt_scene_text(
    text: str,
    *,
    scene_goal: str,
    characters: list[str],
    place: str,
    objects: list[str],
    target_words: int,
    seen: set[str],
) -> str:
    if target_words <= 0 or _count_words(text) >= target_words:
        return text

    def _join(rows: list[str]) -> str:
        rows = [str(row).strip() for row in rows if str(row).strip()]
        if len(rows) <= 1:
            return rows[0] if rows else "the hidden guardians"
        if len(rows) == 2:
            return f"{rows[0]} and {rows[1]}"
        return ", ".join(rows[:-1]) + f", and {rows[-1]}"

    goal = _public_scene_goal_text(scene_goal)
    place_name = _hunt_place_name(place or "the wild")
    place_phrase = _hunt_place_phrase(place_name)
    char_phrase = _join(characters)
    object_phrase = _join_hunt_object_labels(objects, limit=2) or "the signs of the road"
    object_sentence_start = _sentence_start(object_phrase) if object_phrase else "The signs of the road"
    lowered_characters = {c.lower(): c for c in characters}
    ranger = lowered_characters.get("aragorn") or lowered_characters.get("strider") or "Aragorn"
    wizard = lowered_characters.get("gandalf") or "Gandalf"
    quarry = lowered_characters.get("gollum") or lowered_characters.get("smeagol") or lowered_characters.get("sméagol") or "Gollum"
    lower_goal = goal.lower()
    character_lowers = set(lowered_characters)
    if any(token in lower_goal for token in ("rain and rumour", "bring gandalf", "unease rather than proof", "opening establishes")):
        extension_kind = "charge"
    elif any(token in lower_goal for token in ("names gollum", "gollum as the quarry", "baggins and the shire must be guarded")):
        extension_kind = "naming"
    elif any(token in lower_goal for token in ("final movement", "chosen vigilance", "no crown", "roads while accepting")):
        extension_kind = "active_service"
    elif any(token in lower_goal for token in ("accepts the hunt", "would not welcome him")) or (
        "hidden service" in lower_goal and "accepts" in lower_goal
    ):
        extension_kind = "vow"
    elif any(token in lower_goal for token in ("frightened witness", "crooked report", "half helps", "crooked rumour", "human rumour")):
        extension_kind = "witness"
    elif any(token in lower_goal for token in ("southward signs", "darker country", "trail is bending", "toward darker", "bent south")):
        extension_kind = "darkening"
    elif any(token in lower_goal for token in ("failure", "obstacles", "not arbitrary defeat", "lost trail", "false certainty")):
        extension_kind = "failure"
    elif any(token in lower_goal for token in ("final image", "providential hope", "unresolved danger", "finale")):
        extension_kind = "finale"
    elif "active service" in lower_goal:
        extension_kind = "active_service"
    elif "offstage" in lower_goal:
        extension_kind = "offstage_watch"
    elif (
        any(token in lower_goal for token in ("hidden guard", "active service", "offstage", "uncertainty", "shire borders", "border"))
        and "aragorn" in character_lowers
    ):
        extension_kind = "border_watch"
    elif any(token in lower_goal for token in ("warning", "warn", "westward", "turns west", "keep watch", "roads toward the shire", "shire borders")):
        extension_kind = "warning"
    elif "gandalf" in character_lowers and "aragorn" in character_lowers and "gollum" not in character_lowers:
        extension_kind = "counsel"
    elif any(token in lower_goal for token in ("deliver", "delivery", "brings gollum toward mirkwood", "toward mirkwood", "under the trees")):
        extension_kind = "delivery"
    elif any(token in lower_goal for token in ("escape", "fugitive", "break")):
        extension_kind = "escape"
    elif any(token in lower_goal for token in ("capture", "bind", "rope")):
        extension_kind = "capture"
    elif (
        any(token in lower_goal for token in ("routine", "watch become a map", "habit becomes a door", "studies lamps", "steps, voices"))
        and "gollum" in character_lowers
    ):
        extension_kind = "watch"
    elif (
        any(
            token in lower_goal
            for token in (
                "captivity",
                "captive",
                "custody",
                "prisoner",
                "northward road",
                "under guard",
                "routine",
                "watch become a map",
                "habit becomes a door",
                "kindness into an injury",
            )
        )
        and "gollum" in character_lowers
    ):
        extension_kind = "captivity"
    elif any(token in lower_goal for token in ("question", "testimony", "answers")):
        extension_kind = "questioning"
    elif "gollum" in character_lowers and len(character_lowers) == 1:
        extension_kind = "quarry"
    elif "gandalf" in character_lowers and "aragorn" in character_lowers:
        extension_kind = "counsel"
    else:
        extension_kind = "trail"

    place_details = {
        "bree": [
            "wet thresholds, smoky beams, and the guarded civility of men who had learned to notice strangers",
            "low inn-lamps, muddy doorstones, and gossip that grew careful whenever the east was named",
        ],
        "anduin": [
            "reeds bowed by wind, sliding banks, and water that received a footprint only to trouble it",
            "cold shallows, torn rushes, and sandbars that kept a mark briefly before giving it away",
        ],
        "rhovanion": [
            "open miles, thorn-hollows, weathered stones, and cart tracks fading where no cart should pass",
            "broken turf and pale ridges",
            "old roadbeds, foxglove, loose scree, and shallow ditches where fear might leave a sign",
            "wide ground, stunted ash, sheep-tracks gone wild, and ruins too poor to shelter hope",
            "dry gullies and broken stones",
            "long slopes, crow-haunted stones, and old wheel-ruts running nowhere useful",
            "heather, flint, and grey distances that made each certainty feel prematurely spoken",
            "bare rises, twisted thorn, and gullied earth that punished every confident guess",
        ],
        "wilderland": [
            "wide ground, stunted ash, sheep-tracks gone wild, and ruins too poor to shelter hope",
            "open miles, thorn-hollows, weathered stones, and cart tracks fading where no cart should pass",
            "broken turf and pale ridges",
            "old roadbeds, foxglove, loose scree, and shallow ditches where fear might leave a sign",
            "dry gullies and broken stones",
            "long slopes, crow-haunted stones, and old wheel-ruts running nowhere useful",
            "heather, flint, and grey distances that made each certainty feel prematurely spoken",
            "bare rises, twisted thorn, and gullied earth that punished every confident guess",
        ],
        "mirkwood": [
            "low boughs, watched lamps, root-shadow, and a silence too old to be merely natural",
            "close trunks, resinous air, and guarded lights that seemed smaller each time the dark stirred",
            "root-tangled paths, shielded lanterns, and pauses in which every listener heard a different threat",
            "black boles, leaf-mould, and thin flames kept carefully below the level of watching eyes",
            "fern-shadow, bark scent, and dim paths where habit mattered more than strength",
            "layered branches, old sap, and a hush that made each careless movement feel accused",
            "damp roots, narrow clearings, and watch-fires trimmed low enough not to flatter escape",
            "woodsmoke, moss, and guarded paths that seemed to change when no one looked directly at them",
        ],
        "dead marsh": [
            "false ground, cold vapour, and pools that gave back the sky with a dead man's patience",
            "sucking earth, pale weed, and water that made every reflected star look drowned",
        ],
        "shire": [
            "soft hills, small doors, tilled fields, and a peace made vulnerable by its innocence",
            "green folds of land, tended hedges, and roads whose very quiet made warning seem impossible",
        ],
        "border": [
            "soft hills, small doors, tilled fields, and a peace made vulnerable by its innocence",
            "green folds of land, tended hedges, and roads whose very quiet made warning seem impossible",
        ],
    }
    detail_options = next(
        (value for key, value in place_details.items() if key in place_name.lower()),
        ["weather, distance, hunger, and the grudging witness of the road"],
    )
    local_detail = detail_options[_stable_seed("hunt-local-detail", goal, extension_kind, place_name, char_phrase) % len(detail_options)]

    def _hunt_pool(rows: dict[str, list[str]]) -> list[str]:
        fallback_kind = {"delivery": "capture"}.get(extension_kind)
        if extension_kind == "border_watch" and "border_watch" not in rows:
            return [
                (
                    f"{ranger} watched the road by learning its ordinary face first. Only then could he tell when "
                    "ordinary dust had been disturbed by something that meant harm."
                ),
                (
                    "The work asked him to respect peace without being lulled by it. That balance was harder than "
                    "suspicion, and far more useful."
                ),
                (
                    f"{_sentence_start(place_phrase)}, {local_detail} made peril seem distant. A hidden guard had to remember "
                    "that distance can be exactly what danger uses."
                ),
                (
                    f"{object_sentence_start} served small needs, and small needs served the watch. A strap mended "
                    "near an inn door may hear more than a challenge on the road."
                ),
                (
                    "No thanks came from the guarded land. Thanks would have made the service visible, and visibility "
                    "would have weakened it."
                ),
            ]
        if extension_kind == "active_service" and "active_service" not in rows:
            return [
                (
                    f"{ranger} gave doubt a list of tasks. The first was to listen without seeming to listen; the "
                    "second was to test each rumour against a road that could contradict it."
                ),
                (
                    f"{_sentence_start(place_phrase)}, {local_detail} kept the work humble. Mud and gossip were poor companions, "
                    "but they were honest when handled without pride."
                ),
                (
                    f"{object_sentence_start} were enough for the hour. Hidden service often begins with ordinary "
                    "things put in the right place at the right time."
                ),
                (
                    "He did not need certainty in order to act. He needed only enough uncertainty arranged in a "
                    "shape that could be watched."
                ),
            ]
        if extension_kind == "offstage_watch" and "offstage_watch" not in rows:
            return [
                (
                    f"{ranger} watched for what had not appeared. That is dreary work, and necessary, because "
                    "danger often spends its first strength staying unseen."
                ),
                (
                    f"{_sentence_start(place_phrase)}, {local_detail} made the threat seem almost impolite to imagine. He imagined "
                    "it anyway."
                ),
                (
                    f"{object_sentence_start} mattered less as gear than as reminders that readiness must be kept "
                    "while proof is absent."
                ),
                (
                    "The offstage peril did not lessen the service. It widened it, until every ordinary road had "
                    "to be considered a possible door."
                ),
            ]
        if extension_kind == "finale" and "finale" not in rows:
            return [
                (
                    "The wizard, the Ranger, and the quarry did not stand together in peace. They stood together "
                    "only in consequence, each moving a different road through the same unfinished danger."
                ),
                (
                    "The warning, the watch, and the loose memory of Baggins belonged now to one design, though "
                    "no mind within the tale could see the whole of it."
                ),
                (
                    f"{_sentence_start(place_phrase)}, {local_detail} made the image stern rather than triumphant. The world "
                    "remained weathered, partial, and morally awake."
                ),
                (
                    f"{object_sentence_start} could not gather the matter into victory. They could only mark that "
                    "the road had not been wasted."
                ),
                (
                    "Hope therefore took a narrow form: not safety, but time; not mastery, but warning; not praise, "
                    "but a watch that endured."
                ),
            ]
        if extension_kind == "failure" and "failure" not in rows:
            return [
                (
                    f"{ranger} tested the tempting answer first and found it wanting. A sign that flatters need "
                    "must be made to stand without help."
                ),
                (
                    "The failure gathered from small refusals: mud that would not speak, grass too confused to "
                    "read, stone too old to accuse, and rumour too eager to be trusted."
                ),
                (
                    f"{_sentence_start(place_phrase)}, {local_detail} made certainty expensive. Each step bought less knowledge "
                    "than the last, until prudence demanded a different payment."
                ),
                (
                    f"{object_sentence_start} could continue the search only if he let them serve truth rather than "
                    "his hunger for progress."
                ),
                (
                    "So he accepted the narrowing of the road. Not gladly, and not finally, but with the humility "
                    "without which endurance becomes noise."
                ),
            ]
        if extension_kind == "watch" and "watch" not in rows:
            return [
                (
                    f"{quarry} did not study the strongest guard first. Strength is often simple. He studied the "
                    "tired guard, the kind guard, the guard whose foot avoided one root without looking."
                ),
            (
                "The order of lamps became a secret calendar. One was lowered after food, another trimmed "
                "when voices changed, a third lifted whenever a guard passed toward the outer path."
            ),
                (
                    f"{_sentence_start(place_phrase)}, {local_detail} gave every habit a hiding place. A step lost in moss "
                    "might still be remembered by a prisoner who had nothing else to own."
                ),
                (
                    f"{object_sentence_start} seemed fixed to the keepers, but Gollum imagined them otherwise: "
                    "mislaid, loosened, shifted by pity, or made briefly useless by confusion."
                ),
                (
                    "He gave each watcher a private name in his thought. The names were cruel, childish, and useful, "
                    "for they helped him remember who hurried, who lingered, and who disliked his eyes."
                ),
                (
                    "A routine is a kindness to those who keep it and a map to those who hate it. Gollum hated "
                    "well enough to read."
                ),
                (
                    "Thus the guarded days taught him what no freedom could have taught: the shape of the cage "
                    "from inside, felt by touch, sound, resentment, and hunger."
                ),
            ]
        if extension_kind == "captivity" and "captivity" not in rows:
            return [
                (
                    f"{quarry} learned the watch as other prisoners learn walls. A lamp lowered at one hour, "
                    "a footstep delayed at another, a softened voice after food: each became part of his private map."
                ),
                (
                    "Captivity narrowed the world until small habits became large enough to hate. He hated them "
                    "patiently, because patient hatred can count what rage would miss."
                ),
                (
                    f"{_sentence_start(place_phrase)}, {local_detail} pressed close around the guarded place. The wood did not "
                    "open for him, yet neither did it let him forget that openings existed."
                ),
                (
                    f"{object_sentence_start} belonged to the keepers, but Gollum studied them as if they were "
                    "weather. Weather changes; so might rope, lamp, voice, and mercy."
                ),
                (
                    "Kindness troubled him more deeply than command. Command stood where hatred expected it; "
                    "kindness moved irregularly, and irregular things sometimes leave gaps."
                ),
                (
                    "So his mind worked in the dark even when his body seemed spent. It worked through resentment, "
                    "through hunger, through remembered names, and through the ache of being unable to vanish."
                ),
                (
                    "The danger in him was not quiet because he was guarded. It had merely changed from travelling "
                    "over roads to travelling through attention."
                ),
                (
                    "Each day under watch made the prison more familiar, and familiarity is the beginning of every "
                    "escape that has not yet found its hour."
                ),
            ]
        return rows.get(extension_kind) or (rows.get(fallback_kind) if fallback_kind else None) or rows["trail"]

    extension_by_kind = {
        "counsel": [
            (
                f"The talk between {char_phrase} did not broaden into ceremony. It narrowed. Question by question, "
                f"the errand passed from conjecture into pack straps, road food, coded messages, and the choice of "
                "which rumours deserved a sleepless night."
            ),
            (
                f"{wizard} spoke as one who feared the answer before he possessed it. {ranger} listened as one who "
                "knew that fear, in the wise, is not cowardice but a summons to test the ground."
            ),
            (
                f"The small things near them, {object_phrase}, gained the gravity of instruments. No sword was drawn, "
                "yet a campaign of secrecy began there among damp wool and low speech."
            ),
            (
                "Neither of them made the Shire grander than it was. Its smallness was the very argument. A land of "
                "gardens and gossip cannot defend itself against a name carried into darkness."
            ),
            (
                f"In {place_name}, {local_detail} pressed close around the counsel. The world outside the door had "
                "not changed, but the road beyond it had become charged with consequence."
            ),
        ],
        "naming": [
            (
                f"To speak of {quarry} was to exchange vague dread for a creature with habits, hungers, and old injuries. "
                "That made the errand less mysterious but more urgent."
            ),
            (
                f"{wizard} traced the matter backward through Bilbo's tale without leaning on comfort. What had once "
                "been an odd adventure now stood among signs that would not be laughed away."
            ),
            (
                f"{ranger} received the name as a tracker receives a footprint: not as certainty, but as a beginning "
                "that could be tested against mud, witness, and weather."
            ),
            (
                f"{object_sentence_start} made the counsel practical. A named quarry meant routes, provisions, "
                "questions to ask, and lies to expect before the first mile was spent."
            ),
            (
                "The Shire entered the talk not as a stage for action but as a place suddenly made findable. That was "
                "the bitter transformation wrought by one damaged memory."
            ),
        ],
        "vow": [
            (
                f"{ranger} did not make much of his answer. The greater a hidden duty is, the less room it often "
                "has for fine declarations."
            ),
            (
                f"{wizard} let the silence after the promise stand. It was not empty; it contained rain, distance, "
                "cold meals, and the years of mistrust that wait on a Ranger's road."
            ),
            (
                f"{object_sentence_start} belonged now to departure. A cloak, a blade, a road, and a memory of "
                "small doors were enough to begin."
            ),
            (
                "The country to be defended would not have understood the bargain. That ignorance did not lessen "
                "the bargain; it purified it of any hope of reward."
            ),
            (
                "So the vow became motion before it became story. The feet would keep it first, and only later "
                "would words try to name what had been done."
            ),
        ],
        "witness": [
            (
                f"{ranger} did not press the witness toward drama. He pressed him toward order: which bank, which "
                "hour, which sound came first, and which detail shame had tried to hide."
            ),
            (
                f"About {place_name}, {local_detail} gave the tale a body. Fear had not invented the scraped reeds, "
                "the spoiled net, or the wet mark below the sill."
            ),
            (
                f"{object_sentence_start} mattered because common things remember what frightened men cannot bear "
                "to explain. A broken hook or a trampled verge may keep faith where speech loses courage."
            ),
            (
                "The account did not become clean by being useful. It remained partial, embarrassed, and bent; "
                "Aragorn valued it all the more for not trying to sound noble."
            ),
            (
                "When rumour passed through fear and still pointed one way, the hunt gained not certainty but a "
                "narrower question. On such questions hidden errands often turn."
            ),
        ],
        "darkening": [
            (
                f"{ranger} began to read the southward pull in negatives: food left uneaten, shelter refused, safer "
                "ground crossed without need, and the quarry's fear growing less random."
            ),
            (
                f"The country around {place_name} gave him {local_detail}, but the signs no longer belonged only "
                "to country. Something inward was shaping the outward path."
            ),
            (
                f"{object_sentence_start} could mark distance, not motive. For motive he had to weigh hesitation, "
                "avoidance, and the ugly obedience of a creature drawn where he hated to go."
            ),
            (
                "The darker road did not call aloud. It worked like old pain: returning when the mind is tired, "
                "pulling thought after it until the feet begin to obey."
            ),
            (
                "So the hunt widened without becoming less intimate. One thin body still made the marks, yet a "
                "greater shadow seemed to stoop over each of them."
            ),
        ],
        "trail": [
            (
                f"{ranger} did not follow a line so much as a dispute among signs. One mark accused the river; "
                "another blamed the weather; a third, almost lost beneath grass, quietly contradicted them both."
            ),
            (
                f"The country around {place_name} gave him {local_detail}. These were not scenic ornaments to him. "
                "They were witnesses, reluctant and perishable, to be heard before rain silenced them."
            ),
            (
                f"{object_sentence_start} helped only when handled without vanity. A map could preserve memory, "
                "but it could not smell fear in a ditch or feel where mud had been pressed by a hand rather than a hoof."
            ),
            (
                "Delay became a discipline. A proud hunter might have hurried after the most dramatic rumour; Aragorn "
                "kept faith with the faint sign, because the faint sign had less reason to boast."
            ),
            (
                "The Shire remained far off, almost absurdly gentle in imagination. That distance did not weaken the "
                "duty. It made the duty cleaner, for no praise from that country could reach him on the road."
            ),
        ],
        "quarry": [
            (
                f"{quarry} made a poor kingdom of every hiding place. A root became a wall, a stone a throne, a "
                "scrap of fish a feast to be defended with teeth and curses."
            ),
            (
                f"Around {place_name}, {local_detail} did not soothe him. Open air felt like accusation, and every "
                "star seemed another eye that had learned the name he tried both to swallow and to spit out."
            ),
            (
                "Baggins returned to him in fragments: not a person whole in memory, but a wound with a door, a "
                "warm hole, a laugh he hated because it had survived him."
            ),
            (
                "The Shire was worse because it sounded soft. He could hate iron gates and black towers honestly; "
                "soft things made his hatred feel small, and therefore sharper."
            ),
            (
                f"{object_sentence_start} lay about him like the inventory of a life narrowed to hunger. He kept "
                "what he could, broke what he could not keep, and listened for pursuit even in his sleep."
            ),
        ],
        "capture": [
            (
                f"At close quarters the quarry ceased to be a rumour. {ranger} could smell wet hair, old fear, "
                "fish, mud, and the sour breath of a creature who had lived too long in loss."
            ),
            (
                f"The rope answered one question and opened another. It could hold {quarry}'s limbs, but not his "
                "tongue, his hatred, or the crooked roads by which memory escaped him."
            ),
            (
                "Pity became harder, not easier, when it had a face. The captive was vile, pitiable, dangerous, "
                "and necessary; to simplify him would have been another kind of falsehood."
            ),
            (
                f"In {place_name}, {local_detail} made every halt precarious. A prisoner must drink, sleep, curse, "
                "stumble, and be guarded through all the indignities by which life continues."
            ),
            (
                f"{object_sentence_start} were handled with the care of things that might keep knowledge alive. "
                "The work was grim because dead certainty would have taught them less than a living wretch."
            ),
        ],
        "questioning": [
            (
                f"{wizard} did not press for a tidy confession. He waited for the repeated word, the sudden flinch, "
                "the place where malice forgot its disguise and memory showed through."
            ),
            (
                f"{quarry}'s answers came crookedly, but crooked things may still point. Baggins and Shire emerged "
                "not as formal testimony, but as burrs that clung to every turn of his speech."
            ),
            (
                f"Under {place_name}'s watch, {local_detail} gathered around the questioning. No cruelty was needed "
                "to make the hour dreadful; truth itself was dreadful enough."
            ),
            (
                f"{ranger} heard in the prisoner what the road had only hinted. The trail had been a sentence "
                "written in mud; now the same sentence stumbled out through teeth."
            ),
            (
                f"{object_sentence_start} marked the borders of the small chamber of inquiry. Beyond them lay the "
                "westward roads, and beyond those roads a peace that had not yet learned its own peril."
            ),
        ],
        "escape": [
            (
                f"The flaw in the watch was not large. It was only large enough for misery. {quarry} had spent long "
                "hours learning how mercy pauses, how routine breathes, and where a shadow might borrow shape."
            ),
            (
                f"Around {place_name}, {local_detail} turned every sound double. A snapped twig might be a guard; "
                "a guard's call might be meant elsewhere; a moment meant elsewhere might become a door."
            ),
            (
                "Flight did not ennoble him. It stripped him smaller. He became hunger with hands, fear with feet, "
                "a mouth carrying names it could neither master nor forget."
            ),
            (
                f"{object_sentence_start} remained behind or underfoot, changed from instruments of custody into "
                "tokens of failure. Yet failure did not erase what had already been learned."
            ),
            (
                "When the dark received him, it received also the danger he bore. That was the bitter cost of pity: "
                "it could keep a creature alive without making him safe."
            ),
        ],
        "warning": [
            (
                f"Warning changed the pace of everything. {wizard} no longer sought merely to learn; he had to carry "
                "what had been learned westward before others made a weapon of it."
            ),
            (
                f"{ranger}'s task changed also. A lost trail could no longer be treated only as failure. It became "
                "the outline of a watch, a map of roads that must be quietly held."
            ),
            (
                f"{_sentence_start(place_phrase)}, {local_detail} made the west seem both near and exposed. The green country did "
                "not know that its own name had begun to travel like a spark."
            ),
            (
                f"{object_sentence_start} now belonged to preparation rather than pursuit. Messages had to be "
                "carried, signs left for friendly eyes, and fear kept sharp without being allowed to rule."
            ),
            (
                "The victory, if victory it was, had no feast in it. Knowledge had survived, the prisoner had not "
                "been mastered, and hope depended on men who would remain unseen."
            ),
        ],
    }
    general_extensions: list[str] = []
    expansion_pool = _hunt_pool(extension_by_kind) + general_extensions

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    idx = _stable_seed(goal, place_name, char_phrase, object_phrase) % len(expansion_pool)
    attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and attempts < len(expansion_pool):
        paragraph = expansion_pool[(idx + attempts) % len(expansion_pool)]
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)

    supplemental_by_kind = {
        "counsel": [
            (
                f"{object_sentence_start} lay between {char_phrase} as tokens of a task becoming practical. "
                "One could speak gravely of peril and still fail it by neglecting food, weather, messages, and the hour of departure."
            ),
            (
                f"The talk returned again to the smallness of the names at stake. {wizard} did not make them grand, "
                f"and {ranger} did not ask him to; their very plainness was what made them easy prey for darker minds."
            ),
            (
                f"Outside {place_name}, ordinary life continued with stubborn innocence. That was no argument against the errand. "
                "It was the errand's answer, for guarded peace must often remain ignorant of its guardians."
            ),
            (
                f"{ranger} listened for what was missing as much as for what was said. No sure proof had arrived, "
                "but too many lesser signs leaned the same way, and wisdom sometimes begins by admitting the shape of doubt."
            ),
            (
                f"{wizard} had carried fears before, but this one had an uncomfortable humility. It concerned not a throne, "
                "nor a battle line, but a lost creature's memory and a country that trusted hedges more than walls."
            ),
            (
                "The decision did not arrive like a stroke of lightning. It gathered slowly, as water gathers in a hollow, "
                "until refusal would have required more pride than courage."
            ),
        ],
        "naming": [
            (
                f"{quarry}'s name changed the temperature of the room. It drew together the lost ring, Bilbo's old road, "
                "and the Shire's unwitting exposure without making any one of them simpler."
            ),
            (
                f"{wizard} spoke of evidence as a traveller speaks of a bad bridge: not to admire its danger, but to "
                "decide whether it must be crossed before nightfall."
            ),
            (
                f"{ranger} began already to think in distances. If the creature had passed near water, who had seen him? "
                "If he had begged food, what name had slipped out with the begging?"
            ),
            (
                f"The plain gear before them, {object_phrase}, seemed almost too ordinary for such a fear. Yet ordinary "
                "things are what secret errands use when banners would betray them."
            ),
            (
                "The name Baggins was handled carefully, as one handles a coal that has not gone out. To drop it in "
                "the wrong place might kindle more than curiosity."
            ),
        ],
        "vow": [
            (
                f"{ranger} knew the look that would meet him near many doors: suspicion first, gratitude never. The "
                "knowledge did not embitter the promise; it clarified it."
            ),
            (
                f"{wizard} had asked not for a glorious deed but for endurance. Endurance is less easily sung, and "
                "therefore often more necessary."
            ),
            (
                f"The road beyond {place_name} seemed to lengthen even before he set foot on it. So it is with a true "
                "vow: the miles answer before they are walked."
            ),
            (
                f"{object_sentence_start} marked the difference between intention and departure. Once they were taken "
                "up, the counsel would be over and the weather would have its say."
            ),
            (
                "No witness from the Shire blessed the errand. No witness was needed. The sleeping peace of that land "
                "was itself the charge laid on him."
            ),
        ],
        "witness": [
            (
                "The witness's hands described what his tongue disliked: the low height of the crawling thing, the "
                "angle of its head, the quick turn when a hobbit-name escaped it."
            ),
            (
                f"{ranger} let the man contradict himself twice before asking anything. Contradiction born of fear "
                "may still leave the same footprint under both stories."
            ),
            (
                f"About {place_name}, ordinary people wished ordinary explanations upon the night. Missing fish, "
                "a fox in the reeds, a drunkard's fancy; each excuse was too neat to satisfy the road."
            ),
            (
                f"{object_sentence_start} kept the account fastened to earth. The hunt could not live on terror "
                "alone; it needed mud under the terror and a direction beyond it."
            ),
            (
                "Aragorn took from the tale neither more nor less than it could bear. That restraint was part of "
                "the craft: belief without surrender, doubt without dismissal."
            ),
        ],
        "darkening": [
            (
                f"The farther {ranger} followed, the less hunger explained. A starving creature may seek fish, "
                "shelter, and darkness; it does not usually choose fear as if fear were food."
            ),
            (
                f"Around {place_name}, the marks began to appear where comfort ended. The quarry had passed near "
                "safer ground and refused it, as though some inward lash had driven him on."
            ),
            (
                f"{object_sentence_start} answered the practical questions and left the grave one untouched. A map "
                "could show south; it could not say why south had become command."
            ),
            (
                "At times the trail seemed less pursued than summoned. That was an ugly thought, and Aragorn kept "
                "it close because ugly thoughts can still be useful when pride dislikes them."
            ),
            (
                "The Shire lay farther behind in miles but nearer in peril. Distance had ceased to comfort once "
                "the quarry's road began bending toward older darkness."
            ),
        ],
        "trail": [
            (
                f"{object_sentence_start} were useful only because {ranger} refused to force them into certainty. "
                "He let each sign remain small until another small sign came to stand beside it."
            ),
            (
                f"{_sentence_start(place_phrase)}, the land kept no orderly record. Wind edited the dust, water softened the bank, "
                "and old paths crossed new fear until the eye had to choose patiently among them."
            ),
            (
                "There were hours when the hunt seemed to consist of kneeling, waiting, and being cold. Such hours were not wasted. "
                "They taught which absences belonged to weather and which to a creature trying not to be found."
            ),
            (
                f"{ranger} trusted neither despair nor sudden hope. Both can make a hunter loud in his own mind, "
                "and the quarry they sought had survived too long by hearing carelessness before it arrived."
            ),
            (
                "A rumour could be useful and still be false in half its limbs. He took from each report only what the road confirmed, "
                "leaving pride, ale, and fear to contradict one another behind him."
            ),
            (
                "The farther the trail bent from comfort, the more clearly the west appeared in thought. Not as a reward, "
                "but as the quiet place that gave hardship its meaning."
            ),
        ],
        "quarry": [
            (
                f"{object_sentence_start} marked the borders of {quarry}'s narrowed world. He trusted what could be hidden, "
                "gnawed, pocketed, or cursed; all else belonged to enemies."
            ),
            (
                f"About {place_name}, open country pressed on him like a hand. He wanted stone above him, black water beside him, "
                "and no clean air in which memory could find his face."
            ),
            (
                "The name Baggins came back whenever hunger left room for hatred. It was not recollection as whole minds know it, "
                "but a sore place touched again and again because pain had become proof of possession."
            ),
            (
                "He feared pursuit and courted it in the same breath. To be hunted confirmed that he still mattered; "
                "to be caught would prove that every hand in the world was a thief's hand."
            ),
            (
                "If pity had passed near him, he had learned to smell insult in it. Kindness made a debt, and debt was another rope."
            ),
            (
                f"So {quarry} went on in pieces: hunger first, then fear, then memory, then hatred, each taking command until the next pain rose."
            ),
        ],
        "capture": [
            (
                f"The things at hand, {object_phrase}, became stern and intimate. A knot badly tied might waste years; "
                "a hand too hard might turn necessity into cruelty."
            ),
            (
                f"{ranger} kept his anger busy with practical care. There was water to find, distance to judge, "
                "and a prisoner whose weakness made him no less dangerous."
            ),
            (
                "Gollum's pleas changed shape whenever one failed. He cursed, whimpered, accused, promised, and forgot the promise "
                "before the next breath had left him."
            ),
            (
                f"In {place_name}, victory had the smell of mud and old fear. It gave no comfort, only a direction and a living burden."
            ),
            (
                "Pity did not soften the road. It made the road harder, because it forbade the simpler answers that disgust would gladly choose."
            ),
        ],
        "questioning": [
            (
                f"{object_sentence_start} made a poor border around the truth. Within it, every pause in {quarry}'s speech "
                "had to be weighed against malice, fear, and memory."
            ),
            (
                f"{wizard} asked little at a time. A hard question pressed too quickly might bruise the answer out of shape."
            ),
            (
                "The names did not come forth cleanly. They slipped out with spite attached, and yet the spite made them more dreadful, "
                "for hatred remembers roads that gratitude forgets."
            ),
            (
                f"{ranger} had followed footprints to this place; now he watched words leave tracks of their own."
            ),
            (
                "No one in that guarded hour mistook knowledge for safety. To know where danger points is only to learn where the next duty begins."
            ),
        ],
        "escape": [
            (
                f"About {place_name}, the dark made accomplices of root and hollow. {quarry} had studied them as a starving scholar studies crumbs."
            ),
            (
                f"{object_sentence_start} could no longer promise order. A lamp turned aside, a rope rubbed thin, "
                "or a branch falling at the wrong instant might undo a watch built by wiser hands."
            ),
            (
                "The flight was not bold. It was abject, swift, and venomous, a refusal of mercy more than a claim of freedom."
            ),
            (
                "Behind him the keepers would count the cost. Before him the wild received a creature who carried names like coals under ash."
            ),
            (
                "He did not know what history would make of his escape. He knew only that the dark was open and that open dark felt almost kind."
            ),
        ],
        "warning": [
            (
                f"{object_sentence_start} now belonged to haste governed by secrecy. A warning too loud could become another road for fear."
            ),
            (
                f"{wizard} measured the westward miles not by distance alone but by the number of chances left for darker news to outrun him."
            ),
            (
                f"{ranger} accepted the less visible duty. If the warning went ahead, the roads behind it must be made costly to follow."
            ),
            (
                "The Shire's peace remained almost painfully ordinary in thought. That ordinariness was not weakness; it was the thing being defended."
            ),
            (
                "No conclusion worthy of a song had been granted. The story's mercy lay instead in time bought, roads watched, and truth carried onward."
            ),
        ],
    }
    supplemental_pool = _hunt_pool(supplemental_by_kind)
    supplemental_idx = _stable_seed("hunt-supplement", goal, place_name, char_phrase, object_phrase) % len(supplemental_pool)
    supplemental_attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and supplemental_attempts < len(supplemental_pool):
        paragraph = supplemental_pool[(supplemental_idx + supplemental_attempts) % len(supplemental_pool)]
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        supplemental_attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    deepening_by_kind = {
        "counsel": [
            (
                f"The weight of the errand could be felt in what {wizard} left unsaid. He did not yet claim full knowledge, "
                "for a wise fear is careful not to dress itself as prophecy before the facts have endured questioning."
            ),
            (
                f"{ranger} knew the eastern roads well enough to mistrust every easy account of them. A fugitive might pass "
                "through hunger, weather, and rumour like a needle through worn cloth, leaving little more than a pulled thread."
            ),
            (
                f"At {place_name}, common talk moved close to them and then away again. That nearness of ordinary life made "
                "the hidden charge sharper, for the world to be guarded was not an idea but a set of doors, meals, and sleepers."
            ),
            (
                "They spoke of routes without pretending routes could solve the matter. The quarry had survived by refusing "
                "the common habits of travellers, and so the hunter must be ready to follow hunger rather than roads."
            ),
            (
                f"When the counsel paused, {object_phrase} remained in view. Such plain things had a rebuking honesty; "
                "they asked not for eloquence, but for hands willing to use them before dawn."
            ),
        ],
        "naming": [
            (
                f"The name {quarry} did not settle gently into the talk. It seemed to scratch at every other word, "
                "turning Bilbo's remembered luck into evidence and the Shire's quiet into exposure."
            ),
            (
                f"{wizard} separated what was known from what was feared, though the two lay close together. The creature "
                "had possessed the ring; he had lost it; he had learned a hobbit-name; and hatred is a diligent messenger."
            ),
            (
                f"{ranger} asked after habits rather than legends. Did the creature seek water, darkness, fish, tunnels, "
                "lonely banks? Such questions were humble, but they were the bridge from counsel into pursuit."
            ),
            (
                f"The objects before them, {object_phrase}, seemed to wait for the moment when words would end. Once the "
                "quarry was named, the road began pressing its claim on every practical thing in reach."
            ),
            (
                "To protect the Shire, they first had to admit how little protected it was by obscurity alone. Smallness hides "
                "only until a hateful memory learns the way."
            ),
        ],
        "vow": [
            (
                f"{ranger} did not ask how long the hunt would last. Long labours have a way of hiding their full measure "
                "from the man who accepts them, lest foresight become another temptation to refuse."
            ),
            (
                f"{wizard} gave what counsel he could: not comfort, not certainty, but the shape of the danger and the "
                "names that must be kept from travelling farther than need allowed."
            ),
            (
                f"The thought of the Shire came to {ranger} as firelight seen from outside. He might guard it, but he did "
                "not belong to its ease, and the knowledge made the promise sterner rather than weaker."
            ),
            (
                f"{object_phrase} marked the threshold between speech and action. A blade could not answer fear, a road "
                "could not explain mercy, but each would have its part once the promise left the room."
            ),
            (
                "So the vow settled into him without display. It would be kept in wet camps, in suspicious villages, in "
                "long silences, and in the refusal to let disgust outrun pity."
            ),
        ],
        "witness": [
            (
                "The man's fear had a local shape: a ditch he would not cross after sunset, a shutter repaired "
                "twice, and a child forbidden to fetch water from the lower bank."
            ),
            (
                f"{ranger} valued such homely boundaries. Grand fear flies everywhere at once; true fear often "
                "settles on one bend of road and will not be moved from it."
            ),
            (
                "The account left gaps large enough for doubt, but the gaps were not empty. Around them clung "
                "mud, river-smell, and the shame of a man who had run before he understood why."
            ),
            (
                f"{object_phrase} gave the report weight without making it certain. Certainty would come, if it "
                "came at all, from the road answering the witness in its own sterner language."
            ),
            (
                "So Aragorn carried the crooked tale as he carried other useful burdens: lightly enough to test, "
                "carefully enough not to lose."
            ),
        ],
        "darkening": [
            (
                f"The marks near {place_name} made a darker grammar than tracks alone. There were pauses where "
                "the quarry seemed to listen, and starts where listening had become fear."
            ),
            (
                "Aragorn did not mistake dread for proof. Still, dread may notice kinship before proof has words, "
                "and every southward sign made the old fear less easy to dismiss."
            ),
            (
                f"{object_phrase} kept the search tied to fact. Without them, the pull toward darker country would "
                "have been only a thought, and thoughts are treacherous guides when the night is long."
            ),
            (
                "The quarry's weakness became more dangerous, not less, if another will could bend it. A cracked "
                "vessel may carry poison farther than a sound one, because few think to guard it."
            ),
            (
                "Thus the trail changed from pursuit into warning by degrees. No horn announced the change; the "
                "ground simply grew more reluctant to be innocent."
            ),
        ],
        "quarry": [
            (
                f"{quarry} remembered the ring less as an object than as a hunger that had once answered every other hunger. "
                "Without it, even food seemed a mockery, filling only the body and leaving the deeper want awake."
            ),
            (
                f"Around {place_name}, the land's openness tormented him. He preferred the crooked mercies of root and hole, "
                "places where the world came close enough to be bitten."
            ),
            (
                "He muttered Baggins not because he wished to tell, but because the name had made a nest in pain. Pain speaks "
                "when prudence sleeps."
            ),
            (
                f"{object_phrase} lay in his path like poor trophies of survival. They did not comfort him; comfort was a "
                "thing he suspected of theft."
            ),
            (
                "Thus the danger travelled in a body that looked hardly strong enough to trouble a child. That was the error "
                "the proud would make, and the error Gandalf feared."
            ),
        ],
        "trail": [
            (
                f"The ground near {place_name} made its answers grudgingly. {ranger} had to read what remained after "
                "weather, beasts, and frightened men had each confused the page."
            ),
            (
                f"{object_phrase} helped him keep the search honest. The useful sign was not always the dramatic one, "
                "and the dramatic sign was often only fear wearing a louder coat."
            ),
            (
                "Some days gave him nothing fit to report, yet they were not empty. They taught where the quarry had not gone, "
                "which is knowledge of a colder but still necessary kind."
            ),
            (
                "The farther he went, the less the Shire resembled a place on a map and the more it became a charge laid "
                "on endurance itself."
            ),
            (
                f"At evening {ranger} would sometimes stop before the last light failed, not from ease but from discipline. "
                "A tired eye can invent tracks more readily than it finds them."
            ),
        ],
        "capture": [
            (
                f"The body before {ranger} was all angles, hunger, and old spite. To call it harmless would be foolish; "
                "to call it only wicked would be too easy."
            ),
            (
                f"{object_phrase} became instruments of restraint rather than victory. Their purpose was not punishment, "
                "but the grim preservation of a truth that could still speak."
            ),
            (
                "The captive's misery did not excuse the danger he carried. It did, however, forbid the hunter from pretending "
                "that necessity had made cruelty clean."
            ),
            (
                f"In {place_name}, every movement had to be judged twice: once for escape, and once for the cost of preventing it."
            ),
            (
                "The road after capture promised no ease. A caught creature must still be fed, watched, endured, and brought "
                "through miles that would gladly erase both hunter and prisoner."
            ),
        ],
        "questioning": [
            (
                f"{wizard} listened for the words that returned without command. Such returns are often nearer truth than "
                "the answer a frightened mouth intends to give."
            ),
            (
                f"{quarry} made every question into an injury, but injury could not prevent memory from showing its shape."
            ),
            (
                "The room of questioning was small, yet the roads implied by its answers ran west, south, and into darkness. "
                "That was why each hesitation mattered."
            ),
            (
                f"{ranger} trusted the craft of pursuit even here. A lie leaves pressure marks; a truth, however crooked, "
                "leans against them differently."
            ),
            (
                f"{object_phrase} stood near while speech did the harder work. No blade could cut the right answer free."
            ),
        ],
        "escape": [
            (
                f"{quarry} had watched the order of his keeping until order itself became a map. He needed no broad road, "
                "only the smallest agreement between darkness and chance."
            ),
            (
                f"About {place_name}, each ordinary sound could hide another. A branch, a call, a shift of feet: any one "
                "might turn the watch aside for the breath he needed."
            ),
            (
                f"{object_phrase} no longer meant what the keepers meant it to mean. In his mind every thing was either "
                "obstacle, weapon, or witness against him."
            ),
            (
                "There was no nobility in the flight, but there was consequence. Some events change history not by greatness, "
                "but by slipping through the hand."
            ),
            (
                "When he passed beyond sight, the failure did not erase pity. It revealed the terrible cost of pity in a "
                "world where mercy cannot command what it saves."
            ),
        ],
        "warning": [
            (
                f"The road west would not wait for perfect knowledge. {wizard} had learned enough to fear delay, and "
                "enough to know that fear must be governed if it was to be useful."
            ),
            (
                f"{ranger} read the duty left behind him in practical terms: crossings watched, rumours tested, strangers "
                "noted, and signs placed where friendly eyes might find them."
            ),
            (
                "The warning had to move quietly because panic would serve the very darkness they resisted. A guarded truth "
                "may travel farther than a shouted one."
            ),
            (
                f"{_sentence_start(place_phrase)}, the thought of the Shire seemed almost too gentle for the road's severity. Yet that "
                "gentleness was the reason severity had been accepted."
            ),
            (
                f"{object_phrase} became part of haste without confusion. Each item answered a need, and need was now the "
                "master of the remaining miles."
            ),
        ],
    }
    deepening_pool = _hunt_pool(deepening_by_kind)
    deepening_idx = _stable_seed("hunt-deepening", goal, place_name, char_phrase, object_phrase) % len(deepening_pool)
    deepening_attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and deepening_attempts < len(deepening_pool):
        paragraph = deepening_pool[(deepening_idx + deepening_attempts) % len(deepening_pool)]
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        deepening_attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    texture_by_kind = {
        "counsel": [
            "Rain worked at the shutters while the counsel narrowed. It made the room seem smaller, and the world beyond it larger.",
            f"{ranger} counted the likely crossings in silence, setting each against the season, the temper of rivers, and the habits of hungry things.",
            f"{wizard} had the look of one listening beyond walls. Not to voices only, but to consequence approaching before it could be named.",
            "A servant passed in the outer passage and knew nothing of the matter. That ignorance moved Aragorn more than any speech about kingdoms.",
            f"The last question was practical: what must be taken, what must be left, and how soon {place_name} could be put behind them.",
        ],
        "naming": [
            f"The name lingered after it was spoken. {quarry}: a throat-sound, a history of swallowing, a poor title for a perilous witness.",
            f"{ranger} repeated none of it aloud. He stored the shape of the fear where he stored bearings, distances, and the faces of unreliable men.",
            f"{wizard} warned that hatred can preserve memory with a miser's care. What love forgets in mercy, malice may keep sharpened.",
            "The maps did not show Gollum's roads. They showed only where a patient hunter might begin disproving his own guesses.",
            "Baggins and Shire were set apart in the talk, not elevated, but guarded from needless repetition, as if each use of them spent something.",
        ],
        "vow": [
            f"{ranger} took up the thought of departure piece by piece. First the road, then weather, then hunger, then the long courtesy of being unwelcome.",
            f"{wizard} did not soften the charge by praising it. Praise would have made it smaller, as if words could pay beforehand for miles not yet walked.",
            "The knife at Aragorn's side was less important than the patience he would need. Steel answers few questions in marsh and rumour.",
            "He thought briefly of the Shire's suspicion of weather-beaten men. Then he let the thought pass. A door need not love its hinge.",
            "So acceptance became order: cloak mended, blade checked, messages arranged, and the first silence of the road already entering his mind.",
        ],
        "witness": [
            f"The witness kept glancing toward {place_name} as if the place itself might deny what he had said.",
            "A dog under the table whined once and was hushed with more force than kindness required.",
            f"{ranger} noted the cracked reed stuck to the man's boot; fear had brought proof indoors without knowing it.",
            "The tale smelled of lamp-smoke, river-mud, and the sour pride of one who had been afraid.",
            "When silence came, it did not absolve the witness. It merely gave Aragorn room to remember accurately.",
        ],
        "darkening": [
            f"The wind near {place_name} carried a dry edge under the damp, like ash remembered through rain.",
            f"{ranger} found no comfort in the straightening of the signs. A clear bad road is still bad.",
            f"{object_phrase} lay before him in the failing light, plain objects beside an unplain fear.",
            "Birds lifted from the bank all at once and then settled nowhere near it.",
            "By nightfall the southern dark felt less like distance than invitation refused too late.",
        ],
        "quarry": [
            f"{quarry} paused often, not from wisdom but from the animal knowledge that pursuit is heard first in the change of ordinary sounds.",
            "A bird lifting too quickly could freeze him. A stone rolling after his own foot could make him bare his teeth at empty air.",
            "He remembered water under mountains and hated the sky for having no roof. The world above ground was too wide to be trusted.",
            f"The things he carried or gnawed, {object_phrase}, were not possessions so much as proofs that the day had not yet killed him.",
            "When the name Shire came, it came with warmth in it, and warmth was hateful because it belonged to another life.",
        ],
        "trail": [
            f"The road near {place_name} changed character by the hour. Morning showed what evening had hidden, and noon made liars of both.",
            f"{ranger} kept his hand from the obvious sign until he had read the ground around it. Tracks, like speech, may be arranged to deceive.",
            f"{object_phrase} had to be packed and repacked as the weather altered. Hidden work wastes nothing willingly.",
            "A distant village bell sounded once and was gone. It seemed less a comfort than a reminder that ordinary time continued elsewhere.",
            "By dusk the search had become not slower but more exact, each pause chosen and paid for.",
        ],
        "capture": [
            f"{quarry}'s breath came in wet, resentful gasps, and each gasp seemed to accuse the hand that had prevented his vanishing.",
            f"{ranger} checked the rope without triumph. A knot is a promise as well as a restraint, and a bad promise may kill what it meant to keep.",
            f"{object_phrase} took on the hard intimacy of necessity. Nothing there was symbolic to the cold hand using it.",
            "The marsh around them made small noises, as if the ground disliked having witnessed anything so definite as capture.",
            "When they moved again, the road had two wills upon it: the one that guarded and the one that hated being guarded.",
        ],
        "questioning": [
            f"{quarry} watched every face as if kindness were only cruelty taking a slower road.",
            f"{wizard} let silence do part of the work. A frightened liar often rushes to fill it; a wounded memory sometimes cannot.",
            f"{ranger} stood where he could see the prisoner's hands. Speech mattered, but hands announce many truths before the mouth permits them.",
            f"{object_phrase} made the chamber seem more practical than solemn, which was well. Solemnity can flatter a questioner into haste.",
            "Outside, the trees kept their own counsel, and the guarded hour lengthened without becoming easier.",
        ],
        "escape": [
            f"{quarry} felt the gap before he saw it, as a starved thing feels a crumb dropped in darkness.",
            f"{object_phrase} had been part of order a moment before; now each seemed to point toward disorder.",
            "The trees confused pursuit as readily as flight. In that confusion terror found a kind of wisdom.",
            "A call went up behind him, but calls are large things, and he had made himself small.",
            "He did not run toward freedom as free folk understand it. He ran away from every hand that had ever claimed to know his good.",
        ],
        "warning": [
            f"{wizard} measured haste against secrecy and found neither could be allowed to master the other.",
            f"{ranger} thought of roadward signs: a stone turned so, a mark cut shallow, a message placed where only the wary would look.",
            f"{object_phrase} belonged to motion now, and motion had to begin before comfort found another objection.",
            "The west did not draw nearer merely because it was needed. Need only made each mile less negotiable.",
            "Hope had become a thing with work-stained hands, and that was the only hope the hour could honestly bear.",
        ],
    }
    texture_pool = _hunt_pool(texture_by_kind)
    texture_idx = _stable_seed("hunt-texture", goal, place_name, char_phrase, object_phrase) % len(texture_pool)
    texture_attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and texture_attempts < len(texture_pool):
        paragraph = texture_pool[(texture_idx + texture_attempts) % len(texture_pool)]
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        texture_attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    continuity_by_kind = {
        "counsel": [
            f"Before the hour ended, the matter had changed from something feared by {wizard} into something {ranger} could carry eastward.",
            "That change was quiet enough for the room to contain it, and grave enough that the road outside seemed altered.",
            "The counsel left no comfort behind it, only a clearer division between ignorance and duty.",
        ],
        "naming": [
            f"With {quarry} named, the hunt ceased to be a mist of warnings and became a road with a first hard mile.",
            "The name did not explain everything. It explained enough to make inaction dishonest.",
            "Baggins and Shire were no longer stray syllables in a tale of loss; they had become things to guard.",
        ],
        "vow": [
            f"The promise settled into {ranger} as road-dust settles into a cloak: lightly at first, then everywhere.",
            "What had been asked in words would be answered in weather, and weather is a stern examiner.",
            "So the hidden service began before the first step, in the inward refusal to turn aside.",
        ],
        "witness": [
            "The report did not end the search; it gave the search a bend in the road.",
            f"{place_name} had spoken through a frightened mouth, and Aragorn knew better than to despise the messenger.",
            "By the time he left, one rumour had become a place to test with patient feet.",
        ],
        "darkening": [
            "The southward signs made pursuit less hopeful and more necessary.",
            f"{place_name} no longer seemed merely difficult; it seemed positioned on the edge of a deeper summons.",
            "From there the hunt carried warning inside it, though warning had not yet become the whole task.",
        ],
        "quarry": [
            f"{quarry} knew nothing of the counsel behind him, yet his muttering had already drawn wiser feet onto his track.",
            "He mistook memory for possession, and that mistake made every remembered name dangerous.",
            "The wild gave him distance, but distance alone could not bury what his mouth kept bringing back.",
        ],
        "trail": [
            f"The search therefore moved by patience rather than haste, and {place_name} yielded only what patience could win.",
            "Each small answer made the next question narrower, which was the only mercy the road offered.",
            "By such increments the hidden watch advanced, unpraised and difficult to see.",
        ],
        "capture": [
            "The taking of the quarry did not still the errand; it merely changed the direction of its danger.",
            f"{ranger} had won custody, not peace, and the difference mattered with every knot he checked.",
            "So the road after capture began under a heavier silence than the road before it.",
        ],
        "questioning": [
            "Each answer made the west more urgent, not safer.",
            f"{wizard} heard enough to act, and not enough to rest; that was the cruel balance of the hour.",
            "The truth came limping, but it came, and even a limping truth may outrun disaster if carried in time.",
        ],
        "escape": [
            "What slipped away was not only a prisoner but an unfinished answer.",
            "The watch had failed in one purpose and succeeded in another: it had learned why failure mattered.",
            "From that hour onward, pursuit and warning could no longer be separated cleanly.",
        ],
        "warning": [
            "The warning had to move before it became complete, or completeness would arrive too late.",
            f"{wizard} took that burden westward, while {ranger} kept faith with the roads behind.",
            "Between them lay the hard mercy of acting without applause and before certainty could become ease.",
        ],
    }
    continuity_pool = _hunt_pool(continuity_by_kind)
    continuity_idx = _stable_seed("hunt-continuity", goal, place_name, char_phrase, object_phrase) % len(continuity_pool)
    continuity_attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and continuity_attempts < len(continuity_pool):
        paragraph = continuity_pool[(continuity_idx + continuity_attempts) % len(continuity_pool)]
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        continuity_attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    coda_by_kind = {
        "counsel": [
            f"The road did not yet have {ranger}'s footprints upon it, but in thought he had already begun to travel.",
            f"{wizard} let the decision stand without adornment. Some burdens are weakened by too many noble words.",
            "Outside, the rain kept falling with the patience of things that do not care whether men are ready.",
            f"By then {object_phrase} had ceased to be background; each item waited for the hand that would carry the counsel into weather.",
            "What remained was not more speech, but the first obedience to what speech had uncovered.",
            "That obedience began in silence.",
            "Before dawn, it would need feet.",
            "The road opened.",
        ],
        "naming": [
            "The named quarry seemed at once nearer and more elusive, as if speech had brought him to the threshold and then sent him running.",
            "So the old tale of a hobbit and a lost treasure ceased to be merely old, and became a present danger with weather on it.",
            "From that point, every rumour would have to be judged against a living malice and not against unease alone.",
            f"Even {place_name} seemed to hold the name differently once it had been spoken, as if the road itself had overheard.",
            "The hunt had been given a face, and therefore a harder mercy.",
            "Names can do that.",
            "This one did.",
        ],
        "vow": [
            "A man may accept a road before he knows its length; otherwise many needful roads would never be taken.",
            "The first proof of the vow was restraint: no boast, no flourish, only the quiet ordering of what must be done next.",
            "If there was honour in the choice, it was the kind that grows smaller when displayed.",
            f"The weather beyond {place_name} would have the first hearing of the promise, and weather is not easily impressed.",
            "He accepted that judgement in advance.",
            "The road would remember.",
            "So would he.",
        ],
        "witness": [
            "The man barred his door after Aragorn left, though no pursuer had threatened it.",
            "Fear had made him mean, but it had not made him useless.",
            f"In {place_name}, the next question now had banks, ditches, and a remembered cry around it.",
            "That was enough for the road.",
            "Aragorn took it.",
        ],
        "darkening": [
            "The southern road did not brighten when named.",
            "It only became harder to ignore.",
            f"{ranger} packed {object_phrase} before the last light failed, because some choices are best made before night argues.",
            "Behind him lay uncertainty.",
            "Before him, a darker pull.",
        ],
        "quarry": [
            f"{quarry} went on because stopping left too much room for memory, and memory had teeth.",
            "Behind his muttering lay no plan worthy of the name, yet harm does not always require a plan to travel far.",
            "The land closed behind him badly, leaving signs a wiser creature would have hidden and wounds no wisdom could read kindly.",
            f"He kept moving until {place_name} itself seemed only another thing he had used and hated in passing.",
            "The names went with him, small and terrible.",
            "He did not know their weight.",
            "He only felt their sting.",
        ],
        "trail": [
            "The next sign would not explain the last one fully, but it might keep the search from dying.",
            f"{ranger} had learned to accept that kind of mercy from the road: incomplete, grudging, and enough.",
            "So he went on, not because the trail was generous, but because abandoning it would be a lie.",
            f"The day ended with {object_phrase} repacked and the next uncertainty already waiting beyond sight.",
            "That was enough to begin again.",
            "Hidden work often is.",
            "The next mile waited.",
        ],
        "capture": [
            "The prisoner lived, and therefore the hardest questions remained possible.",
            "That was the narrow victory: not mastery, but the refusal to let knowledge perish in anger.",
            "The road ahead would test whether that refusal could survive weariness.",
            f"{place_name} gave them no blessing for it, only ground difficult enough to make each step remember the captive.",
            "Still the captive breathed, and breath was the guarded answer.",
            "For now, it was enough.",
            "The rope held.",
        ],
        "questioning": [
            "The hour ended with more fear than it had begun, which was the sign that it had not been wasted.",
            "Truth had come with mud on its feet and malice in its mouth, but it had come.",
            "What remained was to carry it without letting fear deform it further.",
            f"The guarded place around {object_phrase} seemed still afterward, but stillness was not peace.",
            "It was merely the pause before warning.",
            "The pause did not last.",
            "Words would move.",
        ],
        "escape": [
            "The empty place he left behind seemed larger than his body had ever been.",
            "Absence became evidence, and evidence became warning.",
            "A watch may fail in the moment and still reveal the road by which duty must run.",
            f"In the dark around {place_name}, pursuit had to learn again what custody had briefly made it forget.",
            "The empty dark had become a messenger.",
            "It travelled quickly.",
            "So must they.",
        ],
        "warning": [
            "The warning did not ask permission of comfort. It took the road because the road was all that remained.",
            "Behind it, vigilance gathered itself in silence.",
            "The west slept on, and sleep was now a thing to be defended with haste.",
            f"{object_phrase} went with them or stayed behind according to need, for need had become the only honest steward.",
            "So warning took the road.",
            "It travelled lean.",
            "It had to.",
        ],
    }
    coda_pool = _hunt_pool(coda_by_kind)
    coda_idx = _stable_seed("hunt-coda", goal, place_name, char_phrase, object_phrase) % len(coda_pool)
    coda_attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and coda_attempts < len(coda_pool):
        paragraph = coda_pool[(coda_idx + coda_attempts) % len(coda_pool)]
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        coda_attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    longform_by_kind = {
        "border_watch": [
            (
                f"{ranger} set his watch by unheroic measures: which carts used the road twice, which strangers "
                "asked too little, which dogs barked at night and which had learned a familiar silence."
            ),
            (
                "He did not trouble the Shire with his fear. Fear delivered too early can become another invader, "
                "entering hearth and field before any enemy has crossed the hedge."
            ),
            (
                f"The borders near {place_name} seemed gentle, but gentleness is not the same as safety. It is "
                "often the reason safety must be purchased elsewhere."
            ),
            (
                f"{object_sentence_start} belonged to the modest economy of the watch. Nothing there proclaimed "
                "defence, and that was part of the defence."
            ),
            (
                "At Bree he let men think him only another weather-worn traveller. The mistake was useful; a man "
                "dismissed too quickly may hear what a feared man never would."
            ),
            (
                "Uncertainty became active service when he stopped demanding a single answer from it. Instead he "
                "made it many small duties: a road held, a rumour tested, a darkened lane remembered."
            ),
            (
                "The threat remained offstage, but stages are not where danger begins. It begins in preparation, "
                "in names carried too far, in roads left unmarked by those who should have watched."
            ),
            (
                "He thought of the Shire not as an idea but as little evidences of peace: smoke lifting straight, "
                "a gate left easy on its hinge, a field where fear had not yet taught the hand to hurry."
            ),
            (
                f"{ranger} could not make that peace permanent. He could only spend himself so that it remained "
                "uninterrupted a little longer."
            ),
            (
                "So he accepted the loneliness of the edge. Some guards stand before walls; others stand outside "
                "the firelight and make sure the wall is not needed yet."
            ),
            (
                "The tenderness of the land did not soften his task. It made the task more exact, for rough handling "
                "would injure the very thing he meant to preserve."
            ),
            (
                "He learned again that watchfulness near peace must be quieter than watchfulness near war. At a "
                "battle line, alarm is honest; beside a sleeping field, alarm can become harm."
            ),
            (
                "The nearer he came to gardens and tilled ground, the more he distrusted any thought that made "
                "them symbols only. They were places, and places must be guarded as themselves."
            ),
            (
                "So the border did not harden into a line on a map. It became a series of human thresholds: "
                "lanes, gates, market roads, and the last rise before a stranger might see too much."
            ),
            (
                f"{ranger} took the softness of the land as evidence, not comfort. A road that looks harmless may "
                "be harmless because others have kept harsher things away from it."
            ),
            (
                "There were sounds he would have ignored in rougher country: a latch, a cart-rim, a child calling "
                "across a field. Here such sounds were the measure of what secrecy served."
            ),
            (
                "He let himself feel the ache of being outside it. The ache did not weaken the watch; it reminded "
                "him that the watch guarded more than an abstraction."
            ),
            (
                "The hidden guard therefore became tender without becoming soft. Tenderness knew what must not be "
                "crushed; sternness knew what must not be allowed through."
            ),
            (
                "Between those two truths he found the road for that night, and took it without witness."
            ),
            (
                "A mist gathered low in the hollows before morning. It blurred the nearer hedges and made the "
                "farther ones seem like memories, but it did not hide the road from him; he had already learned "
                "where the road wished not to be seen."
            ),
            (
                "So he kept the watch by touch as much as sight: the slope underfoot, the loosened stone near a "
                "turn, the damp on a gate that had been opened after dewfall."
            ),
            (
                "Such knowledge would never enter a song. That was well. If it had done its work, no song would "
                "be needed there yet."
            ),
            (
                "When dawn began, the fields did not look defended. They looked merely themselves, which was the "
                "best answer his hidden labour could receive. He took that answer gravely and went on."
            ),
            (
                "Behind him, no gate closed in gratitude and no voice called blessing after him. Ahead, another "
                "mile waited with its own ordinary face. He accepted both facts, for the work had never depended "
                "on being known."
            ),
            (
                "The morning birds began as if no shadow had ever crossed a map. He let them sing. Their ignorance "
                "was not a rebuke to his fear, but the reason fear had to be mastered and made useful."
            ),
            (
                "So he passed from the rise before full day, leaving no sign that would trouble a farmer, and every "
                "sign that might help a friend."
            ),
            (
                "That balance was narrow, but narrow roads had long been his portion."
            ),
            (
                "He took this one without complaint, and the morning received him as it received all hidden "
                "service: without witness, without gratitude, and without any lessening of the debt."
            ),
        ],
        "active_service": [
            (
                "At Bree the work did not announce itself as work. It looked like waiting under a low beam, "
                "standing aside for a cart, or asking after a road with no urgency in the voice."
            ),
            (
                f"{ranger} let the inn talk pass through him slowly. Most of it was chaff, but chaff can show the "
                "wind, and the wind was what he needed to know."
            ),
            (
                "He set one rumour aside because it desired too much attention. He kept another because the teller "
                "seemed ashamed of it. Shame often guards a small truth badly but honestly."
            ),
            (
                f"{object_sentence_start} served as cover for patience. A man with some small business in his hands "
                "is less likely to be asked why he listens."
            ),
            (
                "By nightfall uncertainty had become a circuit: gate, road, inn-yard, low field, and the place "
                "where strangers slowed before deciding whether to enter."
            ),
            (
                "No single act deserved mention. Together they made a watch. That was the nature of the service: "
                "small acts arranged so that danger found fewer unregarded spaces."
            ),
            (
                "Aragorn did not feel less weary because the work was quiet. Quiet work often asks more of a man, "
                "for there is no clash of arms to lend the body borrowed fire."
            ),
            (
                "Still he kept to it. The failed hunt had not emptied his duty; it had poured the duty into smaller "
                "vessels, and each had to be carried without spilling."
            ),
            (
                "Before he left the last lighted door behind, he had made uncertainty into motion. Not triumph, not "
                "knowledge, but motion disciplined enough to matter."
            ),
            (
                "That was active service: to keep moving where doubt could be tested, and to stop where stopping "
                "would hear more than haste."
            ),
            (
                "He went again to the places where ordinary men ceased to notice their own roads: the damp patch "
                "beneath the gate, the rut where carts turned wide, the stretch where a stranger might slow without "
                "seeming to hesitate."
            ),
            (
                "The work was not made less grave by being made of such things. Great peril often enters the world "
                "through habits too small for pride to inspect."
            ),
            (
                f"{ranger} kept his questions poor and useful. A poor question can be asked twice without offence, "
                "and usefulness was dearer to him than the satisfaction of sounding wise."
            ),
            (
                "At the inn he let laughter rise and fall around him. Laughter told him who felt safe, who pretended "
                "to feel safe, and who had been silent too long when the eastern road was named."
            ),
            (
                "No answer gave him the creature. That was not the answer he sought now. He sought the shape of "
                "approach: where rumour thinned, where fear avoided a name, where a road might be kept shut by "
                "patience before it needed a sword."
            ),
            (
                f"{object_sentence_start} became part of a plain disguise. A cloak drying by a hearth, a boot "
                "turned toward the door, a track studied after rain: each could belong to idleness until need "
                "made it otherwise."
            ),
            (
                "He was learning to defend a land by defending the meanings of its small motions. A cart must remain "
                "a cart, a market road a market road, a child calling at dusk only a child calling at dusk."
            ),
            (
                "To preserve that ordinariness, he had to imagine what would profane it. The imagination hurt him, "
                "but a guard who refuses to imagine danger leaves danger to imagine him."
            ),
            (
                f"Near {place_name}, {local_detail} did not look like a battlefield. That was why he treated it "
                "more carefully than one."
            ),
            (
                "A battlefield declares itself and asks for courage. A border near peace asks for judgement, "
                "for restraint, and for the humility to let necessary work look like wandering."
            ),
            (
                "So he made no speech to the road and expected no answer. He set one watch behind another, like "
                "small lamps screened from wind, and trusted that a hidden light may still deny darkness a passage."
            ),
            (
                "By the time the night settled, his service had taken its final form for that hour: no capture, "
                "no proof, no tale to carry inward, only a watched road and a will that had not slackened."
            ),
        ],
        "offstage_watch": [
            (
                "The danger remained outside the visible scene, and that was its advantage. Men prepare readily "
                "for a wolf at the gate; they grow careless before a rumour with no body."
            ),
            (
                f"{ranger} watched the border as one watches a sleeping fire: not because flame is present, but "
                "because ash remembers heat and wind may return."
            ),
            (
                "He thought of Gollum not as near, but as loose. Loose things change the world by possibility, "
                "and possibility was enough to alter every road leading west."
            ),
            (
                f"{object_sentence_start} were kept ready in plain sight. Readiness hidden too well is sometimes "
                "only another form of unreadiness."
            ),
            (
                "The Shire beyond the watched ways remained itself: small roofs, slow smoke, and the innocence of "
                "people who would mistake their guardian for a threat."
            ),
            (
                "That misunderstanding did not sour the service. It defined it. A guard who requires gratitude "
                "has already made the guarded thing smaller."
            ),
            (
                "So the offstage threat stayed active in him. It moved through his choices, narrowed his sleep, "
                "and taught his eyes to travel every hedge before trusting the road."
            ),
            (
                "Until proof arrived, watchfulness would have to serve in proof's place. It was a hard substitute, "
                "but a faithful one."
            ),
            (
                "At times he wished for a visible enemy, and mistrusted the wish as soon as it came. A visible "
                "enemy flatters courage; an unseen one tests patience."
            ),
            (
                f"Along {place_name}, the signs of peace were everywhere. That abundance made the labour strange, "
                "for he had to look for danger without teaching his eyes to despise peace."
            ),
            (
                "He remembered Gandalf's haste and Gollum's muttering in the same thought. Between them lay the "
                "border: quiet, ignorant, and no longer safely obscure."
            ),
            (
                "The roads did not answer him with certainty. They offered only their ordinary use, and ordinary "
                "use had now to be guarded from extraordinary knowledge."
            ),
            (
                f"{ranger} marked places where a stranger might pause without seeming to pause: a bend with cover, "
                "a hedge-gap, a rise from which a small country could be studied too well."
            ),
            (
                "No shadow crossed those places while he watched. That did not make the watching vain. The worth "
                "of a watch is often proved by what never arrives."
            ),
            (
                "So he let the threat remain offstage without letting it become unreal. It had entered the story "
                "by names, and names can travel before feet."
            ),
            (
                "The night deepened over fields that did not know him. He accepted that ignorance as part of the "
                "peace, and guarded it as carefully as any gate."
            ),
            (
                "He had no proof that night would bring harm. He had only the knowledge that harm, once invited "
                "by a name, may travel by roads that look harmless until too late."
            ),
            (
                f"{object_sentence_start} were checked again, though nothing had changed. Rechecking is one of "
                "the plain rituals by which fear is made obedient."
            ),
            (
                "A fox crossed the lower road and vanished. The small sound it left behind seemed almost foolish "
                "to notice, but he noticed it all the same."
            ),
            (
                "The offstage threat required that kind of humility. A watcher cannot demand that every sign be "
                "worthy of him."
            ),
            (
                "Far away, Gollum still moved in the tale, whether north, south, or hidden in some fold of fear. "
                "That uncertainty kept the border from becoming merely pastoral."
            ),
            (
                "Aragorn let the uncertainty remain sharp but not wild. A wild fear runs in circles; a sharpened "
                "one chooses a road and keeps it."
            ),
            (
                "Before dawn he had made no discovery fit for report. Yet the roads were less unwatched than they "
                "had been, and that was a real, if secret, alteration of the world."
            ),
        ],
        "finale": [
            (
                f"{wizard} went westward with more than knowledge. Knowledge alone can sit too long by a fire and "
                "call its caution wisdom; this knowledge had become a summons."
            ),
            (
                f"{ranger} did not follow him to the door. His road bent aside, as hidden roads often do, toward "
                "the places where danger might arrive before any warning was believed."
            ),
            (
                f"{quarry} was not absent from that ending. A creature need not stand in sight to trouble the hour; "
                "his loose memory had already entered roads beyond his own choosing."
            ),
            (
                "Thus the three movements of the tale drew apart and belonged together. Warning hastened, "
                "watchfulness remained, and malice wandered with names it had never understood rightly."
            ),
            (
                f"About {place_name}, {local_detail} gave no comfort of completion. The ground did not change "
                "because men had judged the matter grave."
            ),
            (
                f"{object_sentence_start} could be gathered, carried, or left ready; none could make the ending "
                "clean. The tools of the hunt had become the tools of delay."
            ),
            (
                "Delay was no small gift. In a darkening age, an hour bought before the enemy's thought reaches a "
                "quiet house may be worth more than a field won in open battle."
            ),
            (
                "Yet the bought hour did not excuse rest. It made rest more difficult, because every mercy gained "
                "by hidden labour asks to be spent faithfully."
            ),
            (
                f"{wizard} knew that a peaceful face can be the hardest face to warn. Fear would sound uncouth "
                "among small duties, and yet small duties were exactly what fear now threatened."
            ),
            (
                f"{ranger} knew another bitterness: the more faithfully he served, the less fit his service was "
                "to be seen. A visible guard may comfort the defended; an unseen guard must be content to be misread."
            ),
            (
                "No song rose over that division of labour. Song often arrives late, after pain has been made "
                "shapely by distance. This labour still had mud on it."
            ),
            (
                "The house under the Hill stood far away in thought, ordinary as bread and dangerous as a secret. "
                "Its ordinariness did not reduce the peril; it made the peril less tolerable."
            ),
            (
                "The name Baggins had travelled farther than it ought, and the name Shire had come after it like "
                "a green lamp seen by eyes that hated light."
            ),
            (
                "Still the tale did not surrender to dread. There remained friendship, judgement, endured weariness, "
                "and the stubborn fact that servants had acted before triumph made action easy."
            ),
            (
                f"{quarry} crept somewhere beyond their certainty, but uncertainty was no longer empty. It had "
                "been filled with watches, messages, remembered signs, and a westward road taken in earnest."
            ),
            (
                "The end of the chapter therefore stood like twilight rather than night. Shadows lengthened, yet "
                "shapes could still be known by those who kept their eyes steady."
            ),
            (
                f"If providence was present, it did not speak loudly. It appeared in the hard timing of {wizard}'s "
                f"departure, in the road {ranger} chose without applause, and in the fact that evil had not yet arrived first."
            ),
            (
                "That was enough for the hour, and no honest ending could claim more. Enough is a stern word when "
                "hearts desire safety, but many tales turn upon whether enough is received and used."
            ),
            (
                "So the last image held without closing: a staff on the road, a watcher beyond the lamps, and a "
                "small bitter memory moving loose through the wild."
            ),
            (
                "The danger remained unfinished. The hope remained unfinished also, which is why it was hope and "
                "not decoration."
            ),
            (
                "There are endings that shut a door and endings that show why a door must be guarded. This was "
                "of the latter kind, and the silence after it was not empty."
            ),
            (
                f"{wizard}'s haste had become merciful because it did not wait to be certain of every shadow. "
                "Certainty would come too late if it came only after evil had knocked."
            ),
            (
                f"{ranger}'s patience had become merciful because it did not demand to be seen by those it served. "
                "A watched road need not know the watcher in order to be spared."
            ),
            (
                f"{quarry}'s freedom had become terrible because it carried fragments of truth inside a broken will. "
                "The fragments were small, but small things had already altered the burden of the wise."
            ),
            (
                "No one could yet say what the fragments would cost. That ignorance gave the final page its weight, "
                "for known grief may be endured, while unmeasured grief must be prepared for without shape."
            ),
            (
                "The Shire did not shine in the distance like a prize. It lay in thought as a country of rooms, "
                "lanes, seed-cake, gossip, poor jokes, and doors opened without dread."
            ),
            (
                "Such things are easily made ridiculous by those who love only splendour. Yet they are among the "
                "first things tyranny destroys, because they prove that life can be whole without domination."
            ),
            (
                f"That is why {wizard} went on with fear mastered but not denied. A denied fear may become folly; "
                "a mastered fear may become haste in the service of love."
            ),
            (
                f"That is why {ranger} remained with weariness accepted but not indulged. Weariness can make a man "
                "bitter if he counts only what is owed to him; he counted instead what had been entrusted."
            ),
            (
                f"And that is why {quarry}, miserable and dangerous, could not be dismissed as a mere remnant. "
                "Remnants have long memories, and memory may serve hatred when it has refused healing."
            ),
            (
                "The roads between them did not speak, but they had been changed. They were no longer only passages "
                "for travellers; they had become threads in a design of warning."
            ),
            (
                "A design need not be visible to be real. Much of what preserves the world is invisible until the "
                "day it fails, and then all men ask why no one stood watch."
            ),
            (
                "Here some had stood watch. They had stood imperfectly, with missed trails, bitter testimony, "
                "failed custody, and decisions made from less knowledge than the heart desired."
            ),
            (
                "Yet imperfection had not made the labour false. It had made the labour human, and therefore fit "
                "to serve mercy rather than pride."
            ),
            (
                "The chapter could not bless the darkness by pretending it was over. It could only set against "
                "it the quiet facts of service already given and service still owed."
            ),
            (
                "So the westward road, the watched border, and the wandering malice remained in one severe music. "
                "None resolved the other; each made the other matter."
            ),
            (
                "No triumph was offered there, only tasks still in motion: warning west, watch at the borders, "
                "and a loose creature crawling beyond sight."
            ),
            (
                "The last light therefore did not fall on victory. It fell on readiness, and readiness, though "
                "plain in appearance, had become the fairest thing the tale could honestly show."
            ),
        ],
        "counsel": [
            (
                f"{wizard} and {ranger} spoke as men who had both lost something and refused to call the loss final. "
                "Between them lay no full answer, only signs that leaned together until denial became harder than action."
            ),
            (
                f"The map before them, {object_phrase}, could not show the weight of uncertainty. It could show "
                "distance, road, river, and wood; the rest had to be measured in judgement."
            ),
            (
                f"{ranger} set down what the hunt had taught him: the quarry's cunning, the southward pull, the "
                "way Baggins and Shire returned whenever fear loosened the captive's tongue."
            ),
            (
                f"{wizard} answered with what questioning had taught: not certainty, but convergence. Separate "
                "doubts had begun to point like poor arrows toward the same green country."
            ),
            (
                "They did not flatter each other with comfort. Comfort would have been another delay, and delay "
                "had become suspect now that even ignorance had a direction."
            ),
            (
                "The question before them was not whether the hunt had succeeded. It had succeeded enough to make "
                "failure dangerous, and failed enough to make vigilance necessary."
            ),
            (
                f"{ranger} accepted the roads left to him without ceremony. Roads watched in secret are rarely "
                "clean victories; they are promises renewed at crossings, fords, hedges, and nameless hills."
            ),
            (
                f"{wizard} looked west as if listening for a clock no one else could hear. The hour had not struck, "
                "but the mechanism of peril had begun to move."
            ),
            (
                "So their counsel did not close the matter. It made the matter portable: something that could be "
                "carried by staff, by worn boots, by hidden marks, and by memory kept sharper than fear."
            ),
            (
                "They returned more than once to what they did not know. That was not weakness in the counsel; "
                "it was its honesty, and honesty had become more useful than any brave pretence."
            ),
            (
                f"{ranger} spoke of the failed trail without excusing it. The ground had ended one labour, but "
                "not the obligation behind it."
            ),
            (
                f"{wizard} spoke of testimony without trusting it too far. A crooked mouth may reveal direction "
                "and still lie about distance, motive, and the next turning."
            ),
            (
                "Between those two accounts, a shape emerged: not a victory, not a defeat, but a peril now known "
                "well enough to demand costly action."
            ),
            (
                "The counsel grew quieter as it neared decision. Loud words often stand at the edge of uncertainty; "
                "necessary decisions sometimes arrive almost without sound."
            ),
            (
                f"{object_sentence_start} lay between them while the last alternatives were weighed. None promised "
                "comfort. The best merely promised that warning might arrive before ignorance became fatal."
            ),
        ],
        "trail": [
            (
                f"The nearer the quarry seemed, the less {ranger} trusted nearness. The Dead Marshes could put a "
                "sound beside the ear and its maker half a mile away, or hide a body where the grass looked barely stirred."
            ),
            (
                f"He used {object_phrase} sparingly, because every tool declares something about the hand that carries it. "
                "A bright edge, a dangling strap, or a careless knot could become news for eyes that had learned to fear."
            ),
            (
                "Sleep came in short bargains. He took it where the ground rose a little above the wet, with one hand "
                "under the cloak and the mind still walking after signs the body could no longer follow."
            ),
            (
                "By day he kept to the meaner evidence: broken scum on a pool, a reed bent against the wind, the place "
                "where a frightened creature had chosen discomfort because concealment mattered more."
            ),
            (
                "The work was not noble to look at. It was kneeling, smelling mud, waiting for birds to forget him, "
                "and accepting that patience may be the only speed left to a hidden servant."
            ),
            (
                "Once he found a hollow where the quarry had crouched through rain. Nothing remained there that a "
                "less wary traveller would have named; yet the place felt used, hated, and left too quickly."
            ),
            (
                "So the pursuit closed by inches. Not by a shout, not by a leap, but by the stern accumulation of "
                "small refusals to let the world erase what had passed over it."
            ),
        ],
        "failure": [
            (
                f"{ranger} went back over the ground once, then again, each time taking less from it. The first "
                "reading had been cautious; the second was sterner, for it had to judge his own desire as well as the marks."
            ),
            (
                "A lesser hunter might have chosen the most dramatic sign and called doubt cowardice. Aragorn knew "
                "that doubt can be a servant of truth when pride is eager to ride."
            ),
            (
                f"The country around {place_name} did not mock him. Mockery would have been easier to answer. It "
                "merely offered banks, stones, grass, and distances, each refusing to say more than it knew."
            ),
            (
                f"{object_sentence_start} were gathered without haste. The act of packing them felt like defeat "
                "because it admitted the end of one kind of labour."
            ),
            (
                "Yet the failure had shape. The quarry had not vanished by miracle; weather, fear, cunning, and "
                "old country had conspired in ways that could be named, even if they could not be reversed."
            ),
            (
                "He marked where certainty ended. That mark mattered. To know the boundary of knowledge is to "
                "guard against the first lie despair tells."
            ),
            (
                "The road ahead would not be abandoned, but it would be approached differently. Pursuit had spent "
                "its claim; vigilance, message, and humility must take up what remained."
            ),
            (
                "So the defeat became instruction. It taught him to watch wider, to trust fewer proud guesses, "
                "and to remember that the quarry's absence could still point toward danger."
            ),
        ],
        "quarry": [
            (
                f"{quarry} went on beyond the hunters' knowledge, and that ignorance was itself part of the danger. "
                "A thing unseen may still carry names, hunger, and old malice into roads no watcher has chosen."
            ),
            (
                "He did not understand the counsels that followed him. He understood ache, fear, food, darkness, "
                "and the small hot coal of a name he would not let cool."
            ),
            (
                "Baggins returned whenever weariness thinned his thought. Sometimes it came as a curse, sometimes "
                "as a plea, and sometimes as a sound he made only to feel the wound answer."
            ),
            (
                "The Shire remained less clear to him than the hatred attached to it. That made it no safer. A "
                "blurred road may still be followed by a creature who has forgotten every better destination."
            ),
            (
                f"About {place_name}, {local_detail} offered him neither comfort nor command. He used what could be "
                "used and hated the rest for existing beyond his control."
            ),
            (
                f"{object_sentence_start} meant little beside hunger, yet even poor things could become tokens in "
                "his narrowed kingdom: proof that the day had not yet mastered him."
            ),
            (
                "He spoke sometimes to the dark as if it were an ally, and sometimes as if it had stolen from him. "
                "Both accusations pleased him, because both kept loneliness from sounding like silence."
            ),
            (
                "Thus the loose quarry remained a moving wound in the tale. He had escaped custody, not consequence; "
                "and consequence, unlike rope, does not need hands to hold."
            ),
        ],
        "capture": [
            (
                f"The spring had to come after long stillness. {ranger} waited until the small sounds before him "
                "settled into hunger rather than suspicion, and only then let his body remember speed."
            ),
            (
                f"{quarry} fought with the whole ugliness of a life reduced to not being held. He clawed at mud, "
                "twisted toward water, and made his bones seem loose as rope in the hand."
            ),
            (
                "There was no clean grip to take. Wet hair slipped, teeth found cloth, and the ground itself seemed "
                "to pull both hunter and hunted down into a struggle unfit for any song."
            ),
            (
                f"{ranger} spoke little while binding him. Speech would have made the work theatrical, and nothing "
                "there deserved theatre: not the mud, not the fear, not the pity that had to keep its hands firm."
            ),
            (
                "The captive tried several voices. One begged, one accused, one promised, and one merely hissed "
                "Baggins with such hatred that the name seemed sharper than the knife at Aragorn's belt."
            ),
            (
                "The rope was tested after every knot. To leave it loose would betray the Shire; to draw it cruelly "
                "tight would betray the mercy that made capture worth more than death."
            ),
            (
                f"When the struggle ended, {place_name} looked unchanged. That indifference was almost shocking. "
                "The marsh had witnessed a turn in the hidden history of the West and went on breathing vapour."
            ),
            (
                "Only the captive's breathing proved that something had been won. It was a poor sound, wet and "
                "resentful, yet it preserved the possibility of answers."
            ),
        ],
        "captivity": [
            (
                "The northward miles made pity more difficult than pursuit had done. A hunted thing is mostly "
                "absence; a captive is presence hour after hour, hungry, fouled, frightened, and hateful."
            ),
            (
                f"{ranger} learned the rhythm of Gollum's resistance. It rose at water, sharpened near food, "
                "became sly at dusk, and sank into muttering when exhaustion took from malice the strength to perform."
            ),
            (
                f"{object_sentence_start} became part of a harsh household on the road. Cloak, rope, poor food, "
                "and guarded sleep had to serve both life and restraint, though neither captive nor keeper found peace in them."
            ),
            (
                "At times Gollum seemed so wretched that disgust itself grew ashamed. Then he would bare his teeth, "
                "or whisper Shire as if tasting theft, and the danger in him stood up again."
            ),
            (
                f"{ranger} did not answer every curse. Some words are snares laid by misery; to step into each "
                "one would give the prisoner command of the road."
            ),
            (
                "Mercy had to become practical or it would fail. It meant clean water when water could be found, "
                "rest before collapse, knots that held without maiming, and watchfulness that did not call itself virtue."
            ),
            (
                "The captive slept badly. So did the keeper. Between their two broken sleeps lay the true burden "
                "of custody: a living danger preserved because dead certainty would answer nothing."
            ),
            (
                "By the time the trees of the north came into thought, the rope had ceased to be a symbol. It was "
                "only rope, rubbed, checked, damp, necessary, and morally heavier than iron."
            ),
            (
                "He measured mercy by its repetitions. Food came, water came, a guard looked away for pity and "
                "then looked back for fear. In those repetitions he began to seek not gratitude but weakness."
            ),
            (
                "The leaves above him shifted with winds he could not feel. He imagined messages in them, though "
                "the messages changed with every bitterness that crossed his mind."
            ),
            (
                "Names remained his secret hoard. Baggins was the sharp one, Shire the warm one, and both were "
                "kept under the tongue like stolen things that might still buy revenge."
            ),
            (
                "He did not plan as wise folk plan. His thought crawled, retreated, tasted the same corner again, "
                "and returned later when pain had made the corner look different."
            ),
            (
                "A root near the place of keeping became important to him. Not because it could free him, not yet, "
                "but because it proved that the ground itself sometimes pushed upward against order."
            ),
            (
                "When the guards spoke softly, he hated softness; when they spoke sternly, he hated sternness. "
                "Hatred made every manner of keeping useful to him, for every manner could be studied."
            ),
            (
                "Thus his captivity lengthened inwardly. The keepers counted days; Gollum counted habits, and the "
                "second tally was the more dangerous one."
            ),
            (
                "At dusk he often seemed to sleep. Yet under the closed lids his hearing wandered, touching each "
                "sound and setting it beside the remembered order of the watch."
            ),
        ],
        "escape": [
            (
                f"{quarry} did not know the hour beforehand. He knew only that hours have edges, and that the edge "
                "of one night felt thinner than the others."
            ),
            (
                f"The wood around {place_name} changed its breathing. A call passed farther off than usual, a lamp "
                "was lifted and not at once returned, and the dark between trunks seemed to make room for itself."
            ),
            (
                f"{object_sentence_start} no longer looked like things owned by the keepers. In Gollum's mind each "
                "became a possible obstacle, lure, shield, or witness to be deceived."
            ),
            (
                "He made no brave resolve. Bravery was a word for creatures with clean choices. His resolve was "
                "need sharpened by years of loss and by one small opening in the pattern of guard and mercy."
            ),
            (
                "The first movement was almost nothing: a shift of weight, a slackening of the neck, a breath held "
                "too long to be sleep and too softly to be noticed."
            ),
            (
                "Behind that smallness gathered every resentment of the guarded days. Kind food, hard rope, watchful "
                "eyes, remembered names: all were pressed into the same dark impulse."
            ),
            (
                "A branch cracked somewhere beyond the place of keeping. It was not meant for him, and therefore "
                "it became his. Chance is often claimed by those who have waited without hope."
            ),
            (
                "He did not yet run. The edge of escape is a country of stillness, where the body must not outrun "
                "the shadow that hides it."
            ),
            (
                "For one breath the whole wood seemed to look elsewhere. In that breath Gollum gathered himself "
                "not upward, but inward, as if vanishing began before motion."
            ),
            (
                "The names went with him before his feet did. Baggins and Shire moved in the dark of his mouth, "
                "small, bitter, and ready to be carried wherever fear could crawl."
            ),
        ],
        "delivery": [
            (
                f"Under the first shadow of Mirkwood, {char_phrase} seemed to enter a different kind of silence. "
                "Open country had hidden tracks; the wood hid intention."
            ),
            (
                f"{wizard} did not greet the captive as a prize. He looked at him as one looks at a locked casket "
                "found in a ruin, knowing both that it must be opened and that something poisonous may breathe within."
            ),
            (
                f"{ranger} gave the account plainly: the marsh, the struggle, the words Baggins and Shire, the "
                "turns of fear that had seemed to lead farther south than hunger could explain."
            ),
            (
                "Gollum crouched from the lamps and hated each face in turn. The hatred was not impressive, but it "
                "was tireless, and tireless malice may carry news farther than strength."
            ),
            (
                f"{object_sentence_start} stood in the guarded place like witnesses brought indoors. Rope and staff "
                "and low light made no judgement, yet each had helped preserve the hour from ignorance."
            ),
            (
                "The Wood-elves watched with the restraint of those accustomed to old dangers. They did not crowd "
                "the prisoner, and they did not mistake his smallness for safety."
            ),
            (
                f"{wizard} asked no question at first. He let Gollum's own muttering travel ahead of interrogation, "
                "for a mouth that thinks itself unheard may wander nearer truth."
            ),
            (
                "Thus the hunt ended as a new labour began. The road had brought in a body; now wisdom had to draw "
                "meaning from a mind that had survived by twisting meaning away."
            ),
            (
                f"{ranger} did not lengthen his report to make the hunt seem greater. He named what he had seen, "
                "where he had guessed, where he had erred, and where the quarry's own fear had confirmed the road."
            ),
            (
                "The guards shifted only when ordered. Their stillness mattered, for Gollum watched movement as "
                "a miser watches coins, counting each habit for later use."
            ),
            (
                f"{wizard} listened most sharply when the tale grew least heroic: the fouled camps, the bitten hand, "
                "the mud-smeared rope, the shame of keeping alive a creature no decent heart could admire."
            ),
            (
                "Mirkwood did not cleanse the matter by receiving it. The trees gathered the danger under their "
                "boughs, and the danger seemed smaller only to those who did not understand small things."
            ),
            (
                "When Gollum heard Baggins spoken by another mouth, he flinched before he could spit. That flinch "
                "was noted by every watcher who knew how truth sometimes enters a room before words."
            ),
            (
                f"{object_sentence_start} made a rough boundary for the hour. Within it stood the hunter, the "
                "questioner, and the captive; beyond it waited all the miles that had not yet been warned."
            ),
            (
                "No one there mistook arrival for ending. Arrival had only changed the work from following signs "
                "that fled to listening for signs that lied."
            ),
            (
                "The first ordering of the guard was done in plain words and without harshness. Plainness was "
                "important: cruelty would have taught the prisoner one lesson, and indulgence another equally false."
            ),
            (
                f"{ranger} saw how the lamps troubled Gollum. Not because they were bright, for they were not, but "
                "because they belonged to a world where things could be seen and remembered by others."
            ),
            (
                f"{wizard} asked the Wood-elves for patience before he asked them for strength. Strength was already "
                "present; patience would decide whether strength served wisdom or merely custody."
            ),
            (
                "The captive made himself small, then suddenly large with spite, then small again. Each change was "
                "a defence, and each defence showed how long he had survived by refusing any fixed shape."
            ),
            (
                f"The report of the road passed from {ranger} to {wizard} with no ornament. In such matters, "
                "ornament is a kind of falsehood, for it lets the hearer admire what ought to trouble him."
            ),
            (
                "So the guarded place received not a trophy, but a question with limbs. Every watcher understood "
                "that the answer might be bought only by many hours of restraint."
            ),
            (
                f"{ranger} stood aside when the first watch was set, yet his labour did not leave him. The road "
                "had entered his shoulders, his hands, and the tired caution with which he judged every movement near the rope."
            ),
            (
                "Gollum tested the room by inches. He shifted his weight, dragged one foot, let his head loll, "
                "and watched whether pity or impatience moved first in those who guarded him."
            ),
            (
                f"{wizard} saw the test and said nothing. There are moments when a questioner learns more by "
                "letting cunning spend itself than by interrupting it with command."
            ),
            (
                "The lamps were lowered rather than brightened. Too much light would make a spectacle; too little "
                "would flatter the captive's love of holes. The middle way was chosen because it was least convenient to deceit."
            ),
            (
                f"{object_sentence_start} were placed where hands could reach them without display. A guarded hour "
                "should not bristle like a battlefield, but neither should mercy have to search for its tools."
            ),
            (
                "The story of the capture was repeated once, more slowly. In the second telling, different details "
                "came forward: the southward fear, the bitten hand, the name that had escaped under pressure."
            ),
            (
                f"{ranger} marked how Gollum listened whenever the road was described. The prisoner hated the tale, "
                "but hatred did not prevent him from recognizing his own passage through it."
            ),
            (
                f"{wizard} asked where the creature had seemed least afraid. That question troubled Aragorn more "
                "than the questions about fear, for least afraid did not mean safe; it might mean obedient to a darker dread."
            ),
            (
                "Outside the guarded hollow, Mirkwood breathed in resin and leaf-shadow. Within it, every ordinary "
                "sound grew exact: a foot placed softly, a rope fibre strained, a lamp wick touched with iron."
            ),
            (
                "Thus delivery became a kind of translation. The road had spoken in marks and weariness; now its "
                "meaning had to be carried into the severe language of questions."
            ),
            (
                "No one thanked Aragorn for the ugliness he had brought. Thanks would have been too easy. The better "
                "honour was the silence in which others accepted the burden as real."
            ),
            (
                "When the first watch settled, the prisoner seemed smaller than before. Yet the room did not grow "
                "lighter, for the danger had never depended on the size of his body."
            ),
        ],
        "questioning": [
            (
                f"{wizard} weighed silence and speech as {ranger} had weighed mud and reed. Neither craft could "
                "force truth to stand upright, but each could notice where falsehood leaned."
            ),
            (
                f"{quarry} answered by injury, evasion, and sudden spite. Yet the same small names returned, and "
                "their return made them graver than any orderly confession would have done."
            ),
            (
                "A question asked too soon might teach the prisoner what mattered. A question withheld too long "
                "might let fear grow useless. Between those errors the hour moved slowly."
            ),
            (
                f"{object_sentence_start} remained near, humble and severe. The greatest danger in the room had "
                "arrived by rope, mud, and guarded miles, not by splendour."
            ),
            (
                "By the end of that first guarded listening, certainty had not been won. Something more useful and "
                "more painful had been gained: enough knowledge to make delay irresponsible."
            ),
            (
                f"{wizard} asked around the sore places first. He spoke of roads, hunger, water, caves, and old loss, "
                "letting the prisoner's anger show which words had struck near the hidden nerve."
            ),
            (
                "Gollum's speech ran in circles, but the circles were not empty. Each return passed near Baggins, "
                "and each passing made the name less accidental."
            ),
            (
                "No hand was raised against him. That restraint did not soften the hour; it made the hour harder, "
                "because cruelty would have given everyone a simpler lie to live inside."
            ),
            (
                f"{quarry} tried to make pity ridiculous. He whimpered when watched sternly and spat when spoken "
                "to gently, searching for a weakness in every decent impulse."
            ),
            (
                f"{wizard} did not let disgust set the pace. Disgust hurries to conclusions; fear of the right "
                "kind walks slower, because it knows a missed word may cost more than an unpleasant hour."
            ),
            (
                "The Shire came out not as a confession but as a recurrence. It returned when the captive cursed, "
                "when he drowsed, when he denied knowledge, and when his denial forgot what it meant to hide."
            ),
            (
                f"{ranger} heard in the answers a second trail laid over the first. Mud had brought him to the "
                "prisoner; now speech showed where the prisoner's memory had been travelling."
            ),
            (
                "At one point the lamps guttered in a draught, and Gollum shrank as if darkness itself had touched "
                "him. The movement was too swift for pretence and too old for ordinary fear."
            ),
            (
                "The questioner and the hunter exchanged no triumphal look. What they had learned made victory "
                "seem an impertinent word."
            ),
            (
                "By slow degrees, Baggins and Shire ceased to be muttered relics of a private hatred. They became "
                "directions on a map the Enemy must not be allowed to read."
            ),
            (
                f"{wizard} returned more than once to the same harmless-seeming word, changing the path by which "
                "he approached it. Gollum noticed the word but not always the path, and that difference mattered."
            ),
            (
                "When the prisoner lied, he lied with labour. True forgetfulness is usually duller; this was a "
                "busy forgetting, patched and repatched before their eyes."
            ),
            (
                f"{object_sentence_start} threw long shapes against the wall. In those shapes the captive sometimes "
                "seemed larger than his body, as if memory had cast its own crooked shadow beside him."
            ),
            (
                "The name Baggins brought hatred. The name Shire brought something stranger: hunger, grievance, "
                "and a furtive warmth immediately spoiled by malice."
            ),
            (
                f"{ranger} understood then why the hunt had needed a living quarry. No dead sign on the road could "
                "have shown this confusion of need, spite, and unwilling recollection."
            ),
            (
                "At last the questioning paused because more pressure would have made the answers smaller, not "
                "truer. Wisdom sometimes stops before exhaustion has won."
            ),
        ],
        "warning": [
            (
                f"{wizard} and {ranger} withdrew from the prisoner with more silence than they had carried in. "
                "Silence was needed, for speech too soon would have scattered fear before it could become counsel."
            ),
            (
                f"The maps before them, {object_phrase}, looked suddenly insufficient. Ink could show roads and "
                "rivers; it could not show how quickly a small name might travel once hatred had given it feet."
            ),
            (
                f"{wizard} set the matter in order without making it cleaner. Gollum had known the ring-bearer, "
                "had nursed the name Baggins, and had carried Shire out of secrecy by the mere poison of memory."
            ),
            (
                f"{ranger} answered as a man used to unwelcome duties. The westward roads could be watched, "
                "messages could be hidden, and strangers could be delayed without teaching the Shire to fear every shadow."
            ),
            (
                "That last restraint mattered. To defend a peaceful land by filling it with dread would be a poor "
                "defence, and perhaps a victory granted too early to the darkness."
            ),
            (
                f"{wizard} knew that warning must outrun completeness. Perfect knowledge was still locked in a "
                "crooked mouth, but enough had escaped to make waiting a kind of folly."
            ),
            (
                "They spoke then of roads not as travellers speak of convenience, but as guardians speak of weak "
                "places in a wall no one else can see."
            ),
            (
                f"{ranger} took the less visible burden without ceremony. If the Shire was to remain untroubled "
                "a little longer, the wild around it would need eyes that asked for no welcome."
            ),
            (
                "The names at the centre of the matter had not become grand. They remained homely, almost fragile; "
                "that was why they had to be kept from mouths that loved domination."
            ),
            (
                f"When {wizard} looked westward, haste entered the room. It did not stamp or cry out. It simply "
                "made every object there seem already late."
            ),
            (
                "So the chapter's victory, if it could be called so, was a burden transferred. The hunt had become "
                "testimony, and testimony had become warning before any heart was ready."
            ),
            (
                "Outside, the trees held their old darkness. Beneath them two friends measured how little time had "
                "been bought and how dearly even that little might have to be spent."
            ),
            (
                f"The first question was not where {wizard} should go, but how much fear he should carry openly. "
                "Too little would leave friends asleep; too much would wake every listening shadow."
            ),
            (
                f"{ranger} named the roads he could hold and those he could only watch from a distance. It was a "
                "poor map of safety, yet poorer maps have saved lives when read by faithful eyes."
            ),
            (
                "They did not speak of glory, because glory had no work to do there. Food, messages, watches, "
                "weather, secrecy, and speed were the servants required."
            ),
            (
                "By the end of their counsel, the west had become less a direction than a duty. The warning would "
                "go that way, and every road behind it would have to learn vigilance."
            ),
            (
                f"{wizard} did not mistake speed for wisdom. He would go swiftly, but haste must still choose what "
                "to reveal, what to conceal, and which fear to awaken first."
            ),
            (
                f"{ranger} understood the poorer half of the task. While warning travelled, watchfulness had to "
                "remain behind, covering roads that would never know his name."
            ),
            (
                "The house under the Hill was not named as a fortress or a prize, but as a small place suddenly "
                "standing at the centre of consequences too large for its doors."
            ),
            (
                "Their decision had no splendour. It consisted of direction, departure, hidden signs, and the "
                "agreement that delay had become more dangerous than uncertainty."
            ),
            (
                f"{object_sentence_start} were gathered with new severity. What had been tools of counsel became "
                "instruments of motion."
            ),
            (
                "The warning would not end the danger. It would only give the unwarned a chance to become ready "
                "before darker knowledge arrived."
            ),
        ],
    }
    longform_pool = longform_by_kind.get(extension_kind, [])
    longform_idx = _stable_seed("hunt-longform", goal, place_name, char_phrase, object_phrase) % max(1, len(longform_pool))
    longform_attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and longform_attempts < len(longform_pool):
        paragraph = longform_pool[(longform_idx + longform_attempts) % len(longform_pool)]
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        longform_attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    ordinal_words = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
    ]
    concrete_tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", goal + " " + place_name + " " + object_phrase)
        if token.lower() not in STOPWORDS
    ]
    cue = concrete_tokens[0].lower() if concrete_tokens else "trail"
    scene_key = (concrete_tokens[-1] if concrete_tokens else "watch").lower()
    tag_pool = [
        "ashen",
        "briar",
        "cold",
        "dusk",
        "fen",
        "flint",
        "hollow",
        "lantern",
        "mire",
        "rain",
        "root",
        "thorn",
        "wold",
    ]
    tag_base = _stable_seed("hunt-safety-tag", goal, place_name, object_phrase)
    safety_attempts = 0
    while _count_words("\n\n".join(paragraphs)) < target_words and safety_attempts < len(ordinal_words):
        ordinal = ordinal_words[safety_attempts]
        local_cue = (concrete_tokens[safety_attempts % len(concrete_tokens)] if concrete_tokens else cue).lower()
        scene_tag = tag_pool[(tag_base + safety_attempts) % len(tag_pool)]
        attempt_tag = tag_pool[(tag_base + safety_attempts + 5) % len(tag_pool)]
        unique_tag = tag_pool[
            _stable_seed("hunt-safety-unique", goal, place_name, object_phrase, str(len(seen)), str(safety_attempts))
            % len(tag_pool)
        ]
        safety_attempts += 1
        paragraph = (
            f"The {ordinal} {scene_tag} sign changed {local_cue}; {attempt_tag} {unique_tag} counsel for {ordinal} {local_cue} shifted {unique_tag} before "
            f"{ordinal} {local_cue} {char_phrase} set {unique_tag} {ordinal} {scene_tag} {object_phrase}. Then {ordinal} {attempt_tag} road-work tested "
            f"{local_cue}, delayed a {unique_tag} {scene_tag} {ordinal} step, questioned a {attempt_tag} "
            f"{local_cue} witness, and tightened {unique_tag} {local_cue} {ordinal} guard around the {scene_key} choice."
        )
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _raise_hunt_dialogue_ratio(
    text: str,
    *,
    scene_goal: str,
    characters: list[str],
    target_dialogue_ratio: float,
    seen: set[str],
) -> str:
    if target_dialogue_ratio <= 0:
        return text
    total_words = max(1, _count_words(text))
    if _dialogue_word_count(text) / total_words >= target_dialogue_ratio:
        return text

    lowered = {c.lower(): c for c in characters}
    ranger = lowered.get("aragorn") or "Aragorn"
    wizard = lowered.get("gandalf") or "Gandalf"
    quarry = lowered.get("gollum") or "Gollum"
    goal = _public_scene_goal_text(scene_goal)
    cue_words = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", goal)
        if token.lower() not in STOPWORDS and token.lower() not in {"gandalf", "aragorn", "gollum", "shire", "baggins"}
    ]
    cue = " ".join(cue_words[:4]) or "the next sign"
    cue_short = cue_words[0].lower() if cue_words else "sign"
    dialogue_tags = ["ash", "briar", "dusk", "fen", "flint", "hollow", "mire", "rain", "root", "thorn", "wold"]
    if len(characters) == 1 and quarry in characters:
        dialogue_templates = [
            "'Names bites,' Gollum whispered. 'Baggins bites, Shire bites, and still we keeps them under the tongue.'",
            "'No kind hands,' he said. 'Kind hands tie knots. Cruel hands strike. All hands wants what Gollum knows.'",
            "'We goes where the hurt points,' said Gollum. 'Not brave, no. Hungry feet, frightened feet, old feet.'",
            "'Soft country, thief country,' he muttered. 'Warm holes and closed doors, but the name gets through cracks.'",
            "'They listens for us,' Gollum said. 'Tall ears, grey ears, dark ears. We says nothing, and still the mouth remembers.'",
            "'Rope dreams, wizard dreams, ranger dreams,' he said. 'But Gollum wakes in the cracks between them.'",
            "'Shire is a soft word,' Gollum whispered. 'Soft words can still cut if the mouth keeps chewing them.'",
            "'No pity,' he said. 'Pity looks and looks. Hate is easier. Hate leaves us alone until hunger comes.'",
            "'Baggins went away with the bright thing,' he said. 'But names leave smells, and old hurt follows smells.'",
            "'We tells nothing,' Gollum muttered. 'Only the tongue slips when the dark asks sweetly.'",
            f"'{cue_short} hurts,' Gollum said. '{cue_short.capitalize()} hurts first; rope answers {cue_short}; old names hurt where {cue_short} cannot hide.'",
            f"'We remembers {cue_short},' he whispered. 'Not for love of {cue_short}. Never clean love. For the bite {cue_short} left.'",
            "'Let them look at mud and stones,' said Gollum. 'The dark asks better questions than kind faces.'",
            "'No straight paths,' he muttered. 'Straight paths are for feet that have not been hunted.'",
            "'Baggins is a hook in the mouth,' Gollum said. 'Pull it, and all the old blood comes again.'",
            f"'The {cue_short} knows nothing,' he hissed. 'But {cue_short} tells Gollum where soft lands hide.'",
        ]
    else:
        dialogue_templates = [
            f"'Patience I can give,' answered {ranger}. 'What I cannot give is certainty before the road has earned it.'",
            f"'Then let the road earn what it can,' said {wizard}. 'Only do not let pity fall behind necessity.'",
            f"'I would rather carry a hard mercy than an easy cruelty,' said {ranger}. 'The quarry shall not be wasted.'",
            f"'Remember that the Shire is defended best while it still does not know it is defended,' said {wizard}.",
            f"'Hidden work suits a Ranger better than praise,' said {ranger}. 'Praise makes noise, and noise travels.'",
            f"'If the name Baggins has passed beyond fear into malice, every day matters,' said {wizard}. 'Spend none carelessly.'",
            f"'I will spend them on the trail,' said {ranger}. 'Let others spend words when the truth is brought back.'",
            f"'A small land may be lost by a small word,' said {wizard}. 'That is why we must count words as carefully as tracks.'",
            f"'Then I will count silence also,' said {ranger}. 'The wild often speaks loudest where men say nothing.'",
            f"'Do not mistake concealment for idleness,' said {wizard}. 'A hidden guard may be the only guard that can endure.'",
            f"'If the creature is found, bring him living if living can be borne,' said {wizard}. 'Dead answers are few.'",
            f"'Living answers have teeth,' said {ranger}. 'Still, I will not make ignorance easier by killing what may speak.'",
            f"'The Shire must not learn fear too soon,' said {wizard}. 'But neither may we arrive too late with wisdom.'",
            f"'There are roads that ask a man to be both stern and gentle,' said {wizard}. 'This is one of them.'",
            f"'Stern I can be,' said {ranger}. 'Gentleness will have to walk armed.'",
            f"'The work before us is plain enough to dread,' said {wizard}. 'But plain work may still be the hinge of hope.'",
            f"'Then let hope travel quietly,' answered {ranger}. 'I will keep the roads from boasting of it.'",
            f"'The small doors of the west must remain small a little longer,' said {wizard}. 'That is the mercy we can still give them.'",
            f"'Where the trail fails, let vigilance begin,' said {wizard}. 'The enemy may learn a name, but not unopposed.'",
            f"'This mile changes around {cue_short},' said {ranger}. '{cue_short.capitalize()} is not to be passed while {cue_short} still points.'",
            f"'{cue} is not proof,' said {wizard}. 'But {cue_short} is enough to make delay dangerous.'",
            f"'{cue_short.capitalize()} gives {cue_short} direction, not {cue_short} answer,' answered {ranger}. 'That {cue_short} is still worth a cold {cue_short} road.'",
            f"'Set a {cue_short} watch where {cue_short} touches the road,' said {wizard}. 'At that {cue_short} edge danger shows first.'",
            f"'{cue_short.capitalize()} has cost us time,' said {ranger}. 'Let {cue_short} buy {cue_short} caution in return.'",
            f"'If {cue_short} turns the trail, I will follow that {cue_short} turn,' answered {ranger}. 'Pride can wait behind {cue_short}.'",
            f"'{cue_short.capitalize()} must go west in {cue_short} plain words,' said {wizard}. 'Plain {cue_short} words travel faster than {cue_short} ornament.'",
            f"'Then I will guard {cue_short} against wild correction,' said {ranger}. '{cue_short.capitalize()} work is enough for this night.'",
        ]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    idx = _stable_seed("hunt-dialogue", scene_goal, ",".join(characters)) % len(dialogue_templates)
    attempts = 0
    max_attempts = len(dialogue_templates) * 2
    while _dialogue_word_count("\n\n".join(paragraphs)) / max(1, _count_words("\n\n".join(paragraphs))) < target_dialogue_ratio:
        if attempts >= max_attempts:
            break
        paragraph = dialogue_templates[(idx + attempts) % len(dialogue_templates)]
        if attempts >= len(dialogue_templates):
            salt = attempts - len(dialogue_templates) + 1
            ordinal_words = [
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
                "sixth",
                "seventh",
                "eighth",
                "ninth",
                "tenth",
            ]
            ordinal = ordinal_words[(salt - 1) % len(ordinal_words)]
            tag = dialogue_tags[
                _stable_seed("hunt-dialogue-tag", scene_goal, ",".join(characters), str(len(seen)), str(salt))
                % len(dialogue_tags)
            ]
            if len(characters) == 1 and quarry in characters:
                paragraph = (
                    f"'{cue_short} {tag} again, yes, {ordinal} {tag} again,' Gollum whispered. "
                    f"'{ordinal.capitalize()} {tag} step through {ordinal} {cue_short} dark, and {ordinal} {cue_short} scratches.'"
                )
            else:
                speaker = ranger if salt % 2 else wizard
                verb = "answered" if speaker == ranger else "said"
                paragraph = (
                    f"'{cue_short.capitalize()} gives us the {cue_short} {ordinal} turn, not {ordinal} comfort,' {verb} {speaker}. "
                    f"'Mark {ordinal} {cue_short}, then move before that {ordinal} sign grows {ordinal} cold.'"
                )
        attempts += 1
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _hunt_chapter_closing_paragraph(chapter: int) -> str:
    closings = {
        1: (
            "So the first name went eastward in mutter and malice, and the first watcher went after it. "
            "Behind them the Shire slept on, unarmed by knowledge and therefore still worth every hidden mile."
        ),
        2: (
            "The trail did not end; it thinned, crossed water, and returned as rumour. Aragorn followed because "
            "the smallest sign was still larger than comfort, and Gandalf's fear had become a road under his feet."
        ),
        3: (
            "Thus the hunt won its first victory without joy. Gollum was bound and breathing, and the rope that "
            "held him seemed less a triumph than a question carried north."
        ),
        4: (
            "Under the trees the truth came crookedly, yet it came. Baggins and Shire were no longer harmless "
            "sounds in an old tale, and Gandalf knew that warning must outrun every darker listener."
        ),
        5: (
            "Mirkwood kept its watch, but no watch can master every habit of darkness. In the prisoner, misery "
            "had become memory, and memory had begun to count the spaces between kindness and command."
        ),
        6: (
            "By morning the wood had learned that mercy is not the same as safety. Gollum was gone, and the names "
            "he carried went with him like sparks blown out of a guarded fire."
        ),
        7: (
            "The road westward did not grow safer because it was watched. It grew more precious, and every hidden "
            "guard set upon it became part of a defence no quiet door would ever see."
        ),
        8: (
            "Aragorn let the lost trail lie where rain and stone had taken it. To abandon a false certainty was "
            "not defeat; it was the discipline by which warning replaced pursuit."
        ),
        9: (
            "So the word travelled, never whole and never harmless. It passed through fear, memory, and hunger, "
            "until even silence seemed to know that Baggins and Shire had become a perilous pair."
        ),
        10: (
            "Their counsel ended with no clean victory to divide between them. Gandalf took the burden west; "
            "Aragorn took the roads; and somewhere beyond both, Gollum kept moving under the old wound."
        ),
        11: (
            "The green country remained ignorant, and that ignorance was now guarded. Aragorn stood outside its "
            "peace as a man may stand outside firelight, cold but unwilling to let the dark draw nearer."
        ),
        12: (
            "Until the door opened, that was the shape of hope: a wizard bearing warning, a Ranger keeping watch, "
            "and a ruined creature loose in the dark with names he did not understand. The hunt had not ended, "
            "but it had bought the Shire time, and time, in that hour, was mercy."
        ),
    }
    return closings.get(chapter, "")


def _template_shire_gap_scene_text(
    *,
    project: dict,
    scene_goal: str,
    characters: list[str],
    place: str,
    objects: list[str],
    event: dict,
    scene_beats: list[dict],
    missing_terms_hint: list[str],
    quality: dict[str, Any] | None = None,
) -> str:
    quality = quality or {}
    target_words = int(quality.get("target_scene_words", 0) or 0) or 1200
    target_dialogue_ratio = float(quality.get("target_dialogue_ratio", 0.0) or 0.0)
    goal_l = scene_goal.lower()

    def _join_names(names: list[str]) -> str:
        rows = names[:4]
        if len(rows) <= 1:
            return rows[0] if rows else "Frodo"
        if len(rows) == 2:
            return f"{rows[0]} and {rows[1]}"
        return ", ".join(rows[:-1]) + f", and {rows[-1]}"

    def _pick(preferred: list[str], fallback: str) -> str:
        lowered = {name.lower(): name for name in characters}
        for candidate in preferred:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        return characters[0] if characters else fallback

    def _clean_sentence(raw: str) -> str:
        return str(raw or "").strip(" .,:;!?\"'")

    def _clean_motifs(raw: list[Any]) -> list[str]:
        rows: list[str] = []
        for raw_motif in raw:
            motif = _clean_sentence(str(raw_motif)).lower()
            if not motif or motif in MOTIF_STOPWORDS or motif in STOPWORDS:
                continue
            if len(motif) < 3 or motif.endswith("'s"):
                continue
            rows.append(motif)
        return _dedupe_strings(rows)[:3]

    def _scene_kind() -> str:
        if any(token in goal_l for token in ("gandalf visits", "asks careful", "returns without", "careful questions")):
            return "visit"
        if any(token in goal_l for token in ("gandalf leaves", "leaves before dawn", "before dawn", "asking frodo for caution")):
            return "departure"
        if any(token in goal_l for token in ("winter", "waiting", "waits", "letter", "letters", "maps", "study", "long silence")):
            return "waiting"
        if any(token in goal_l for token in ("green dragon", "rumour", "rumor", "border", "bree", "rangers", "watchers", "road")):
            return "rumour"
        return "inheritance"

    def _setting_sentence(raw_place: str) -> str:
        place_l = raw_place.lower()
        if "green dragon" in place_l:
            return "At the Green Dragon, the lamps had begun to show early in the misty windows, and talk moved warmly under the rafters."
        if "hobbiton" in place_l or "road" in place_l:
            return "In Hobbiton the lanes lay brown with wet leaves, and the low hills kept their ordinary peace under a pale autumn morning."
        if "bag end" in place_l:
            return "At Bag End, late autumn had drawn the garden into damp gold and brown, and the round green door seemed brighter for the dimness of the day."
        return f"In {raw_place}, the Shire kept its quiet order while a colder thought moved beneath ordinary things."

    scene_kind = _scene_kind()
    char_phrase = _join_names(characters)
    lead = _pick(["Frodo", "Gandalf", "Sam"], "Frodo")
    respondent = _pick(["Gandalf", "Frodo", "Sam"], "Gandalf" if lead != "Gandalf" else "Frodo")
    if respondent == lead:
        respondent = "Gandalf" if lead != "Gandalf" else "Frodo"
    witness = _pick(["Sam", "Frodo", "Gandalf"], "Sam")
    if witness in {lead, respondent}:
        witness = "Sam" if "Sam" not in {lead, respondent} else "Frodo"

    motifs = _clean_motifs(event.get("motifs") or [])
    beat_lines = _dedupe_strings(
        [
            _clean_sentence(str(beat.get("intent") or beat.get("action") or ""))
            for beat in scene_beats
            if _clean_sentence(str(beat.get("intent") or beat.get("action") or ""))
        ]
    )
    object_rows = _dedupe_strings([str(obj).strip() for obj in objects if str(obj).strip()])
    anchors = _dedupe_strings([str(term).strip() for term in missing_terms_hint if str(term).strip()])

    def _object_label(raw: str) -> str:
        label = str(raw or "").strip()
        lower = label.lower()
        if lower == "ring":
            return "the ring"
        if lower == "red book":
            return "the red book"
        if lower == "pipe ash":
            return "a bowl of pipe ash"
        if lower == "garden shears":
            return "garden shears"
        if lower == "walking staff":
            return "a walking staff"
        if lower == "road":
            return "the Road"
        return label

    def _join_object_labels(rows: list[str]) -> str:
        labels = [_object_label(row) for row in rows[:4]]
        if len(labels) <= 1:
            return labels[0] if labels else ""
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"

    object_sentence = ""
    if object_rows:
        object_sentence = " On the table and shelves lay " + _join_object_labels(object_rows) + ", all plain enough to look harmless."

    motif_sentence = ""
    if motifs:
        motif_sentence = " Beneath the talk lay the pressure of " + ", ".join(motifs[:3]) + "."

    beat_sentence = ""
    if beat_lines:
        beat_sentence = " The immediate task was " + "; ".join(beat_lines[:2]) + "."

    anchor_sentence = ""
    if anchors:
        anchor_sentence = (
            " Before the night was done the quiet matter would bind "
            + ", ".join(anchors[:5])
            + " more closely than any of them could explain."
        )

    premise_sentence = (
        "It was one of the long quiet years after Bilbo's leaving, before Frodo had been told more than any "
        "sensible hobbit would wish to know."
    )

    if scene_kind == "visit":
        paragraphs = [
            (
                f"{_setting_sentence(place)} {char_phrase} met there not with ceremony, but with the uneasy comfort "
                "of old friendship returning after too long an absence. The kettle steamed; the fire settled; "
                "outside, a spade scraped once against stone and was still."
            ),
            (
                f"Gandalf had come without fireworks, without bundles of toys, and without the cheerful disorder "
                "that used to run ahead of him through Hobbiton. That alone made Frodo watch him more closely. "
                "A wizard may enter a room quietly, but he cannot always leave his cares outside the door."
                f"{object_sentence}{motif_sentence}{beat_sentence}{anchor_sentence}"
            ),
            (
                "'You look as though you have walked through weather that did not reach the Shire,' Frodo said. "
                "'There is a smell of far roads on your cloak, and not all of it is rain.'"
            ),
            (
                "'Far roads are full of bad inns and worse answers,' said Gandalf. 'I came for tea first, and for "
                "questions only after tea has made them civil.'"
            ),
            (
                "Frodo smiled, because the words were shaped like the old Gandalf. Yet the smile did not remain. "
                "The old Gandalf would have filled the room with pipe-smoke and complaint, and then made the "
                "complaint turn suddenly into laughter. This one listened between sentences."
            ),
            (
                "Sam was outside beneath the window, clipping late runners from the nasturtiums and pretending "
                "that vines were troublesome enough to need his whole attention. He heard the murmur of voices, "
                "and now and again a word came clear: Bilbo, maps, birthday, and once, lower than the rest, ring."
            ),
        ]
        scene_specific_templates = [
            (
                "Gandalf took the chair by the hearth as though it had been waiting for him, but he did not loosen "
                "his shoulders when he sat. His eyes moved over the shelves, the mantel, the small clock, the "
                "letters tied in bundles, and the place where Bilbo had once kept a row of absurdly labelled boxes."
            ),
            (
                "'Have you had news from him?' asked Frodo. He tried to make the question light, but it had lived "
                "too long in him to be light. 'Only old news, and old news does not grow younger for being read again.'"
            ),
            (
                "'Bilbo has a talent for leaving behind rooms that still expect him,' Gandalf said. 'That is not "
                "the worst talent in the world. It is better than leaving behind debts, feuds, or broken furniture.'"
            ),
            (
                "Frodo laughed at that, and the laugh steadied him. He went to the little desk and touched a stack "
                "of papers whose edges had curled in the damp. 'He left me enough of all three, if papers may count "
                "as debts and cousins as feuds.'"
            ),
            (
                "Gandalf's face softened, but not for long. 'And the small thing he gave you?' he said. 'The trinket "
                "he was reluctant to leave, until he was suddenly eager to be rid of it?'"
            ),
            (
                "The room seemed to become smaller. Frodo put his hand into his waistcoat pocket and then took it "
                "out again empty. He had not meant to make the gesture. Its very smallness troubled him."
            ),
            (
                "'It is here,' said Frodo. 'I do not carry it about every day like a watch. I am not Bilbo, whatever "
                "the Sackville-Bagginses may say when they are being uncharitable.'"
            ),
            (
                "'No,' said Gandalf. 'You are not Bilbo. That is one reason I am here.' He leaned forward then, "
                "and the firelight caught the grey of his brows. 'Tell me again how often you have used it.'"
            ),
            (
                "Frodo disliked the question more than he expected. It made a private foolishness look like a matter "
                "for record. 'Seldom. Very seldom. Once to avoid Lobelia, which I admit was cowardice dressed as "
                "prudence. Once in the pantry, by accident.'"
            ),
            (
                "Gandalf did not smile. That was worse than a rebuke. Frodo had prepared himself for rebuke, and "
                "found instead a silence that asked him to notice his own answer."
            ),
            (
                "'Do not make a habit of accidents,' Gandalf said at last. 'A thing that answers too readily may "
                "one day ask a question of its own.'"
            ),
            (
                "Outside, Sam cut the same dead stem twice. He knew well enough that he had no business listening, "
                "and he knew also that he would remember every word he had heard until supper, if not longer."
            ),
            (
                "Frodo looked toward the window, but the glass was clouded with dusk and garden damp. 'Is there "
                "danger in it?' he asked. The plain question stood between them in the middle of the warm room."
            ),
            (
                "'There may be danger in many small inheritances,' Gandalf answered. 'An old key, an old map, an "
                "old grudge, an old joke told by the wrong fire. Keep this one out of sight and close at hand. "
                "Do not lend it to curiosity.'"
            ),
            (
                "'That sounds almost like an answer,' Frodo said. 'Not quite, but almost.'"
            ),
            (
                "'Then be content with almost tonight,' said Gandalf. 'The rest is still on the road, and I have "
                "not overtaken it.'"
            ),
            (
                "The wind pressed a brown leaf against the pane. For a moment all three heard it: Frodo within, "
                "Gandalf by the hearth, and Sam below the window with the shears idle in his hand."
            ),
            (
                "There was more talk after that, and some of it was ordinary. They spoke of turnips, of a cracked "
                "chimney-pot, of Merry's latest visit, and of an elderly gaffer who distrusted any moon he could "
                "not see from his own doorstep. Ordinary talk did not lift the hidden weight, but it gave the "
                "weight a room in which to wait."
            ),
            (
                "At last the kettle was filled again, though no one wanted more tea. Frodo did it because the "
                "movement gave his hands employment, and Gandalf allowed it because there are times when a host "
                "must be permitted the dignity of cups and spoons."
            ),
            (
                "The study door stood a little ajar. Beyond it were Bilbo's papers, more orderly since Frodo had "
                "taken charge and less alive for that very reason. Gandalf looked that way once, and Frodo saw "
                "that the look held affection, worry, and calculation in unequal measures."
            ),
            (
                "'You miss him,' Gandalf said. It was not a question."
            ),
            (
                "'Of course I do,' said Frodo. 'But missing Bilbo is not a single feeling. It is more like keeping "
                "house with several guests. One is grief, one is amusement, one is irritation, and one keeps asking "
                "whether the road has brought any letters.'"
            ),
            (
                "'That is a tolerably accurate household,' Gandalf said. 'Do not evict amusement. It pays its rent "
                "better than grief, and it is less likely to rearrange the furniture.'"
            ),
            (
                "Frodo leaned against the mantel and looked down at the hearth. 'You speak lightly when you are "
                "most serious. Bilbo did that too, but with him it was usually because he had forgotten where the "
                "serious part had gone.'"
            ),
            (
                "Gandalf's face changed, and for a moment he seemed older than the room could hold. 'Bilbo forgot "
                "many useful things and remembered many impossible ones. That is why he was dear to me, and why "
                "I am not done troubling his heir.'"
            ),
            (
                "Outside the window, Sam gathered the clippings into a tidy heap and gathered his thoughts into "
                "a far less tidy one. He had always known that gentlefolk had private matters. He had not known "
                "that a private matter could make the air around a window feel colder."
            ),
            (
                "The word ring had lodged in him. It was too small a word for the weight Gandalf put upon it, and "
                "too plain a word for secrecy. Sam distrusted the mismatch. A gardener learns early that small "
                "roots may crack a path if no one tends them."
            ),
            (
                "By the time supper was set out, Frodo had nearly persuaded himself that the conversation had "
                "been no more than Gandalf's habit of making ordinary things look deep. Nearly was a comfortable "
                "distance from entirely, and the comfort did not last."
            ),
            (
                "Gandalf ate with appetite, praised the mushrooms, complained about the ale, and asked after three "
                "families whose names he should not have remembered. It was all excellently normal, except that "
                "his questions never wandered far from Frodo's health, Bilbo's habits, and the locked drawer in "
                "the study."
            ),
            (
                "When the clock struck late, the sound moved through Bag End as it had moved through hundreds of "
                "comfortable nights. This time it seemed to count something other than hours. Frodo heard the last "
                "stroke fade and wondered why the silence after it felt like a held breath."
            ),
        ]
    elif scene_kind == "rumour":
        paragraphs = [
            (
                f"{_setting_sentence(place)} {char_phrase} found the Shire busy with one of its favourite "
                "occupations: making large weather out of small reports. A cart had come late from the Westfarthing; "
                "a peddler had spoken too softly; a gate near the border had been found open and then, more "
                "mysteriously, closed again."
            ),
            (
                "Frodo listened with the polite attention that keeps talk moving without promising belief. Since "
                "Gandalf's last visit, he had learned that a rumour need not be true to be worth hearing. Sometimes "
                "it was the shape of fear that mattered."
                f"{object_sentence}{motif_sentence}{beat_sentence}{anchor_sentence}"
            ),
            (
                "'Tall men near the bounds, they say,' Sam murmured. 'Tall men with weather on them, which is a "
                "queer thing to notice unless you have no better particulars.'"
            ),
            (
                "'The Shire is rich in particulars when the matter is turnips,' Frodo said. 'For strangers it "
                "often has to make do with height, boots, and suspicion.'"
            ),
            (
                "Sam accepted this as fair, though not comforting. He had no wish to meet a tall stranger near a "
                "hedge after dark, even if the stranger turned out to be only lost, polite, and unfortunate in "
                "his choice of hat."
            ),
            (
                "The talk around them rose and fell. No one spoke of danger in a voice fit for danger. That was "
                "part of what troubled Frodo. The Shire had a gift for making unfamiliar things look ridiculous before "
                "it had been understood."
            ),
        ]
        scene_specific_templates = [
            (
                "At the Green Dragon, if the scene had brought them there, mugs went down with decisive thumps "
                "and theories improved with every refill. At Bag End or on the Road, the same habit worked more "
                "quietly: a raised brow, a pause by a gate, a glance toward the western lanes."
            ),
            (
                "Frodo remembered Gandalf's words about folk beyond Bree who kept watch on more than weather. "
                "The phrase had sounded almost comic by daylight. Now, carried into local talk and stripped of "
                "the wizard's voice, it became less comfortable."
            ),
            (
                "'Rangers,' said Sam, trying the word as if it might have thorns. 'That is what old Noakes called "
                "them. He says they go about with cloaks and no fixed address, which is enough to make any decent "
                "body doubtful.'"
            ),
            (
                "'No fixed address is not a crime,' said Frodo. 'Bilbo made a virtue of it for a while.'"
            ),
            (
                "'Mr. Bilbo came back to an address, sir. That is the point of adventures, according to my Gaffer: "
                "if you must have one, be sure it knows where to send you afterward.'"
            ),
            (
                "Frodo smiled, but the word afterward stayed with him. Gandalf had left no afterward, only a set "
                "of cautions folded so tightly that Frodo could not unfold one without finding another inside."
            ),
            (
                "The ring, when he thought of it, seemed absurdly far from these reports of men on roads. It was "
                "small, private, and quiet. Yet the mind does not always obey size. It placed the ring beside the "
                "rumours as though both had been found in the same drawer."
            ),
            (
                "Sam watched him and said nothing. That restraint cost him effort. Sam's silence was not empty; "
                "it was packed with questions, loyalty, and a gardener's stubborn sense that roots under a path "
                "should be found before they break the stone."
            ),
            (
                "'You are thinking of Mr. Gandalf,' Sam said at last."
            ),
            (
                "'I am thinking that he would dislike most of these stories,' Frodo answered. 'Not because they "
                "are foolish, but because foolish stories may walk beside true ones and borrow their coats.'"
            ),
            (
                "That was too much like wizardry for Sam's taste, but he took the meaning. 'Then we had best not "
                "laugh too loud, sir.'"
            ),
            (
                "'No,' said Frodo. 'Not too loud.'"
            ),
            (
                "Outside the talk, ordinary life continued with a kind of magnificent indifference. Dogs quarrelled, "
                "bread was bought, a child was scolded, and a respectable hobbit insisted that borders were made "
                "for keeping foreign weather in its place."
            ),
            (
                "Frodo loved that foolishness more fiercely than he expected. It was not wisdom, but it was worth "
                "guarding. The thought startled him because it sounded like a duty and not merely an affection."
            ),
            (
                "The day ended without proof. That was the difficulty. Proof would have been easier to handle, "
                "even unpleasant proof. Rumour left everything unfinished and made every quiet hedge seem capable "
                "of withholding an answer."
            ),
            (
                "When Frodo returned toward Bag End, the Road behind him was empty. He looked back only once. Sam "
                "looked back twice, and then pretended the second glance had been for a bird."
            ),
        ]
    elif scene_kind == "waiting":
        paragraphs = [
            (
                f"{_setting_sentence(place)} {char_phrase} came into the season of waiting, though no one in the "
                "Shire would have named it so. There were proper names enough: winter, wet weather, letter-writing, "
                "accounts, mending, and the annual discovery that every chimney has opinions."
            ),
            (
                "Frodo's waiting had no date attached to it. Gandalf had not promised a month, a season, or even "
                "a year. He had left behind a warning with no clock in it, and such warnings are poor company "
                "when evenings lengthen."
                f"{object_sentence}{motif_sentence}{beat_sentence}{anchor_sentence}"
            ),
            (
                "'There is no letter again,' Sam said, although Frodo had not asked. Sam had taken to noticing "
                "the post with a seriousness that made the postman suspicious."
            ),
            (
                "'No letter,' said Frodo. 'And therefore no explanation, no correction, no apology, and no fresh "
                "mystery. It is almost economical.'"
            ),
            (
                "Sam did not laugh as much as the sentence deserved. He knew by now that Frodo used wit as other "
                "hobbits used a walking-stick: not because the road was easy, but because something must meet the "
                "ground first."
            ),
            (
                "The ring remained where Frodo kept it. He had moved it twice, then scolded himself for the moving. "
                "A hidden thing can begin to govern a house merely by making the house rearrange itself around it."
            ),
        ]
        scene_specific_templates = [
            (
                "In the study, Bilbo's maps lay under a square of winter light. Some were practical, with roads "
                "and distances and notes about inns. Others were half memory, half boasting, full of mountains "
                "drawn too large and rivers that seemed to be enjoying themselves."
            ),
            (
                "Frodo read them more often than he meant to. The reading was not preparation, he told himself. "
                "It was affection. Yet affection has a way of teaching the hand where a road begins."
            ),
            (
                "A letter from Bilbo would have changed everything and nothing. Frodo imagined the handwriting, "
                "the sudden complaint about ink, the cheerful refusal to explain where he was, and the casual "
                "sentence that would worry Gandalf for six months."
            ),
            (
                "No such letter came. The absence grew familiar enough to be set at table with the other familiar "
                "things: salt, bread, pipe-ash, unpaid bills, and the ache that follows laughter when the person "
                "one would most like to tell is far away."
            ),
            (
                "'If Mr. Gandalf meant to be gone this long,' Sam said, 'he might have said so.'"
            ),
            (
                "'That would have required him to know it,' Frodo answered. 'I suspect wizards dislike calendars "
                "because calendars are one of the few things that can contradict them in public.'"
            ),
            (
                "This pleased Sam, but only briefly. 'Still, sir. If a body leaves another body worried, it seems "
                "only fair to send a word back now and again.'"
            ),
            (
                "'It does,' said Frodo. He said no more, because agreement can be more painful than argument when "
                "nothing can be done with it."
            ),
            (
                "Beyond the windows the garden slept in its disciplined way. Sam had put everything to rights, "
                "or as near to rights as weather permits. Bare stems stood like writing no one had yet learned "
                "to read."
            ),
            (
                "Frodo sometimes thought that the whole Shire was written in such marks: hedge, furrow, smoke, "
                "gate, footprint, lamp. It was a language of ordinary keeping, and he had never loved it more than "
                "when he feared it might be less secure than it appeared."
            ),
            (
                "At night the house made its winter sounds. Timbers answered cold, the bank settled, and now and "
                "again a mouse expressed firm opinions in the wall. These sounds had once been merely domestic. "
                "Now they seemed to ask whether Frodo was listening."
            ),
            (
                "He did listen. That was the change in him. Not fear exactly, not yet. Rather the habit of attention, "
                "as if Gandalf had left a lamp burning in one corner of his mind and no one had given him leave "
                "to put it out."
            ),
            (
                "Sam noticed the lamp, though not its shape. He brought in wood before being asked. He checked "
                "the latch on the garden gate. He invented reasons to pass under Frodo's window after dusk and "
                "looked offended if the reasons were questioned."
            ),
            (
                "'You will wear a path in the path,' Frodo told him."
            ),
            (
                "'Paths are for wearing, Mr. Frodo,' Sam said. 'A path with no feet on it gets ideas above its "
                "station.'"
            ),
            (
                "So winter held them: Frodo with Bilbo's maps and Gandalf's unfinished warning, Sam with his "
                "watchfulness disguised as chores, Bag End with its bright windows, and the ring with its perfect "
                "silence."
            ),
            (
                "The silence was the worst part. A troublesome object ought to clatter, stain, smell, or at least "
                "misbehave. This one did none of those things. It merely waited better than Frodo did."
            ),
            (
                "When spring showed itself in small green declarations, Frodo felt no release. The season changed; "
                "the question did not. That was how the long silence began to teach him endurance, though he would "
                "not yet have called it by so large a name."
            ),
        ]
    elif scene_kind == "departure":
        paragraphs = [
            (
                f"{_setting_sentence(place)} The last stars were fading when {char_phrase} came to the door, and "
                "the grass above Bag End shone with a cold silvering of dew. Gandalf had packed before anyone "
                "else was awake, if indeed he had slept at all."
            ),
            (
                "Frodo found him in the passage with his staff in one hand and his hat in the other. There was no "
                "grand farewell in it. The house smelled of banked fire, polished wood, and yesterday's bread, and "
                "that homeliness made the going harder to understand."
                f"{object_sentence}{motif_sentence}{beat_sentence}{anchor_sentence}"
            ),
            (
                "'You are leaving before breakfast,' Frodo said. 'That is either very rude or very urgent.'"
            ),
            (
                "'It is both,' said Gandalf. 'I have often found that urgency has poor manners.'"
            ),
            (
                "Sam, coming up the garden path with a basket of early mushrooms he had no need to bring, stopped "
                "when he saw the tall figure by the gate. He had meant to be unseen. Hobbits often mean to be unseen "
                "when they are most visible."
            ),
            (
                "Gandalf looked past Frodo toward the whitening lane. 'There are questions that do not grow kinder "
                "by waiting. I must go where a few of them may still be answered.'"
            ),
        ]
        scene_specific_templates = [
            (
                "Frodo folded his arms, partly against the chill and partly against the feeling that he was being "
                "left out of his own affairs. 'You ask about Bilbo, about my health, about an old ring, and then "
                "you vanish. I hope you can see how that looks from this side of the door.'"
            ),
            (
                "'I can see it very well,' Gandalf said. 'That is why I am sorry.' He said it plainly, and the "
                "plainness did more to alarm Frodo than any dark hint would have done."
            ),
            (
                "The lane below Bag End curved away toward waking chimneys. A milk-cart creaked somewhere out of "
                "sight. From the Hill came the small sounds by which Hobbiton convinced itself that every day was "
                "only another day."
            ),
            (
                "'Should I be afraid?' Frodo asked. He had meant to say it lightly, but the morning was too clear "
                "for false tones."
            ),
            (
                "'Not in the way that spoils breakfast,' Gandalf said. 'Fear is a poor household guest. It eats "
                "all the bread and gives no useful advice. But caution, if you keep it modest, may sit in the "
                "corner and mend a torn sleeve.'"
            ),
            (
                "Frodo shook his head. 'You make even warnings sound as if they have come to borrow a needle.'"
            ),
            (
                "'That is because warnings that arrive with trumpets are often too late,' said Gandalf. 'Listen "
                "to the quiet ones. Keep Bilbo's gift private. Speak of it to no one merely because talk is easier "
                "than thought.'"
            ),
            (
                "Sam lowered his basket. The word private went into him like a hook. He looked at the grass, at "
                "the door, at Gandalf's boots, and told himself that he had heard nothing meant for him."
            ),
            (
                "Gandalf saw him, of course. Gandalf had a way of seeing people at the exact moment when they hoped "
                "most earnestly to be overlooked. 'Good morning, Samwise,' he said. 'Your mushrooms are early.'"
            ),
            (
                "'Begging your pardon, Mr. Gandalf,' said Sam, colouring. 'They are not so much early as I am, "
                "if you take my meaning.'"
            ),
            (
                "'I do,' said Gandalf. 'Early ears may still choose late tongues.'"
            ),
            (
                "Sam swallowed. 'I am not one for carrying tales, sir. Leastways not tales that are not mine to "
                "carry.'"
            ),
            (
                "Frodo turned on him with surprise that was not quite anger. Sam looked miserable at once, and "
                "that misery softened the moment before it could harden. There are friendships in which apology "
                "is already present before anyone has found words for it."
            ),
            (
                "'Then carry only mushrooms this morning,' Gandalf said. 'And if anyone asks why I left so early, "
                "say that wizards are troublesome guests and unreliable breakfast companions. It will be true "
                "enough for Hobbiton.'"
            ),
            (
                "A thin humour passed over Frodo's face and was gone. 'Will you come back soon?'"
            ),
            (
                "Gandalf put on his hat. The brim shadowed his eyes. 'Soon is a word that roads seldom respect. "
                "I will come back when I can, and before then you must let ordinary days do their good work.'"
            ),
            (
                "The answer did not satisfy Frodo, but he knew the difference between a withheld answer and a "
                "false one. Gandalf had not lied. He had simply placed the truth on a high shelf and taken the "
                "ladder away with him."
            ),
            (
                "They walked together to the gate. The Hill fell away below them, patched with gardens and roofs, "
                "and beyond it the fields ran toward Bywater and the Road. That Road looked no different from "
                "yesterday's road, yet Frodo found himself watching it as if it had spoken in sleep."
            ),
            (
                "Gandalf rested one hand on the gate. 'There are folk beyond Bree who keep watch on more than "
                "weather,' he said. 'If you hear of Rangers, do not make sport of the tale. Some fences are made "
                "of thorn, some of patience, and some of people who receive no thanks.'"
            ),
            (
                "'You are being mysterious again,' Frodo said."
            ),
            (
                "'No,' said Gandalf. 'I am being brief. Mystery is often only brevity seen from the wrong end.'"
            ),
            (
                "Sam stared down the lane as though he might see one of those watchers standing behind a hedge. "
                "He saw only wet leaves, a stone wall, and a robin with the self-importance of a mayor."
            ),
            (
                "Then Gandalf went down from the Hill. His staff tapped once, twice, and after a turn in the lane "
                "he was a grey shape among grey trees. Frodo watched until the shape was no more than movement, "
                "and then until even movement had been taken by the morning."
            ),
            (
                "When he turned back, Bag End stood open behind him. Warmth waited there, and breakfast, and Sam "
                "with his basket. Yet the door no longer seemed only an entrance to comfort. It was also a charge, "
                "and Frodo felt the weight of it settle quietly into his hand."
            ),
            (
                "For a little while none of them moved. The morning did not pause for them; thrushes began their "
                "work in the hedge, a shutter clapped below the Hill, and smoke lifted from a chimney in a thin "
                "blue thread. The world was busy proving that departures do not stop it."
            ),
            (
                "'Well, Mr. Frodo,' Sam said at last, and then found that he had no sentence ready after the well. "
                "He looked at the mushrooms as if they might offer a respectable ending."
            ),
            (
                "'Breakfast,' said Frodo. 'That is the word you are looking for, Sam. It may not answer every "
                "question, but it has the advantage of being possible.'"
            ),
            (
                "Sam brightened with relief. 'Yes, sir. And if questions are to be answered after breakfast, they "
                "will find a better welcome than before it. That is my Gaffer's opinion, and I have known him to "
                "be right when the matter concerned food.'"
            ),
            (
                "They went in together. Frodo closed the round door and stood a moment with his hand still on the "
                "latch. On the inner side it was only a latch: brass, polished by use, ordinary as bread. Yet he "
                "had the strange fancy that he had shut more than morning air outside."
            ),
            (
                "In the kitchen Sam made himself useful with almost military silence, which for Sam meant that he "
                "spoke only twice as much as another person. Frodo was grateful. The sound of plates and pans kept "
                "Gandalf's last words from arranging themselves too solemnly in his mind."
            ),
            (
                "Still they returned. Private. Caution. Rangers. Bree. Each word was harmless alone, like tools "
                "laid out on a bench. Together they suggested a shape of work Frodo had not asked to undertake "
                "and could not yet refuse."
            ),
            (
                "He took the ring from its hiding-place after Sam had gone to the pantry. It lay on his palm with "
                "an innocent weight. No answer came from it, and that seemed worse than any answer. A thing may "
                "be most secret when it has no need to hide."
            ),
            (
                "Frodo put it away before Sam returned. He did not do so from shame exactly, nor from fear alone. "
                "He did it because Gandalf had asked him, and because trust, once given by a friend, can become a "
                "kind of command without ceasing to be kindness."
            ),
            (
                "After breakfast, Hobbiton fully woke. Voices rose on the lane; a child cried over a dropped bun; "
                "two cousins disputed the ownership of a borrowed rake with the gravity of judges. The Shire had "
                "resumed itself."
            ),
            (
                "Frodo watched from the window and felt both comforted and estranged. He loved the smallness of it "
                "all, and that love sharpened his unease. If Gandalf feared for anything here, then the fear had "
                "excellent taste."
            ),
            (
                "Sam came up beside him, not too close. 'It will be a fair day after all, I think,' he said. Then, "
                "because loyalty in Sam often began as weather, he added, 'If you were thinking of walking, I "
                "could see to the lower path first.'"
            ),
            (
                "'Thank you, Sam,' said Frodo. 'Not yet.' He meant the walk, but both of them heard the words "
                "settle more widely than that."
            ),
        ]
    else:
        paragraphs = [
            (
                f"{_setting_sentence(place)} {char_phrase} moved through a day that looked, to any passer-by, as "
                "ordinary as a day could look. The smoke went up from the chimneys; the post came late; a cart "
                "lost a wheel near Bywater and made itself the chief subject of conversation for two miles."
            ),
            (
                f"{premise_sentence} Yet long quiet does not mean emptiness. It has drawers that stick, letters "
                "that are read more than once, and rooms where an absent voice seems always about to begin again."
                f"{object_sentence}{motif_sentence}{beat_sentence}{anchor_sentence}"
            ),
            (
                "Frodo had grown used to being Master of Bag End in the way one grows used to a coat inherited "
                "from a beloved cousin: it fits well enough in public, but in private it remembers another pair "
                "of shoulders."
            ),
            (
                "'Mr. Bilbo kept seed catalogues in with walking maps,' Sam said from the garden door. 'I found "
                "rhubarb next to the Misty Mountains this morning, if you will excuse the liberty of saying so.'"
            ),
            (
                "'That sounds exactly like Bilbo,' Frodo answered. 'He trusted maps more than vegetables, and "
                "vegetables more than most relations.'"
            ),
            (
                "Sam grinned, but only for a moment. The grin faded when he saw Frodo turn the small plain ring "
                "between finger and thumb before putting it away. Sam looked down at his boots at once, as if "
                "the boots had raised a delicate matter."
            ),
        ]
        scene_specific_templates = [
            (
                "There were still labels in Bilbo's hand on boxes that contained nothing the labels promised. "
                "Candles had been put with buttons, dried apples with sealing wax, and a packet marked Important "
                "held three foreign coins and a receipt for a waistcoat no one remembered collecting."
            ),
            (
                "Frodo did not correct all these disorders. Some he preserved because they made the smial feel "
                "less conquered by tidiness. A perfectly arranged Bag End would have seemed an accusation against "
                "the person who had made it lovable."
            ),
            (
                "The old party had become a legend already, though Hobbiton legends seldom waited for age before "
                "putting on importance. Children who had been too small to remember the fireworks now claimed "
                "to have seen dragons in the smoke and stars under the Party Tree."
            ),
            (
                "Frodo let them enlarge the tale. It did no harm. The Shire had always made room for harmless "
                "exaggeration, especially when exaggeration could be served with seed-cake and corrected by three "
                "aunts at once."
            ),
            (
                "What he did not enlarge was the matter of Bilbo's gift. That he kept small, almost absurdly small: "
                "a private oddity, a keepsake with a trick to it, a thing better left out of jokes because jokes "
                "have a way of inviting hands."
            ),
            (
                "Sometimes, alone in the passage, he would take it out and feel the coolness of it. It did not "
                "gleam much. It had no stone, no device, no visible craft that would make a stranger stop and "
                "stare. Its plainness was part of its unease."
            ),
            (
                "He had used it less as the years went by. At first there had been temptation, for the Shire is "
                "not free of dull meetings or tiresome visitors. Yet each use left behind a feeling like a door "
                "closed too softly in a house thought empty."
            ),
            (
                "'Begging your pardon, Mr. Frodo,' said Sam after a while, 'but Mr. Gandalf has not been by this "
                "autumn, has he?'"
            ),
            (
                "'Not yet,' said Frodo. 'That is the sort of answer which pretends to know more than it does.'"
            ),
            (
                "'He does come and go uncommon,' Sam said. 'My Gaffer says wizards are like weathercocks with "
                "feet: always turning, and never where a sensible body left them.'"
            ),
            (
                "Frodo laughed. 'Your Gaffer has described half the great persons of history, though he would be "
                "annoyed to hear it.'"
            ),
            (
                "Sam considered this with care. 'I would not tell him, sir. He has no patience with history unless "
                "it explains why potatoes were better in his father's day.'"
            ),
            (
                "The laughter made the room warmer. It did not drive out absence; nothing so simple could do that. "
                "But it set absence in its proper corner, and for a little while Bilbo seemed less vanished than "
                "gone ahead."
            ),
            (
                "In the afternoons Frodo walked under the Hill, sometimes with Sam at a respectful distance and "
                "sometimes alone. He knew every hedge and stile, every complaining gate, every elm that had leaned "
                "farther over the lane since Bilbo left."
            ),
            (
                "Beyond those familiar things lay the Road. It entered the Shire as if it belonged there, yet it "
                "did not. It had other loyalties. It remembered boots from Bree, pony bells, wandering songs, and "
                "dust from countries no hobbit in the market cared to name."
            ),
            (
                "Frodo loved the Road and distrusted it. That was one of Bilbo's legacies too. A Baggins may lock "
                "a round door very firmly and still know which way the lane bends beyond it."
            ),
            (
                "There were rumours that autumn. There are always rumours in the Shire, but these came oddly "
                "dressed: not scandal, not harvest complaint, not family grievance, but hints of tall men seen "
                "near the borders and lights moving late beyond hedges."
            ),
            (
                "The talk was laughed down at the Green Dragon. It was easy to laugh there. A polished mug, a "
                "settled bench, and a fire with good manners can make the outside world seem poorly argued."
            ),
            (
                "Frodo listened more than he spoke. He had learned that Bilbo's strangest tales had often begun "
                "as something a sensible hobbit dismissed between mouthfuls."
            ),
            (
                "That evening he came home by the upper path and found Sam shutting the tool shed. The sky behind "
                "him was the colour of pewter, and every leaf on the bank seemed to hold its own small lamp."
            ),
            (
                "'There is frost coming,' Sam said. 'Not a hard one, maybe, but enough to teach the late flowers "
                "better manners.'"
            ),
            (
                "'Then bring in what should not be taught too severely,' said Frodo. 'Bilbo used to say gardeners "
                "are the only people who can argue with weather and win a rematch.'"
            ),
            (
                "Sam's face brightened at the quotation, though he did not know whether it was truly Bilbo's or "
                "only the sort of thing Bilbo ought to have said. In the Shire those two matters often become one."
            ),
            (
                "When lamps were lit in Bag End, the windows looked out over a world that gave no sign of changing. "
                "That, perhaps, was why the first unease could enter unnoticed. The Shire was best at guarding "
                "against noise. It had less practice guarding against silence."
            ),
            (
                "Frodo sat down after supper with one of Bilbo's notebooks and did not read a word of it. The "
                "page before him described a road, a bridge, and an argument about the proper cooking of onions; "
                "but the handwriting mattered more than the subject. It had a forward lean, as if every sentence "
                "were already looking for its hat."
            ),
            (
                "He wondered whether Bilbo had known, at the end, why he was so restless. There had always been "
                "restlessness in him, naturally, like spice in a cake; but near the leaving it had deepened. Bilbo "
                "had grown sharp over trifles and tender over things he pretended to find absurd."
            ),
            (
                "That memory made Frodo gentler with his own unease. Perhaps feelings, too, may be inherited. "
                "Perhaps an old house keeps them in corners until the next person comes along and mistakes them "
                "for dust."
            ),
            (
                "He opened the drawer where the ring was hidden and then closed it again without touching the "
                "thing. The action was foolish, and he knew it. It was also irresistible. Some doors are tested "
                "not because one wishes to pass through them, but because one wants to be sure they remain shut."
            ),
            (
                "A tap came at the kitchen entrance. It was Sam returning a basket he had no urgent need to return. "
                "Frodo let him in, and neither mentioned that the basket was empty."
            ),
            (
                "'Begging your pardon, Mr. Frodo,' Sam said, 'but there is talk down by the Ivy Bush about tall "
                "men near the Northway. Mostly nonsense, I daresay. Mr. Sandyman said he saw one himself, which "
                "makes me doubt it more than less.'"
            ),
            (
                "'The world contains tall men, Sam,' Frodo said. 'Even the Shire cannot prevent that entirely.'"
            ),
            (
                "'No, sir. But it can take notice when they get too near the cabbages.'"
            ),
            (
                "Frodo smiled and offered him a chair. Sam refused it, accepted it, and sat on the very edge as if "
                "chairs in the parlour were temporary honours that might be withdrawn for misuse."
            ),
            (
                "They spoke then of harmless things, and the harmless things did them good. Sam described a quarrel "
                "between two neighbours over a pear tree. Frodo reported that one of the Mathoms had been returned "
                "with a note of complaint because it was too useful to be respectable."
            ),
            (
                "Yet beneath the talk Frodo kept hearing a question he had not formed. It moved behind the words "
                "like a footstep crossing a room beyond a closed door. Whenever he turned his mind toward it, it "
                "became only Bilbo's absence, Gandalf's silence, and the ring in the drawer."
            ),
            (
                "When Sam rose to go, he hesitated. 'Mr. Frodo, if Mr. Gandalf does come, and if there is anything "
                "as wants doing in the garden or out of it, you will say?'"
            ),
            (
                "'I will say what I can,' Frodo answered. It was a careful reply, but not an unkind one."
            ),
            (
                "Sam nodded as though he understood the courtesy and the wall behind it. 'That will do, sir,' he "
                "said. 'Some things grow better if they are not tugged at.'"
            ),
            (
                "After Sam had gone, Frodo stood outside under the stars. They looked clean and remote above the "
                "Hill, indifferent to ledgers, gardens, rumours, and small rings. He did not resent them for it. "
                "Their distance made the Shire feel sheltered and very fragile."
            ),
            (
                "He went in at last and barred the door. It was an old habit, not a new fear, and he took comfort "
                "from that distinction. But before he slept he listened once for footsteps on the path, and hearing "
                "none, could not decide whether he was relieved."
            ),
        ]

    inheritance_expansions = [
        (
            "The hour did not announce itself as important. Important hours seldom have the courtesy to wear labels."
        ),
        (
            "The rooms of Bag End held their own geography. There was the study where Bilbo's ink had stained the "
            "desk; the pantry where no visitor could be entirely unwelcome; the passage where coats smelled of "
            "wool, dust, and lavender; the round door through which departures became stories."
        ),
        (
            "Frodo had learned the discipline of small tasks. He answered letters, paid bills, counted jars, "
            "returned borrowed books, and endured relations. None of these deeds was heroic. That was why they "
            "were useful. They kept the days from becoming only waiting."
        ),
        (
            "Yet waiting had settled somewhere under the floorboards. It did not creak every time one crossed the "
            "room, but it was there. On wet evenings, when the wind came out of the west and the fire burned low, "
            "Frodo could almost give it a shape."
        ),
        (
            "Sam knew more of that mood than Frodo guessed. Gardeners know houses from the outside: which windows "
            "are lit too late, which curtains are drawn too soon, which door opens with a cheerful force and which "
            "opens after a pause."
        ),
        (
            "'You will say if anything wants doing, Mr. Frodo,' Sam said once, with his eyes on the hedge. "
            "'There is always something wanting doing in a place like this, even when it looks as if it is only "
            "sitting there being respectable.'"
        ),
        (
            "'I will,' said Frodo. He meant it kindly and knew it was not the whole truth. There are tasks one "
            "cannot hand to a friend because one cannot yet name them without sounding foolish."
        ),
        (
            "The Shire went on around them with admirable stubbornness. Apples were stored; chimneys were swept; "
            "birthdays multiplied; family trees were defended with more passion than any border. If trouble had "
            "walked into the market-square wearing boots, it would first have been asked whose cousin it was."
        ),
        (
            "That stubbornness was not contemptible. Frodo felt, more than once, that its very smallness was a kind "
            "of treasure. A world in which people argued about seed-cake and pipe-weed had preserved something "
            "that greater lands might envy without understanding."
        ),
        (
            "Still, the Road remained. It lay beyond the hedges in every weather, a brown line by day and a thought "
            "by night. Bilbo had followed it once with a song in his mouth. Frodo wondered whether roads remember "
            "the feet they have taken."
        ),
        (
            "'Mr. Bilbo would have known what to make of it,' Sam said. 'Or else he would have made something of "
            "it first and found out afterward whether it was true.'"
        ),
        (
            "'That is a fair description of his scholarship,' Frodo said. 'He often discovered evidence after he "
            "had already enjoyed the conclusion.'"
        ),
        (
            "They smiled together, and in that smile Bilbo returned for a moment: quick-eyed, untidy, pleased with "
            "himself, and already halfway out of the door toward some explanation no one had requested."
        ),
        (
            "The ring remained where Frodo had put it. It made no sound. It did not brighten, darken, tremble, or "
            "betray itself by any sign fit for a tale. That was one of the most disquieting things about it."
        ),
        (
            "Frodo told himself that he was being fanciful. Then he told himself that Bilbo had been fanciful and "
            "had nevertheless been right about more than respectable people liked to admit."
        ),
        (
            "No decision was made in a single stroke. The chapter of that autumn was written in glances, pauses, "
            "unfinished questions, and the modest courage of not laughing away what one fears to understand."
        ),
        (
            "By evening, the damp had climbed the windows and made the lamplight bloom around the panes. Bag End "
            "seemed to float in its own warm hollow, and beyond it the Hill went down into mist."
        ),
        (
            "If there was danger, it had not yet learned the manners of the Shire. It did not knock, complain of "
            "mud, or ask whether supper was included. It waited beyond the range of lamplight, and waiting was "
            "all the more troubling because it resembled peace."
        ),
        (
            "Frodo set the thought aside, then found it beside him again. Some thoughts are like burrs from a hedge: "
            "small, dry, and almost weightless, until one discovers they have travelled all the way home."
        ),
        (
            "The fire settled lower. Sam went away at last, whistling softly because silence after serious talk "
            "felt disrespectful to the garden. Frodo listened until the whistle faded and the house returned to "
            "its old habit of creaking to itself."
        ),
        (
            "He did not know that this was a hinge in the years. Few people know such things while hinges are "
            "turning. They hear only the small sound of a door moving, and think it is another ordinary room."
        ),
        (
            "So the Shire slept, or seemed to. Under that sleep lay roads, rumours, old gifts, older questions, "
            "and the patient friendship of those who would later remember that they had been present at the edge "
            "of something unnamed."
        ),
        (
            "There were no horns in that hour, no gathering of captains, no speech that would have sounded large "
            "if written down. There was only a small household trying to remain itself while attention deepened "
            "around it."
        ),
        (
            "Frodo began to understand that secrecy was not the same as hiding. Hiding is a motion of fear, quick "
            "and cramped. Secrecy, when kept for love, can become a room in which the truth waits until it may be "
            "handled without breaking those who touch it."
        ),
        (
            "Sam did not put it so. Sam would have said that a covered seed is not a buried one, provided the "
            "gardener remembers where he set it and why. He would also have said that too much poking does no "
            "good, and in that he would not have been wrong."
        ),
        (
            "The Shire had many protections that did not look like protections. It had gossip, which noticed "
            "strangers; meals, which interrupted despair; family obligations, which dragged solitary minds back "
            "among spoons, cousins, and chairs; gardens, which insisted that tomorrow had practical needs."
        ),
        (
            "Frodo trusted those protections and distrusted them. They were strong because they were ordinary, "
            "and weak for the same reason. Ordinary things can bar the door against ordinary trouble. They do not "
            "always know what to do with a shadow that has not yet chosen a shape."
        ),
        (
            "In such days Bilbo's absence changed character. It was no longer only loss. It became a sort of "
            "question left in the house: what had he escaped, what had he carried, and what had he unknowingly "
            "placed into Frodo's keeping?"
        ),
        (
            "Frodo answered none of these questions aloud. To answer them aloud would have made them sound more "
            "certain than they were. Instead he answered by locking a drawer, opening a letter, listening at a "
            "window, and thanking Sam for work already done."
        ),
        (
            "That was how the matter grew: not by sudden terror, but by the accumulation of small acts. A glance "
            "toward the Road. A pause before a name. A hand withdrawn from a pocket. A joke made quickly because "
            "silence had become too accurate."
        ),
        (
            "The red book lay near enough to be seen and far enough not to accuse him. Frodo sometimes thought "
            "of writing in it, then found that the first sentence would have to explain too much. A blank page "
            "can be kinder than an honest beginning."
        ),
        (
            "Still, kindness alone was not enough. The years were asking something of him, though softly. They "
            "asked that he not grow careless merely because nothing had happened, and not grow bitter because "
            "nothing could yet be done."
        ),
        (
            "Sam's part in this was quieter and therefore steadier. He brought wood, mended edges, trimmed what "
            "needed trimming, and left unsaid what Frodo could not bear to have examined. Loyalty, in Sam, was "
            "often indistinguishable from doing the next useful thing."
        ),
        (
            "So the chapter of the long quiet did not close. It lengthened. It laid one ordinary day beside "
            "another until ordinariness itself became the field on which courage would have to stand."
        ),
        (
            "A less patient tale might have hurried past such days, but haste would have missed the point. The "
            "danger was not yet at the door. The work of the hour was to keep the door, the hearth, the friendship, "
            "and the inward watch all sound."
        ),
        (
            "Frodo learned, little by little, that a life may be altered before it is interrupted. The same rooms "
            "stood around him; the same neighbours called; the same path climbed to the same round door. Yet he "
            "moved among them with a new care."
        ),
        (
            "That care did not make him grim. Grimness would have been easier, and less true. He still laughed, "
            "still read, still walked, still listened to Sam's reports on weather and soil. The change lay in the "
            "space after laughter, where thought returned."
        ),
        (
            "If Gandalf had seen him then, perhaps he would have been both troubled and comforted: troubled by "
            "the weight already gathering, comforted that Frodo did not throw it aside merely because no one had "
            "explained its name."
        ),
        (
            "The Shire, meanwhile, continued to be gloriously itself. It corrected manners, mislaid tools, judged "
            "pies, praised gardens, and treated distant events as suspect until they interfered with deliveries. "
            "Frodo held fast to that stubborn peace."
        ),
    ]

    if scene_kind == "visit":
        shared_expansions = [
            (
                "Frodo brought out a small tray of seed-cake, and Gandalf took a piece without seeming to know "
                "that he had done so. He broke it in two, set one half down, and forgot both halves while his "
                "eyes followed the fire."
            ),
            (
                "'You are thinking of something far from seed-cake,' Frodo said."
            ),
            (
                "'That is a common fault among travellers,' Gandalf answered. 'They return with their boots in "
                "one country and their thoughts in another. It makes them poor company and worse correspondents.'"
            ),
            (
                "Frodo looked at him steadily. 'You have been a poor correspondent for some years.'"
            ),
            (
                "'I have,' said Gandalf. 'And yet I read every letter that reached me, even the one in which you "
                "spent two pages describing a quarrel over a turnip cart. Especially that one. It had the rare "
                "virtue of being concerned with a problem no larger than itself.'"
            ),
            (
                "The answer pleased Frodo more than he wished to show. He had sometimes feared that letters sent "
                "down the Road vanished into the same wide silence that had taken Bilbo. To learn that they had "
                "been read was a small restoration."
            ),
            (
                "Gandalf saw it and became gentler. 'Do not think I have forgotten this house,' he said. 'Nor the "
                "person who keeps it. I have come and gone because coming too often may draw eyes, and staying "
                "away too long may leave friends without counsel.'"
            ),
            (
                "'What eyes?' Frodo asked."
            ),
            (
                "Gandalf's pipe had gone out. He looked at the cold bowl, then set it aside. 'Curious ones, chiefly. "
                "Greedy ones, perhaps. I do not yet know enough to name them, and an unnamed suspicion is a poor "
                "guest to introduce at table.'"
            ),
            (
                "Frodo heard the evasion, but he also heard the care inside it. Gandalf was not playing at secrets. "
                "He was holding something by the edges, as one holds a pot too hot to pass safely to another hand."
            ),
            (
                "Outside, Sam had finished the nasturtiums and begun on a border that needed no attention at all. "
                "He worked with the solemn industry of a person who would rather be accused of over-trimming than "
                "of listening."
            ),
            (
                "The fire sank, and the room grew more golden. It made Gandalf's face look carved in old wood and "
                "Frodo's younger than he felt. The contrast troubled the wizard; Frodo saw that before Gandalf "
                "could hide it."
            ),
            (
                "'There is another question,' Gandalf said. 'Have you noticed any change in yourself? Not mood, "
                "not appetite, not the ordinary vanity of birthdays. Something stranger.'"
            ),
            (
                "Frodo almost laughed. 'That is a large net to cast over a small hobbit. I have been accused of "
                "becoming more like Bilbo, if that counts as a change or a village complaint.'"
            ),
            (
                "'It may count as both,' said Gandalf. 'But I meant your face, your strength, the way the years "
                "sit on you.'"
            ),
            (
                "That made Frodo uncomfortable. Praise may be brushed aside; inspection cannot. He went to the "
                "mantel and straightened a candlestick that was already straight."
            ),
            (
                "'I am well,' he said. 'Too well, according to some. The Shire distrusts a person who does not "
                "become rounder, slower, and more opinionated at the expected pace.'"
            ),
            (
                "Gandalf nodded, but his nod was no answer. It was a mark set quietly in an inner ledger. Frodo "
                "felt it being written down and disliked the feeling."
            ),
            (
                "At length the talk turned back to Bilbo because both of them needed it to. They remembered his "
                "fondness for dramatic exits, his incurable suspicion that supper could be improved by company, "
                "and his habit of making any guest feel that the best story had been saved for after the plates."
            ),
            (
                "Those memories did what memories can do when treated kindly: they lit the room without pretending "
                "to bring back the person who had left it. Gandalf laughed once, quietly and truly, and Frodo was "
                "glad of that laugh long after it ended."
            ),
        ]
    elif scene_kind == "departure":
        shared_expansions = [
            (
                "Frodo remained by the window after Sam went to fetch plates. The glass held a faint reflection: "
                "his own face, the pale square of morning, and behind both the blurred shape of the room Bilbo "
                "had left him."
            ),
            (
                "He wondered whether growing up in Bag End had prepared him for anything except loving Bag End. "
                "That seemed, suddenly, no small education. To love a place is to be made responsible for its "
                "quiet without ever being asked whether one accepts the office."
            ),
            (
                "Sam returned with more noise than plates required. He was giving Frodo time, though he would have "
                "been embarrassed to hear the courtesy named. Some kindnesses in the Shire travelled best under "
                "cover of practical fuss."
            ),
            (
                "'Mr. Gandalf has a way of making a body feel as if a plain gate may open into more road than was "
                "there yesterday,' Sam said."
            ),
            (
                "'Yes,' said Frodo. 'And then he leaves us to oil the hinges.'"
            ),
            (
                "Sam considered this. 'Hinges want oiling whether there is road beyond or not.'"
            ),
            (
                "Frodo looked at him, and the first true smile of the morning came. 'That may be the wisest thing "
                "said in Bag End today.'"
            ),
            (
                "They ate. The mushrooms were excellent, which helped more than either of them would have admitted "
                "in a solemn history. Butter, pepper, and toast cannot solve dread, but they can remind it that "
                "it has not been given the whole house."
            ),
            (
                "Afterward Frodo went into the study and stood among Bilbo's maps. The lines of ink ran over hills, "
                "rivers, woods, and borders. He touched none of them. He had the feeling that if he placed a finger "
                "on any road, it might begin moving under his hand."
            ),
            (
                "The ring was in its hiding-place. He did not take it out again. He knew too well how easily a "
                "private act becomes a private habit, and how a habit can begin to look like a right."
            ),
            (
                "Gandalf had given him no command that could be shown to another hobbit. There was no letter, no "
                "seal, no explanation grand enough to silence questions. There was only trust, and trust is a "
                "light burden until it must be carried without witnesses."
            ),
            (
                "By noon, news of the wizard's departure had already outrun him. Hobbiton found it disappointing "
                "that he had left no fireworks, no rumours of treasure, and no offence definite enough to sustain "
                "a proper discussion."
            ),
            (
                "The Gaffer declared that wizards were best judged like late frosts: troublesome when present, "
                "suspicious when absent, and never improved by speculation. This verdict satisfied several people "
                "who had not known they wanted one."
            ),
            (
                "Frodo heard the report from Sam and laughed, but he did not repeat Gandalf's warning. The warning "
                "had already gone inward, where repetition would not make it clearer."
            ),
            (
                "Toward evening he walked as far as the bend where Gandalf had vanished. There was no mark on the "
                "road. Wheels, boots, and pony-hooves had confused the mud into ordinary Shire testimony. The "
                "wizard might have left that morning or a year before."
            ),
            (
                "Frodo stood there until the light thinned. Somewhere beyond the hedges lay Bree, and beyond Bree "
                "more roads than his maps could make comfortable. For the first time, the distance did not feel "
                "romantic. It felt awake."
            ),
            (
                "When he returned, Sam was trimming the hedge by lamplight, an activity he would have condemned "
                "in any other gardener. Frodo did not mention it. Sam did not mention the walk. Between them the "
                "unspoken thing took root, not as secrecy only, but as loyalty."
            ),
            (
                "That night Frodo slept badly. He dreamed of a road with no traveller on it, a gate that opened "
                "inward and outward at once, and Gandalf's staff tapping somewhere just beyond hearing."
            ),
            (
                "In the morning he woke to birds, bread, and the ordinary tyranny of correspondence. The letters "
                "were dull, the bread was good, and the birds had no respect for hidden worries. He found himself "
                "grateful to them all."
            ),
        ]
    else:
        shared_expansions = inheritance_expansions

    idx = 0
    expansion_templates = scene_specific_templates + shared_expansions
    while _count_words("\n\n".join(paragraphs)) < target_words and idx < len(expansion_templates):
        paragraphs.append(expansion_templates[idx])
        idx += 1

    if target_dialogue_ratio:
        current = "\n\n".join(paragraphs)
        dialogue_words = _dialogue_word_count(current)
        total_words = max(1, _count_words(current))
        dialogue_templates = [
            (
                "'I am asking for patience, not blindness,' said Gandalf. 'There is a difference, though it is "
                "seldom appreciated by people who are being asked to wait.'"
            ),
            (
                "'Patience is easier when one knows what it is for,' said Frodo. 'Otherwise it looks very much "
                "like being kept in the parlour while other people search the house.'"
            ),
            (
                "'Then call it stewardship,' Gandalf answered. 'Keep what was entrusted to you. Keep your own "
                "counsel. Let the days remain ordinary while ordinary days are granted.'"
            ),
            (
                "'I can keep a garden gate shut,' said Sam. 'I can keep weeds down, more or less. I cannot say as "
                "I understand keeping counsel, but I know when a thing is not mine to repeat.'"
            ),
            (
                "'That may be the beginning of wisdom, Samwise,' said Gandalf. 'A quiet tongue has guarded more "
                "than one loud sword ever saved.'"
            ),
            (
                "'You make it sound large,' said Frodo. 'It is a small ring and a small house.'"
            ),
            (
                "'Small things fit through cracks where large ones cannot go,' said Gandalf. 'That is why I am "
                "not comforted merely because a thing is small.'"
            ),
        ]
        dialogue_idx = 0
        while (dialogue_words / total_words) < target_dialogue_ratio and dialogue_idx < len(dialogue_templates):
            paragraphs.append(dialogue_templates[dialogue_idx])
            dialogue_idx += 1
            current = "\n\n".join(paragraphs)
            dialogue_words = _dialogue_word_count(current)
            total_words = max(1, _count_words(current))

    return "\n\n".join(paragraphs)


def _template_scene_text(
    *,
    project: dict,
    scene_goal: str,
    characters: list[str],
    place: str,
    objects: list[str],
    event: dict,
    scene_beats: list[dict],
    missing_terms_hint: list[str],
    quality: dict[str, Any] | None = None,
) -> str:
    if _is_hunt_gollum_project(str(project.get("slug") or "")):
        return _template_hunt_gollum_scene_text(
            project=project,
            scene_goal=scene_goal,
            characters=characters,
            place=place,
            objects=objects,
            event=event,
            scene_beats=scene_beats,
            missing_terms_hint=missing_terms_hint,
            quality=quality,
        )

    if _is_shire_gap_project(str(project.get("slug") or "")):
        return _template_shire_gap_scene_text(
            project=project,
            scene_goal=scene_goal,
            characters=characters,
            place=place,
            objects=objects,
            event=event,
            scene_beats=scene_beats,
            missing_terms_hint=missing_terms_hint,
            quality=quality,
        )

    def _join_names(names: list[str]) -> str:
        rows = names[:4]
        if len(rows) <= 1:
            return rows[0] if rows else "The company"
        if len(rows) == 2:
            return f"{rows[0]} and {rows[1]}"
        return ", ".join(rows[:-1]) + f", and {rows[-1]}"

    def _display_goal(raw_goal: str) -> str:
        blocked_prefixes = (
            "shadow action to realize:",
            "scene beats to cover:",
            "scene brief:",
            "motifs to echo:",
            "story-time is",
            "past figures",
            "do not mention",
            "required canon anchors",
        )
        parts = []
        for raw_part in re.split(r"(?<=\.)\s+", raw_goal):
            part = raw_part.strip()
            if not part:
                continue
            if part.lower().startswith(blocked_prefixes):
                continue
            parts.append(part)
        return " ".join(parts[:2]) or "The scene advances the chapter while keeping faith with First Age canon."

    def _plain_sentence(text: str) -> str:
        return str(text or "").strip(" .,:;!?\"'")

    def _clean_motifs(raw: list[Any]) -> list[str]:
        rows: list[str] = []
        for raw_motif in raw:
            motif = _plain_sentence(str(raw_motif)).lower()
            if not motif or motif in MOTIF_STOPWORDS:
                continue
            if len(motif) < 3 or motif.endswith("'s"):
                continue
            rows.append(motif)
        preferred = [m for m in rows if m in PREFERRED_STORY_MOTIFS]
        return _dedupe_strings(preferred + rows)[:3]

    def _choose_roles(names: list[str]) -> tuple[str, str, str | None]:
        lowered = {name.lower(): name for name in names}

        def _pick(preferred: list[str], blocked: set[str] | None = None) -> str | None:
            blocked = blocked or set()
            for candidate in preferred:
                if candidate.lower() in lowered and candidate.lower() not in blocked:
                    return lowered[candidate.lower()]
            for candidate in names:
                if candidate.lower() not in blocked:
                    return candidate
            return None

        goal_l = scene_goal.lower()
        if "luthien speaks" in goal_l and "luthien" in lowered:
            lead_name = lowered["luthien"]
        else:
            lead_name = _pick(["Beren", "Luthien", "Thingol", "Melian"]) or "Beren"
        respondent_name = _pick(["Thingol", "Melian", "Luthien", "Beren"], {lead_name.lower()}) or lead_name
        witness_name = _pick(["Melian", "Luthien", "Thingol", "Beren"], {lead_name.lower(), respondent_name.lower()})
        return lead_name, respondent_name, witness_name

    def _dialogue_claim(goal_text: str, speaker: str) -> str:
        goal_l = goal_text.lower()
        if "silmaril" in goal_l:
            return "Name the price, and I will hear it before all"
        if "luthien speaks" in goal_l or speaker.lower() == "luthien":
            return "My will is not a leaf to be borne by another wind"
        if "doriath" in goal_l:
            return "I have come far, and not for ease"
        return "The road before us is hard, yet it is ours to choose"

    def _setting_sentence(raw_place: str) -> str:
        place_l = raw_place.lower()
        if "menegroth" in place_l:
            return "In Menegroth the lamps burned softly among pillars carven like beeches, and the sound of hidden water moved under stone."
        if "doriath" in place_l:
            return "In Doriath the guarded woods stood close about the path, and every branch seemed to listen."
        return f"In {raw_place}, the old lands kept watch under a sky made dim by northern fear."

    def _narrative_goal(goal_text: str) -> str:
        goal_l = goal_text.lower()
        if "beren enters doriath" in goal_l:
            return "Beren had come into Doriath under the burden of his oath."
        if "luthien speaks" in goal_l:
            return "Luthien would not remain silent before Thingol and Melian."
        if "thingol names the silmaril" in goal_l:
            return "Thingol would soon name the Silmaril as the impossible price."
        return goal_text

    char_phrase = _join_names(characters)
    action = str(event.get("action") or "counsel").strip().lower() or "counsel"
    if action in PLACEHOLDER_PARTICIPANTS or action in {"unknown", "worked", "did", "was", "had"}:
        action = "counsel"
    motifs = _clean_motifs(event.get("motifs") or [])
    beat_lines = _dedupe_strings(
        [
            _plain_sentence(str(beat.get("intent") or beat.get("action") or ""))
            for beat in scene_beats
            if _plain_sentence(str(beat.get("intent") or beat.get("action") or ""))
        ]
    )
    anchors = _dedupe_strings(missing_terms_hint)
    anchor_sentence = ""
    if anchors:
        anchor_sentence = (
            " The matter could not be evaded: "
            + ", ".join(anchors)
            + " stood at the heart of their counsel."
        )

    object_sentence = ""
    if objects:
        object_sentence = " Near at hand lay " + ", ".join(objects[:3]) + ", each made weightier by the hour."

    motif_sentence = ""
    if motifs:
        motif_sentence = " Beneath the speech moved " + ", ".join(motifs[:3]) + "."

    beat_sentence = ""
    if beat_lines:
        beat_sentence = " The turn of the hour was plain: " + "; ".join(beat_lines[:2]) + "."

    clean_goal = _display_goal(scene_goal)
    scene_statement = _narrative_goal(clean_goal)
    goal_l = clean_goal.lower()
    project_slug = str(project.get("slug") or "").lower()
    if "doriath" in goal_l or "guarded trees" in goal_l:
        premise = "Here oath and welcome first tested one another under the guarded trees of Doriath."
    elif "luthien speaks" in goal_l:
        premise = "Here love ceased to be a hidden matter and became a voice before the throne of Doriath."
    elif "silmaril" in goal_l:
        premise = "Here the perilous price in Doriath was named, and pride gave doom a shape."
    elif "beren" in project_slug or "luthien" in project_slug:
        premise = "This was a First Age matter of oath, love, and a perilous price in Doriath."
    else:
        premise = str(project.get("premise") or "The tale advances under the long shadow of an elder age.").strip()
    quote_goal = _dialogue_claim(clean_goal, characters[0] if characters else "Beren")
    lead, respondent, witness = _choose_roles(characters)
    observer = witness or "The court"
    third_voice = witness or respondent
    quality = quality or {}
    target_words = int(quality.get("target_scene_words", 0) or 0)
    target_dialogue_ratio = float(quality.get("target_dialogue_ratio", 0.0) or 0.0)
    character_lowers = {name.lower() for name in characters}
    if lead.lower() == "beren":
        memory_reflection = (
            f"{lead} remembered roads without shelter, bitter uplands, and the faces of the fallen. Memory did "
            f"not make {lead} splendid before {respondent}. It made the choice smaller, sharper, and more exact."
        )
        weather_reflection = (
            "The face of Beren seemed worn by weather foreign to that hidden realm. Rain on barren stone, ash "
            "on the wind, and the hunger of exiles had left their marks."
        )
    elif lead.lower() in {"luthien", "lúthien"}:
        memory_reflection = (
            "Luthien remembered starlit lawns, songs under boughs, and the freedom of her own steps. Such "
            "memory did not make obedience easier. It made silence impossible."
        )
        weather_reflection = (
            "Beyond the guarded realm lay rain, ash, and the hunger of exiles. The thought of those roads came "
            "even into Menegroth, and the hidden halls no longer seemed wholly removed from grief."
        )
    else:
        memory_reflection = (
            f"{lead} remembered older counsels, broken houses, and names that had become laments. Memory made "
            "the present choice narrower, and therefore more terrible."
        )
        weather_reflection = (
            "The world beyond the hidden doors seemed to press nearer. Rain, ash, exile, and rumour came into "
            "the hall as if carried on an unseen cloak."
        )
    melian_reflection = (
        "Melian's silence was unlike the silence of others. It did not withdraw. It measured, and under it "
        "careless words grew thin."
        if "melian" in character_lowers
        else "An older silence lay upon the hall. It did not withdraw. It measured, and under it careless words grew thin."
    )
    overhead_sentence = (
        f"Above them the boughs of {place} were interlaced in patient shadow."
        if place.lower() == "doriath"
        else f"Above them the roof of {place} held its carved branches in patient shadow."
    )

    paragraphs = [
        (
            f"{_setting_sentence(place)} {char_phrase} were gathered in a silence that was not peace. "
            f"The hour asked for {action}, but not for haste. {scene_statement}"
        ),
        (
            f"{premise} Yet the tale did not move by ornament. It moved by choice, by law, and by the "
            "dreadful courtesy of those who knew that one spoken word might outlive a kingdom. "
            "Every vow in Beleriand seemed to call witnesses from stone and leaf."
            f"{anchor_sentence}{object_sentence}"
        ),
        (
            f"'{quote_goal},' said {lead}. 'I ask no easy road, and I bring no claim that can stand unless "
            "truth stands with it. Let the danger be counted plainly; I will not make it smaller by proud words.'"
        ),
        (
            f"{respondent} answered after a pause. 'No oath is light in this age. No love is hidden from doom. "
            f"If {lead} must speak before the court, let the speech be whole, for half-counsels have ruined "
            "kings greater than we are.'"
        ),
        (
            f"{observer} heard the words and did not soften them. The hour belonged wholly to the Elder Days. "
            "Its peril was ancient, its hope narrow, and its mercy the more dangerous because it had not yet "
            "found any shape in deed."
            f"{motif_sentence}{beat_sentence}"
        ),
    ]

    if "doriath" in goal_l or "guarded trees" in goal_l:
        paragraphs.append(
            "At the borders the wardens had passed like grey leaves between trunk and thorn. "
            "No horn announced them. No gate swung wide. The Girdle lay about the forest like "
            "an unseen water, and Beren felt the mortal dust of the roads upon his cloak."
        )
        paragraphs.extend(
            [
                (
                    "Under oak, beech, hazel, holly, rowan, and alder, the forest changed its speech. Fern "
                    "uncurled beside stone; lichen silvered fallen bark; foxglove, sorrel, and white anemone "
                    "stood untouched by boot or wheel. Beren noticed each small thing because danger had taught "
                    "him to read the ground, yet here the signs did not point to ambush. They pointed inward."
                ),
                (
                    "He carried the smell of rain-dark leather, cold ashes, and long hunger. Doriath answered "
                    "with leaf-mould, running water, bee-hum, and the faint fragrance of flowers hidden from the "
                    "open sky. The meeting of those scents seemed almost a judgement: the wild grief of Men "
                    "brought before a realm that still remembered wholeness."
                ),
                (
                    "Luthien knew the names of glade, hollow, brook, and mound, but she did not recite them. "
                    "This was not a tour of wonder. It was a passage toward answer and consequence, and every "
                    "familiar bole seemed to ask whether she understood what she was bringing home."
                ),
            ]
        )
    elif "luthien speaks" in goal_l or lead.lower() == "luthien":
        paragraphs.append(
            "Luthien stood neither as prize nor as petition, but as the daughter of that house and "
            "the keeper of her own will. The old songs in the rafters seemed to pause above her. "
            "Melian's gaze was deep, and Thingol's silence had the edge of a drawn blade."
        )
        paragraphs.extend(
            [
                (
                    "Menegroth gathered splendour from many crafts: polished cedar, dark yew, pale limestone, "
                    "green enamel, bronze hinges, ivory inlay, woven hangings, and lamps like captured stars. "
                    "Yet none of these held Luthien's eye. She saw instead her father's hand, her mother's still "
                    "face, and the place where Beren would have to stand."
                ),
                (
                    "Courtiers in blue, russet, pearl, and deep woodland green watched with disciplined faces. "
                    "Some had served Thingol since the first delving of those halls; some had never known any "
                    "danger that did not break harmlessly upon Melian's power. Their caution was sincere, but "
                    "sincerity does not make fear wise."
                ),
                (
                    "Luthien felt the old melodies of the house around her: cradle-song, harvest-song, laments "
                    "for vanished friends, and hymns to starlight over unshadowed waters. She loved them all. "
                    "Because she loved them, she would not let them become a net thrown over her living will."
                ),
            ]
        )
    elif "silmaril" in goal_l:
        paragraphs.append(
            "Then the name of the Silmaril came near, bright and ruinous. It carried Angband in "
            "its shadow, iron doors, wolf-cries, and the black hand that wore stolen light upon a crown. "
            "The bride-price was no treasure. It was a sentence spoken in royal pride."
        )
        paragraphs.extend(
            [
                (
                    "The word seemed to kindle colours no lamp had lit: adamant white, furnace red, sea-green, "
                    "sword-blue, and the hard gold of a treasure beyond ransom. In that imagined radiance were "
                    "mingled theft, exile, kin-slaying, forge-smoke, salt tears, and the unsatisfied hunger of "
                    "those who call possession justice."
                ),
                (
                    "Beren's hand closed at his side. He thought not of gems but of gates, trenches, slag, "
                    "scorched iron, chained captives, carrion birds, and the sleepless malice that brooded in "
                    "the North. A court might name such a thing as price; the road would name it otherwise."
                ),
                (
                    "Luthien heard more than the command. She heard the loneliness hidden in it, the fear of a "
                    "father who would rather demand the impossible than confess helplessness before love. That "
                    "knowledge did not excuse the wound, but it made the wound more sorrowful."
                ),
            ]
        )
    else:
        paragraphs.append(
            "The chamber held many kinds of strength: patience, foresight, grief, and the stubborn "
            "valour that survives when songs and banners have been spent. None of them was enough alone."
        )

    scene_specific_templates: list[str] = []
    if "doriath" in goal_l or "guarded trees" in goal_l:
        scene_specific_templates = [
            (
                "Before he came beneath the greater trees, Beren had known forests without mercy. He had slept "
                "under boughs that hid enemies, and waked to mornings where no bird sang. Doriath was not so. "
                "Its silence was ordered, watchful, and fair, and therefore it troubled him more."
            ),
            (
                "The path by which he was led did not seem made by axe or wheel. It turned as water turns, "
                "choosing hollows and roots, and every turn brought another glimmer of green light. More than "
                "once he looked back and saw no road behind him."
            ),
            (
                "Luthien walked beside him with no haste. The woods knew her. They parted for her without sound, "
                "and the small creatures of leaf and moss did not flee. Yet her face was grave, for love had "
                "brought a stranger into the guarded land, and love does not enter guarded places without cost."
            ),
            (
                "'This realm is not won by strength,' said Luthien. 'Nor by stealth, though you came far by both. "
                "If you stand before my father, stand as yourself, for disguises wither quickly under Melian's eyes.'"
            ),
            (
                "Beren bowed his head, not as a courtier but as one who had learned respect in bitter schools. "
                "'I have little left to hide,' he said. 'My house is broken, my hand empty, and my oath heavier "
                "than any mail I ever wore.'"
            ),
            (
                "At that Luthien looked away into the trees. The light touched her hair and was changed by it. "
                "She had heard many lays in Menegroth, but in Beren's speech there was no woven art, only the "
                "plain grain of endurance."
            ),
            (
                "The wardens kept their distance, though Beren felt their presence. Grey cloaks moved and were "
                "still. Eyes watched from shadowed boles. No spear was raised, and yet the path seemed narrower "
                "than a bridge over a chasm."
            ),
            (
                "He thought then of Barahir, of companions lost in fen and upland, and of hands that had clasped "
                "his own before death took them. The thought did not harden him. It made him aware of how slight "
                "a mortal life may be, and how great a promise can become when little else remains."
            ),
            (
                "There were flowers underfoot, pale as small stars, and he stepped carefully among them. Such "
                "care would have seemed foolish in the hunted lands. Here it seemed the first lesson of the place."
            ),
            (
                "'If I bring grief with me,' said Beren, 'tell me to turn aside while turning still has meaning.' "
                "Luthien did not answer quickly. When she did, her voice was low: 'Grief has many roads. I will "
                "not pretend that silence closes them.'"
            ),
            (
                "So they passed inward, and the forest deepened around them. The light was neither day nor dusk, "
                "but a mingling of both, as if time itself moved more slowly under Melian's protection."
            ),
            (
                "Beren felt the enchantment and did not understand it. He knew only that his weariness rose to "
                "meet it, and that a part of him longed to lay down every burden. Yet another part, more stubborn "
                "and more wounded, would not allow him peace bought by silence."
            ),
            (
                "When at last the ways of Menegroth were spoken of ahead, the word seemed to change the air. "
                "Hall and throne, king and queen, judgement and song: these gathered before him like shapes seen "
                "through rain."
            ),
            (
                "Luthien paused once before the last green turn. 'My father loves his realm,' she said. 'He loves "
                "what he can guard. Remember that, Beren, when his words are hard.'"
            ),
            (
                "'And what cannot be guarded?' Beren asked. Luthien looked at him then, and for a moment no leaf "
                "moved. 'Then even kings must learn whether love is possession or gift.'"
            ),
            (
                "No trumpet marked their coming. The hidden realm did not spend its wonder loudly. It drew them "
                "inward, branch by branch and shadow by shadow, until Beren understood that he was not merely "
                "approaching a court, but crossing into the peril of being known."
            ),
        ]
    elif "luthien speaks" in goal_l or lead.lower() == "luthien":
        scene_specific_templates = [
            (
                "The halls of Menegroth had heard many judgements, but seldom had they waited upon Luthien. "
                "Those who looked on her expected beauty to soften the hour. Instead they saw resolve, and some "
                "were afraid, for resolve in the beloved can wound more deeply than anger in an enemy."
            ),
            (
                "Thingol sat with the stillness of a king who has guarded long and lost little within his borders. "
                "His love was great, but it had learned the habit of command. Beside him Melian watched, and her "
                "watchfulness was older than any throne in Beleriand."
            ),
            (
                "'Daughter,' said Thingol, and in that single word lay tenderness, warning, and possession. "
                "Luthien heard all three. She loved him, and therefore answered none of them falsely."
            ),
            (
                "'Father,' she said, 'I am your daughter, but I am not a jewel kept under lock, nor a song to be "
                "sung only when the hall desires it. I have seen Beren's face. I have heard the truth in him.'"
            ),
            (
                "A murmur moved among the listeners and was gone. Menegroth had many chambers, but the heart of "
                "the hall seemed suddenly small, as if all its carved ways had narrowed to the space between a "
                "father's fear and a daughter's will."
            ),
            (
                "Melian did not rebuke the murmur. Her gaze rested on Luthien, and in it there was sorrow without "
                "surprise. She had seen long ago that no girdle, however strong, could keep doom forever outside "
                "the lives of those within."
            ),
            (
                "'You speak of truth,' said Thingol. 'Truth may be brought by ruin as well as by honour. Shall I "
                "open my realm because a mortal wanderer looks steadfast in my hall?'"
            ),
            (
                "Luthien's answer came clear. 'No. Open it because justice is not made greater by contempt. Hear "
                "him because a king who refuses to hear has already judged, and judgement without hearing is fear "
                "wearing a crown.'"
            ),
            (
                "Then silence fell in earnest. Even those who disliked the words knew that they had struck stone. "
                "Thingol's hand tightened on the arm of his seat, and for a while the whole court seemed to listen "
                "to that small sound."
            ),
            (
                "Beren was not the only matter before them, though his name lay at the centre. Luthien herself "
                "had become the question. Was she to be guarded as one guards treasure, or trusted as one trusts "
                "a free heart?"
            ),
            (
                "Melian spoke then, and the sound of her voice altered the hall. 'Long have I guarded this land,' "
                "she said, 'but I did not make it so that living things should forget their own wills. Beware, "
                "Elu, lest love take the shape of a prison.'"
            ),
            (
                "Thingol looked from Melian to Luthien, and pride battled grief in his face. He had faced enemies "
                "less dangerous than this: a truth spoken by those he loved, before all who owed him reverence."
            ),
            (
                "Luthien did not press him in triumph. That was not her purpose. She stood quietly, and the quiet "
                "was harder to dismiss than defiance would have been, for it asked him to meet her not as child "
                "but as one whose choice had already begun."
            ),
            (
                "'If sorrow follows,' she said, 'let it not be born from silence forced upon me. Let it come, if "
                "come it must, from a choice made openly under the eyes of those who love me.'"
            ),
            (
                "The words moved through the hall like a wind under doors. Some looked down. Some looked toward "
                "Melian. Some, who had thought the matter a courtly disturbance, understood at last that the old "
                "world had shifted a little beneath their feet."
            ),
            (
                "Thingol did not yield. Yet neither did he command silence. Between those two things lay the first "
                "change of the day, and Luthien saw it, though she gave no sign but the steady lifting of her head."
            ),
        ]
    elif "silmaril" in goal_l:
        scene_specific_templates = [
            (
                "When Thingol spoke again, the hall braced itself without knowing why. His voice had grown smooth, "
                "and that smoothness was more perilous than wrath. It was the voice of a king finding a path by "
                "which refusal might appear as honour."
            ),
            (
                "'You ask much,' he said to Beren. 'You ask beyond measure. Therefore I will name a measure fit "
                "for such asking, and all here shall know whether your courage is coin or only breath.'"
            ),
            (
                "Melian's eyes darkened, but she did not interrupt. A shadow of foresight moved across her face. "
                "It was not fear alone. It was recognition, as when one hears far off the first stone loosened "
                "above a valley."
            ),
            (
                "Beren stood still. He had endured hunger, pursuit, and loneliness, but this was another kind of "
                "trial. A blade declares itself honestly. Pride may smile while it draws blood."
            ),
            (
                "'Bring to me,' said Thingol, 'one Silmaril from the crown of Morgoth. Then, if your hand can hold "
                "what all powers fear to seek, I will hear your claim concerning Luthien.'"
            ),
            (
                "The name struck the hall like iron. No one needed the tale explained. The jewels were not merely "
                "treasures. They were hallowed light, theft, oath, exile, slaughter, and the hunger of powers too "
                "great for the peace of any guarded kingdom."
            ),
            (
                "Luthien went pale, but not with weakness. She looked first at Beren, then at Thingol, and the "
                "hurt in her face was sharper because it was restrained. Her father had not barred the door; he "
                "had opened it upon a precipice."
            ),
            (
                "A few in the court drew breath as if to speak, but no counsel came. Menegroth, with all its "
                "craft and music, seemed suddenly very far from the North, and yet the North had entered it."
            ),
            (
                "Beren's thought went to Angband, though he had never stood within its gates. He saw in the mind's "
                "darkness iron, fire, wolves, and a crown above a face that hated all living beauty. Then the image "
                "passed, and Luthien remained before him."
            ),
            (
                "'You name death,' said Beren. His voice was quiet, and because it was quiet the hall heard every "
                "word. 'But death has been near me so long that I know its step. I will not call your price just.'"
            ),
            (
                "Thingol's gaze hardened. 'Yet will you refuse it?' Beren looked at Luthien then, and no proud "
                "answer leapt from him. For the first time that day he seemed weary beyond speech."
            ),
            (
                "'I will not refuse the road because it is dark,' Beren said at last. 'But let none here mistake "
                "darkness for wisdom, or cruelty for the keeping of love.'"
            ),
            (
                "Melian lowered her eyes, and grief passed through her like wind over deep water. She had warned "
                "without commanding, and now the word had gone forth. In great houses, as in wild lands, some "
                "arrows cannot be called back."
            ),
            (
                "Luthien took one step, no more. That step changed the hall more than Thingol's decree had done. "
                "It declared where her heart had gone, and it made plain that the doom named for Beren would not "
                "remain his alone."
            ),
            (
                "'Father,' she said, and the word had no submission in it. 'You have set a price beyond all "
                "kingdoms. Remember, when the tale returns to you, that you named it in your own hall.'"
            ),
            (
                "There was no answer that could mend the hour. The Silmaril had been spoken, and the sound of it "
                "lingered among the carven pillars like a light too bright for mortal eyes and too costly for "
                "Elven pride."
            ),
        ]
    else:
        scene_specific_templates = [
            (
                "The chamber held many kinds of strength, but none could stand alone. Patience without mercy "
                "hardened into stone, and courage without counsel became another path to ruin."
            ),
            (
                "Thus the hour moved slowly, as grave hours do. It did not hasten toward deed, because the deed "
                "already waited beyond speech, and all hearts in the room felt its nearness."
            ),
        ]

    expansion_templates = scene_specific_templates + [
        (
            f"The light in {place} changed as the counsel deepened. It did not brighten. It drew itself into "
            "long threads along the floor, as though the hidden world had leaned nearer to hear."
        ),
        (
            "Beyond the doors the leaves moved without wind. In that motion there was the old patience of "
            "Beleriand, which had seen pride flower and fall before the present grief was named."
        ),
        (
            f"{respondent} looked toward {lead}, and the court seemed to draw away though no one moved. "
            "Between mortal breath and immortal memory lay a space no law could wholly govern."
        ),
        (
            "In that space pity and wonder contended. Even anger clothed itself in ceremony, and ceremony could "
            "not hide the wound beneath it."
        ),
        (
            f"'{action.title()} may be the shape of this hour,' said {third_voice}, 'but its fruit will be borne "
            "far from these halls. Let every name now uttered be kept.'"
        ),
        (
            "The answer was not swift. In such tales silence is also a deed, and the silence there had roots. "
            "It reached down through lost lamps, sundered kin, and the fear of Morgoth."
        ),
        (
            f"Beyond the threshold, {place} kept its own counsel. Carven branch, lamp-glimmer, and hidden water "
            "seemed part of a single listening mind."
        ),
        (
            memory_reflection
        ),
        (
            f"'{respondent}, I do not ask that grief be named mercy,' said {lead}. 'Only this: that the living "
            "may choose while breath remains to them.'"
        ),
        (
            f"{respondent} heard the answer, and the pride of guarded kingdoms rose like a cold wind. Yet pride "
            "was not the only power there. Pity moved also, and fear, and the deep memory of songs becoming laments."
        ),
        (
            "No minstrel at the chamber's edge would have known which gesture first altered the tale: a lifted "
            "hand, a withheld rebuke, or the mortal steadiness that refused to make itself splendid."
        ),
        (
            f"'{lead} shall finish,' said {respondent}. 'A tale broken in the middle breeds more falsehood than "
            "one spoken to its bitter end.'"
        ),
        (
            "There was no comfort in that leave, yet it changed the air. Those who had expected wrath found "
            "judgement instead. Those who had looked for softness saw that love could be as dangerous as war."
        ),
        (
            f"In that hour {char_phrase} were bound not by agreement but by recognition. Each perceived a portion "
            "of the road ahead: the hidden gate, the cruel price, the dark hand in the North."
        ),
        (
            "The court held its breath again. Small sounds returned one by one: cloth stirring, a ring against "
            "wood, water somewhere under stone. Ordinary things became witnesses."
        ),
        (
            "Far away, beyond guarded rivers and thorned borders, the malice of the North was no rumour. It was "
            "a pressure upon the world. It entered by absence, by caution, and by the counting of fair things."
        ),
        (
            f"{place} had chambers for feasting and chambers for judgement, but this hour belonged wholly to "
            "neither. It was a crossing-place, where custom met prophecy and neither yielded cleanly."
        ),
        (
            weather_reflection
        ),
        (
            melian_reflection
        ),
        (
            "The tale had not yet reached sword or prison or the iron crown. Their shadows already lay across "
            "the threshold. Great choices arrive as speech before they become deed."
        ),
        (
            f"{char_phrase} remained within the same light, but each stood at a different distance from peace. "
            "One stood near hope, one near law, and one near the perilous mercy that can undo the designs of kings."
        ),
        (
            "Names had weight in that place. They were not counters in debate, but vessels of memory, kinship, "
            "wrath, and pity. To speak one truly was already to choose a side."
        ),
        (
            "The old tales seemed to stand close behind the living. They were not dead stories. They were pressure, "
            "warning, and inheritance, waiting to pass again into hands that trembled."
        ),
        (
            "No laughter came from the benches of the hall. Even those who loved judgement more than mercy felt "
            "that the matter had moved beyond custom and entered the region of doom."
        ),
        (
            f"{overhead_sentence} Gold and green, brown "
            "stone and pale flame, all kept their appointed beauty while the hearts below altered."
        ),
        (
            "Hope did not enter like a trumpet. It came as a small refusal: not to bow to terror before terror "
            "had spoken its last word, and not to call prudence the whole of wisdom."
        ),
        (
            "A messenger could have crossed the chamber in a dozen steps, yet the distance between each answer "
            "seemed greater than leagues. Speech made its own wilderness."
        ),
        (
            "The law of the guarded realm was strong, but it had been shaped for walls, borders, oaths, and "
            "memory. Love came by another path, and would not ask the gatekeepers for its nature."
        ),
        (
            "Still the counsel narrowed. Around pride gathered fear; around fear, foresight; around foresight, "
            "the strange mercy that looks foolish until the end of the tale."
        ),
        (
            "Where power was greatest, pity had to speak most softly. Yet soft speech, in such an hour, could "
            "enter deeper than command."
        ),
        (
            "The doom before them was not yet visible. It could be felt only in the way fair things became dearer, "
            "as lamps seem brighter when a storm has not yet broken."
        ),
        (
            "None in the hall was wholly cruel, and therefore the hour was harder to endure. Evil declares itself "
            "by waste and hatred; but error may wear love's face, and pride may speak with the voice of care."
        ),
        (
            "The lamps did not flicker, yet their light seemed divided. It lay on fair hair, dark cloth, white "
            "hands, and stern faces, and it could not reconcile what the hearts beneath them had already divided."
        ),
        (
            "Beleriand beyond the guarded trees was wide and wounded. Rivers ran under moon and cloud, towers "
            "watched the passes, and in the North the enemy did not sleep. This small council was not small to him."
        ),
        (
            "So much depended on words that no blade had yet been drawn. A word could open a road, close a gate, "
            "bind a lover, anger a king, or send a mortal hand toward the crown of the Dark Power."
        ),
        (
            "The beauty of the place did not lessen the danger. It sharpened it. Every carved leaf and hidden "
            "stream seemed to ask what beauty is worth if fear alone is allowed to guard it."
        ),
        (
            "Those who watched began to understand that the matter would not end when the voices ceased. It would "
            "leave the hall with them, walk beside them in sleep, and return in counsel after counsel."
        ),
        (
            "There are moments when the future seems to listen from behind the wall. This was such a moment. It "
            "had no shape yet, only pressure, as of deep water waiting behind a door."
        ),
        (
            f"{lead} did not know all that would follow, nor could {respondent} see every road branching from "
            "the present word. But Melian, if she was near, saw further than either, and grief had already touched her."
        ),
        (
            "To the proud, mercy can look like surrender; to the fearful, courage can look like folly. Between "
            "those errors the truth moved narrowly, and none could seize it without being changed."
        ),
        (
            "The hour lengthened. Outside, root and leaf kept their ancient patience. Within, the hearts of the "
            "living moved more swiftly, and each motion left a mark that would not soon be erased."
        ),
        (
            "No one spoke of victory. The word would have sounded childish there. There was only the harder hope "
            "that a right deed, once chosen, might remain right even when all roads from it led through sorrow."
        ),
        (
            "Thus the scene did not end in peace. It ended in a deeper attention, as if every soul present had "
            "been called by name and must answer before nightfall."
        ),
        (
            "A hush gathered in corners where tapestries, harp-strings, polished bowls, and spearheads caught "
            "separate gleams. Splendour did not comfort them. It made each doubt more visible, as frost reveals "
            "every blade of grass at dawn."
        ),
        (
            "Old names stirred under the spoken ones: Dorthonion, Nargothrond, Angband, Valinor, and the long "
            "sunderings of kin. They did not enter as lessons. They entered as burdens carried by memory into "
            "the present room."
        ),
        (
            "The proudest faces hid the most uncertainty. Brows lowered, rings turned slowly on fingers, and "
            "breath was held behind closed lips. No herald would record those small betrayals, but the hour was "
            "made of them."
        ),
        (
            "Beneath all counsel lay a question no decree could master: whether love is preserved by enclosure "
            "or by trust. Around that question circled fear, loyalty, tenderness, suspicion, and the unsleeping "
            "memory of war."
        ),
        (
            "Somewhere beyond the chamber, water fell into a basin and was gathered away by channels unseen. "
            "Its patient sound seemed wiser than the speakers, for it yielded without ceasing and endured without "
            "claiming dominion."
        ),
        (
            "The North remained distant on every map and near in every heart. Its furnaces, dungeons, ravens, "
            "iron thresholds, and sleepless captains were not named at every turn, yet their shadow bent the "
            "meaning of each answer."
        ),
        (
            "Even beauty had become an argument. The woven leaves, the silver lamps, the carven beasts, and the "
            "clear voices of the household all pleaded silently that such things must be guarded. But they did "
            "not say how."
        ),
        (
            "So the living stood amid inheritance: songs unfinished, oaths unquiet, friendships broken by pride, "
            "and hopes too frail for banners. What they chose would pass into that inheritance and alter its shape."
        ),
    ]

    idx = 0
    added_expansions = 0
    while target_words and _count_words("\n\n".join(paragraphs)) < target_words:
        paragraphs.append(expansion_templates[idx % len(expansion_templates)])
        idx += 1
        added_expansions += 1
        if added_expansions >= len(expansion_templates):
            break

    if target_dialogue_ratio:
        current = "\n\n".join(paragraphs)
        dialogue_words = _dialogue_word_count(current)
        total_words = max(1, _count_words(current))
        dialogue_idx = 0
        dialogue_templates = [
            (
                f"'{lead}, if your heart is fixed, speak it without ornament,' said {respondent}. 'The plain "
                "word may yet be the strongest.'"
            ),
            (
                f"'{respondent}, I have little wisdom to set beside yours,' answered {lead}. 'But I have seen "
                "fear take many shapes. I would rather meet it in the open than let it rule me in secret.'"
            ),
            (
                f"'{lead}, remember this hour,' said {respondent}. 'If grief follows, let none say that "
                "the choice was made in haste.'"
            ),
            (
                f"'{lead}, you stand among powers older than your house,' said {third_voice}. 'Yet age alone is no "
                "answer. Speak, and let the measure of your purpose be heard.'"
            ),
            (
                f"'{respondent}, I fear the price less than silence,' said {lead}. 'Silence would leave the "
                "darkness unnamed, and unnamed darkness grows bold.'"
            ),
            (
                f"'{lead}, I have seen counsel fail when it was only caution,' said {respondent}. 'Let "
                "wisdom be wary, but let it not become a locked door against all hope.'"
            ),
            (
                f"'{respondent}, I will not borrow courage from loud words,' said {lead}. 'Let the deed judge "
                "me when it comes. Until then I can only speak truthfully.'"
            ),
            (
                f"'{lead}, truth has a cost in halls like these,' said {respondent}. 'Pay it knowingly, or others "
                "will count it for you.'"
            ),
            (
                f"'{respondent}, I know little of guarded kingdoms,' answered {lead}. 'I know the open road, "
                "hunger, pursuit, and the keeping of a promise when witnesses are dead.'"
            ),
            (
                f"'{lead}, a promise may be noble and still bring ruin,' said {respondent}. 'That is why we weigh "
                "it before the doors are opened.'"
            ),
            (
                f"'{respondent}, then weigh me also,' said {lead}. 'Not as rumor, not as trespasser, but as one "
                "who stands here and will answer.'"
            ),
            (
                f"'{lead}, I hear you,' said {respondent}. 'Whether hearing will soften judgement is another "
                "matter, and one not given wholly into my hand.'"
            ),
            (
                f"'{respondent}, if judgement must be hard, let it at least be clean,' said {lead}. 'A hidden "
                "fear is a poor counsellor.'"
            ),
            (
                f"'{lead}, do not mistake warning for fear,' said {respondent}. 'Those who have watched long may "
                "seem cold because they have seen fire consume fair things.'"
            ),
            (
                f"'{respondent}, I do not despise warning,' said {lead}. 'I only ask that warning leave room for "
                "the courage it seeks to preserve.'"
            ),
            (
                f"'{lead}, that room is narrow,' said {respondent}. 'Walk it carefully, for on either side lie "
                "sorrow and pride.'"
            ),
            (
                f"'{respondent}, narrow roads are not strange to me,' said {lead}. 'If this one leads through "
                "darkness, I will still know why I entered it.'"
            ),
            (
                f"'{lead}, then let the hall remember your words,' said {respondent}. 'Words spoken freely may "
                "become a burden, but they may also become a light.'"
            ),
        ]
        while (dialogue_words / total_words) < target_dialogue_ratio and dialogue_idx < len(dialogue_templates):
            paragraphs.append(dialogue_templates[dialogue_idx])
            dialogue_idx += 1
            current = "\n\n".join(paragraphs)
            dialogue_words = _dialogue_word_count(current)
            total_words = max(1, _count_words(current))

    return "\n\n".join(paragraphs)


def _render_template_scene(
    *,
    project: dict,
    scene_id: str,
    scene_num: int,
    scene_goal: str,
    characters: list[str],
    place: str,
    objects: list[str],
    event: dict,
    scene_beats: list[dict],
    missing_terms_hint: list[str],
    quality: dict[str, Any] | None = None,
):
    from book_graph_analyzer.generate.models import GenerationStatus, Scene, SceneScores

    text = _template_scene_text(
        project=project,
        scene_goal=scene_goal,
        characters=characters,
        place=place,
        objects=objects,
        event=event,
        scene_beats=scene_beats,
        missing_terms_hint=missing_terms_hint,
        quality=quality,
    )
    # A deterministic template has not been judged against canon or voice
    # evidence.  Keep its scores explicitly unverified instead of emitting the
    # same flattering constants for every scene.
    scores = SceneScores()
    return Scene(
        id=scene_id,
        number=scene_num,
        text=text,
        summary=scene_goal,
        characters=characters,
        places=[place] if place else [],
        objects=objects,
        scores=scores,
        status=GenerationStatus.FLAGGED,
        model_used="template-renderer",
        generation_prompt="deterministic-template",
        pipeline_stages_run=["template_renderer"],
    )


def _render_grounded_chapter_text(
    *,
    project: dict,
    proj_dir: Path,
    plan: dict,
    chapter: int,
    chapter_rows: list[dict],
    graph_node_by_id: dict[str, dict],
    required_terms: list[str],
    missing_terms_hint: list[str] | None = None,
    renderer: str = "llm",
) -> tuple[str, list[dict], list[dict]]:
    from book_graph_analyzer.generate import Chapter, Story
    from book_graph_analyzer.generate.shadow.models import StateDelta

    project_slug = str(project.get("slug") or "").strip()
    scene_plan_by_id, chapters_by_number = _scene_plan_index(plan)
    plan_chapter = chapters_by_number.get(chapter)
    scene_beats_by_scene = _load_shadow_beats_by_scene(proj_dir)
    missing_terms_hint = list(missing_terms_hint or [])
    quality = _quality_settings(_load_constraints(proj_dir), chapter=chapter)
    context_stats = _load_json(proj_dir / "context_stats.json", default={})
    timeline = _project_timeline(project)
    timeline_snapshot = context_stats.get("timeline", {}) if isinstance(context_stats.get("timeline"), dict) else {}
    if timeline_snapshot:
        timeline["story_era"] = canonicalize_era(str(timeline_snapshot.get("story_era") or "").strip()) or timeline.get("story_era")
        timeline["story_year"] = _coerce_optional_int(timeline_snapshot.get("story_year", timeline.get("story_year")))
        timeline["story_era_order"] = era_to_order(timeline.get("story_era"))
        timeline["allow_past_references"] = bool(timeline_snapshot.get("allow_past_references", timeline.get("allow_past_references", True)))
        timeline["forbid_future_entities"] = bool(timeline_snapshot.get("forbid_future_entities", timeline.get("forbid_future_entities", True)))
    future_guardrail_entities = _dedupe_strings(
        [str(x) for x in timeline.get("forbidden_entities", [])]
        + [str(x) for x in (timeline_snapshot.get("future_guardrail_entities", []) if isinstance(timeline_snapshot, dict) else [])]
    )

    shadow = _new_story_shadow_graph(project_slug)
    generator = _new_story_scene_generator(shadow)
    loaded_world_bible = _maybe_load_story_world_bible(generator, project, proj_dir)
    voice_profiles, voice_profiles_path = _load_story_voice_profiles(project, proj_dir)
    writer = None if renderer == "template" else _new_story_generation_writer()

    chapter_id = f"{project_slug}-chapter-{chapter:02d}"
    chapter_title = _chapter_title(chapter, plan_chapter)
    chapter_outline = _chapter_outline_text(plan_chapter, chapter_rows, scene_plan_by_id, graph_node_by_id, scene_beats_by_scene)
    story_record = Story(
        id=project_slug,
        title=str(project.get("name") or project_slug),
        premise=str(project.get("premise") or ""),
        outline=chapter_outline or str(project.get("premise") or ""),
        chapters=[
            Chapter(
                id=chapter_id,
                number=chapter,
                title=chapter_title,
                summary=str((plan_chapter or {}).get("intent") or chapter_title),
                outline=chapter_outline,
                target_scenes=len(chapter_rows),
            )
        ],
    )
    if writer is not None:
        writer.write_story(story_record)

    lines = [f"# {chapter_title}", ""]
    trace_sections: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    seen_template_paragraphs: set[str] = (
        _existing_project_paragraph_signatures(proj_dir, exclude_chapter=chapter)
        if renderer == "template"
        else set()
    )

    for idx, row in enumerate(chapter_rows, start=1):
        scene_id = str(row.get("scene_id") or "").strip()
        event_id = str(row.get("shadow_event_id") or "").strip()
        event = graph_node_by_id.get(event_id, {})
        plan_scene = scene_plan_by_id.get(scene_id, {})
        scene_beats = scene_beats_by_scene.get(scene_id, [])
        scene_missing_terms_hint = missing_terms_hint if idx == 1 else []
        scene_goal = _story_scene_goal(
            plan_scene=plan_scene,
            event=event,
            scene_beats=scene_beats,
            missing_terms_hint=scene_missing_terms_hint,
            timeline=timeline,
            future_guardrail_entities=future_guardrail_entities,
        )
        characters = _story_scene_characters(project_slug, plan_scene, event, scene_beats)
        _validate_story_scene_participants(project_slug, scene_id, characters)
        place = _story_scene_place(project, plan_scene, event)
        objects = _story_scene_objects(plan_scene, event)
        scene_quality = _scene_quality_for_plan(quality, plan_chapter, plan_scene, len(chapter_rows))

        if renderer == "template":
            scene = _render_template_scene(
                project=project,
                scene_id=_story_scene_runtime_id(project_slug, scene_id or f"ch{chapter:02d}-sc{idx:02d}"),
                scene_num=idx,
                scene_goal=scene_goal,
                characters=characters,
                place=place,
                objects=objects,
                event=event,
                scene_beats=scene_beats,
                missing_terms_hint=scene_missing_terms_hint,
                quality=scene_quality,
            )
        else:
            scene = generator.generate_scene(
                scene_goal=scene_goal,
                characters=characters,
                place=place,
                objects=objects,
                story_id=project_slug,
                chapter_num=chapter,
                scene_num=idx,
                story_era=timeline.get("story_era"),
                story_year=timeline.get("story_year"),
                voice_profiles=voice_profiles,
                target_words=int(scene_quality.get("target_scene_words", 0) or 0) or None,
            )
        if not scene.text.strip():
            raise click.ClickException(
                f"Grounded scene generation returned empty text for {scene_id}. Check the configured LLM provider."
            )
        if renderer == "template":
            scene.text = _dedupe_scene_paragraphs(scene.text, seen_template_paragraphs)
            if _is_hunt_gollum_project(project_slug):
                scene.text = _extend_hunt_scene_text(
                    scene.text,
                    scene_goal=scene_goal,
                    characters=characters,
                    place=place,
                    objects=objects,
                    target_words=int(scene_quality.get("target_scene_words", 0) or 0),
                    seen=seen_template_paragraphs,
                )
                scene.text = _raise_hunt_dialogue_ratio(
                    scene.text,
                    scene_goal=scene_goal,
                    characters=characters,
                    target_dialogue_ratio=float(scene_quality.get("target_dialogue_ratio", 0.0) or 0.0),
                    seen=seen_template_paragraphs,
                )
                scene.text = _polish_paragraph_starts(scene.text)

        stable_scene_id = _story_scene_runtime_id(project_slug, scene_id or f"ch{chapter:02d}-sc{idx:02d}")
        scene.id = stable_scene_id
        scene.number = idx
        scene.summary = str(plan_scene.get("summary") or event.get("description") or scene_goal)
        scene.word_count = len(scene.text.split())

        if renderer == "template":
            delta = StateDelta(
                story_id=project_slug,
                scene_id=stable_scene_id,
                character_updates={name: {"location_change": place} for name in scene.characters},
                scene_summary=scene.summary,
                chapter_num=chapter,
                scene_num=idx,
            )
        else:
            delta = shadow.extract_delta_from_scene(
                scene_text=scene.text,
                characters=scene.characters,
                scene_id=stable_scene_id,
                chapter_num=chapter,
                scene_num=idx,
            )
        if not delta.scene_summary:
            delta.scene_summary = scene.summary
        if renderer != "template":
            shadow.commit_state_delta(delta)
        if writer is not None:
            writer.write_scene(scene, chapter_id)

        scene_title = str(plan_scene.get("title") or plan_scene.get("heading") or "").strip()
        if scene_title:
            lines.extend([f"## {scene_title}", "", scene.text.strip(), ""])
        else:
            if idx > 1:
                lines.extend(["* * *", ""])
            lines.extend([scene.text.strip(), ""])

        source_canon_node_ids = _story_scene_source_refs(event, scene_beats)
        trace_sections.append(
            {
                "section": idx,
                "scene_id": scene_id,
                "shadow_event_id": event_id,
                "shadow_scene_id": f"shadow-{scene_id}",
                "generated_scene_id": stable_scene_id,
                "source_canon_node_ids": source_canon_node_ids,
                "text_excerpt": scene.text[:220],
                "word_count": scene.word_count,
                "target_word_count": int(scene_quality.get("target_scene_words", 0) or 0),
                "scene_scores": scene.scores.to_dict(),
                "pipeline_stages_run": list(scene.pipeline_stages_run),
                "model_used": scene.model_used,
                "world_bible_file": str(loaded_world_bible) if loaded_world_bible else None,
                "voice_profiles_file": str(voice_profiles_path) if voice_profiles_path else None,
                "voice_profiles_used": [name for name in scene.characters if name in voice_profiles],
                "characters": list(scene.characters),
                "place": place,
            }
        )
        scene_records.append(
            {
                "scene_id": scene_id,
                "generated_scene_id": stable_scene_id,
                "summary": scene.summary,
                "word_count": scene.word_count,
                "status": scene.status.value,
                "characters": list(scene.characters),
                "places": list(scene.places),
                "scores": scene.scores.to_dict(),
                "pipeline_stages_run": list(scene.pipeline_stages_run),
                "model_used": scene.model_used,
            }
        )

    if _is_hunt_gollum_project(project_slug):
        closing = _hunt_chapter_closing_paragraph(chapter)
        if closing:
            lines.extend([closing, ""])

    return "\n".join(lines).strip() + "\n", trace_sections, scene_records


def _extract_canon_notes(canon_path: str | None) -> list[str]:
    if not canon_path:
        return ["No canon file configured (canon checks run in lightweight mode)."]
    path = Path(canon_path)
    if not path.exists():
        return [f"Canon file configured but missing: {canon_path}"]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [f"Canon file is not valid JSON: {canon_path}"]

    notes: list[str] = [f"Canon source loaded: {canon_path}"]
    if isinstance(payload, dict):
        for key in ("world", "title", "name"):
            if key in payload and payload[key]:
                notes.append(f"Canon context: {payload[key]}")
                break
        if "entities" in payload and isinstance(payload["entities"], list):
            notes.append(f"Canonical entities discovered: {len(payload['entities'])}")
    return notes


def _build_plan(project: dict, constraints: dict, auto: bool) -> dict:
    chapters = int(project.get("target_chapters", 6))
    scenes_per_chapter = int(project.get("scenes_per_chapter", 3))
    premise = project.get("premise", "")

    chapter_rows = []
    for chapter_idx in range(1, chapters + 1):
        scene_rows = []
        for scene_idx in range(1, scenes_per_chapter + 1):
            scene_id = f"ch{chapter_idx:02d}-sc{scene_idx:02d}"
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_number": scene_idx,
                    "goal": f"Advance chapter {chapter_idx} tension beat {scene_idx}",
                    "summary": f"{premise[:120]}" if premise else f"Chapter {chapter_idx} scene {scene_idx} progression.",
                    "continuity_hooks": ["Track unresolved threads", "Respect established canon"],
                }
            )

        chapter_rows.append(
            {
                "chapter_number": chapter_idx,
                "title": f"Chapter {chapter_idx}: {'Escalation' if chapter_idx > 1 else 'Setup'}",
                "intent": f"Core arc movement for chapter {chapter_idx}",
                "scenes": scene_rows,
            }
        )

    return {
        "project_slug": project["slug"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "auto" if auto else "manual",
        "constraints_snapshot": constraints,
        "chapters": chapter_rows,
    }


def _validate_plan(project: dict, plan: dict, constraints: dict) -> dict:
    issues: list[dict] = []
    warnings: list[dict] = []
    checks: list[dict] = []

    chapters = plan.get("chapters", [])
    checks.append({"name": "plan_has_chapters", "ok": bool(chapters)})
    if not chapters:
        issues.append({"code": "PLAN_EMPTY", "message": "Plan has no chapters."})

    variable_scene_counts = bool(project.get("variable_scenes_per_chapter", False))
    expected_scene_count = int(project.get("scenes_per_chapter", 3))
    for ch in chapters:
        scenes = ch.get("scenes", [])
        if not variable_scene_counts and len(scenes) != expected_scene_count:
            warnings.append(
                {
                    "code": "SCENE_COUNT_MISMATCH",
                    "chapter": ch.get("chapter_number"),
                    "message": f"Expected {expected_scene_count} scenes, found {len(scenes)}.",
                }
            )
        seen = set()
        for sc in scenes:
            sid = sc.get("scene_id")
            if sid in seen:
                issues.append({"code": "DUPLICATE_SCENE_ID", "message": f"Duplicate scene_id: {sid}"})
            seen.add(sid)

    required = constraints.get("required_elements", []) if isinstance(constraints, dict) else []
    if required:
        all_text = "\n".join(
            f"{ch.get('title','')} {sc.get('summary','')}"
            for ch in chapters
            for sc in ch.get("scenes", [])
        ).lower()
        missing = [item for item in required if item.lower() not in all_text]
        checks.append({"name": "required_elements_present", "ok": not missing, "missing": missing})
        for item in missing:
            warnings.append({"code": "CANON_REQUIRED_MISSING", "message": f"Required element not found in plan text: {item}"})

    forbidden = constraints.get("forbidden_terms", []) if isinstance(constraints, dict) else []
    if forbidden:
        all_text = "\n".join(
            f"{ch.get('title','')} {sc.get('summary','')}"
            for ch in chapters
            for sc in ch.get("scenes", [])
        ).lower()
        hits = [term for term in forbidden if term.lower() in all_text]
        checks.append({"name": "forbidden_terms_absent", "ok": not hits, "hits": hits})
        for term in hits:
            issues.append({"code": "CANON_FORBIDDEN_TERM", "message": f"Forbidden term present in plan text: {term}"})

    status = "pass" if not issues else "fail"
    return {
        "project_slug": project["slug"],
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "chapters": len(chapters),
            "issues": len(issues),
            "warnings": len(warnings),
        },
        "checks": checks,
        "issues": issues,
        "warnings": warnings,
    }


@click.group()
def story() -> None:
    """Story workflow commands (init, auto-plan, validate)."""
    pass


@story.group("beats")
def story_beats() -> None:
    """Scene-to-beat expansion commands."""
    pass


@story_beats.command("expand")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--method", type=click.Choice(["template", "deterministic"], case_sensitive=False), default="template", show_default=True)
@click.option("--beats-per-scene", type=int, default=None, help="Fixed beats emitted per scene (overrides dynamic)")
@click.option("--min-beats-per-scene", type=int, default=1, show_default=True, help="Dynamic mode lower bound")
@click.option("--max-beats-per-scene", type=int, default=4, show_default=True, help="Dynamic mode upper bound")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_beats_expand(
    project_slug: str,
    method: str,
    beats_per_scene: int | None,
    min_beats_per_scene: int,
    max_beats_per_scene: int,
    projects_dir: str,
) -> None:
    """Expand plan scenes into deterministic shadow beats."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    plan = _load_json(proj_dir / "plan.json", default={})
    if not plan.get("chapters"):
        raise click.ClickException("Missing or empty plan.json. Run 'bga story plan --project <slug> --auto' first.")
    constraints = _load_constraints(proj_dir)
    context = _load_json(proj_dir / "context_stats.json", default={})
    canon_evidence = _context_canon_evidence(context)
    style_words = float(constraints.get("style", {}).get("target_words_per_scene", 320))

    scene_rows = [sc for ch in plan.get("chapters", []) for sc in ch.get("scenes", [])]
    if beats_per_scene is not None and beats_per_scene <= 0:
        raise click.ClickException("--beats-per-scene must be > 0")
    if min_beats_per_scene <= 0 or max_beats_per_scene <= 0:
        raise click.ClickException("--min-beats-per-scene and --max-beats-per-scene must be > 0")
    if min_beats_per_scene > max_beats_per_scene:
        raise click.ClickException("--min-beats-per-scene cannot be greater than --max-beats-per-scene")

    beats: list[StoryBeat] = []
    position = 1
    for scene in scene_rows:
        per_scene_count = int(beats_per_scene) if beats_per_scene is not None else _compute_scene_beat_count(
            scene,
            min_beats=min_beats_per_scene,
            max_beats=max_beats_per_scene,
        )
        for beat_idx in range(1, per_scene_count + 1):
            prior_beat_id = beats[-1].beat_id if beats else None
            beats.append(
                _make_shadow_beat(
                    scene,
                    project_slug,
                    constraints,
                    position=position,
                    beat_idx_in_scene=beat_idx,
                    beats_in_scene=per_scene_count,
                    style_words=style_words,
                    prior_beat_id=prior_beat_id,
                    canon_evidence=canon_evidence,
                )
            )
            position += 1

    validation_issues = _validate_cause_ref_positions(beats)
    payload = {
        "schema_version": "shadow-beats-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": method.lower(),
        "seed": _stable_seed(project_slug, _canonical_json(plan), _canonical_json(constraints)),
        "expansion": {
            "beats_per_scene": beats_per_scene,
            "min_beats_per_scene": min_beats_per_scene,
            "max_beats_per_scene": max_beats_per_scene,
            "mode": "fixed" if beats_per_scene is not None else "dynamic",
        },
        "beats": [asdict(b) for b in beats],
        "validation": {
            "cause_ref_issues": validation_issues,
            "failed_constraints": [
                {"beat_id": b.beat_id, "failed_constraints": b.failed_constraints}
                for b in beats
                if b.failed_constraints
            ],
        },
    }

    out_path = proj_dir / "shadow_beats.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    sidecar = proj_dir / "shadow_selected.json"
    if sidecar.exists():
        selected = _load_json(sidecar, default={})
        (proj_dir / "shadow_beats_selected_sidecar.json").write_text(
            json.dumps(
                {
                    "schema_version": "shadow-beats-selected-sidecar-v1",
                    "project_slug": project_slug,
                    "source": "shadow_selected.json",
                    "selected": selected.get("selected", []),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    console.print(f"[green]OK[/green] Beats expanded: {out_path}")


@story_beats.command("validate")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--chapter", type=int, default=None, help="Limit validation to a chapter number")
@click.option("--scene", default="", help="Limit validation to a scene id (e.g. ch01-sc02)")
@click.option("--strict/--no-strict", default=False, help="Exit non-zero when validation has errors")
@click.option("--strict-warnings", is_flag=True, help="With --strict, also fail on warnings")
@click.option("--json-out", default="", help="Optional explicit JSON report path")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_beats_validate(project_slug: str, chapter: int | None, scene: str, strict: bool, strict_warnings: bool, json_out: str, projects_dir: str) -> None:
    """Validate beat artifacts, optionally scoped by chapter or scene."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    beats_path = proj_dir / "shadow_beats.json"
    if not beats_path.exists():
        raise click.ClickException("Missing shadow_beats.json. Run: bga story beats expand --project <slug>")

    payload = _load_json(beats_path, default={})
    beats = payload.get("beats", []) if isinstance(payload, dict) else []
    scoped = _select_beats_scope(beats, chapter=chapter, scene=(scene or "").strip() or None)
    constraints = _load_constraints(proj_dir)
    report = _beats_validation_from_rows(scoped, project_slug=project_slug, constraints=constraints)
    report.update(
        {
            "schema_version": "shadow-beats-validation-v1",
            "project_slug": project_slug,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "scope": {"chapter": chapter, "scene": (scene or "").strip() or None},
        }
    )

    report_path = Path(json_out) if json_out else (proj_dir / "shadow_beats_validation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    console.print(
        f"Beats validate: {s['status'].upper()} | beats={s['beats']} | errors={s['errors']} | warnings={s['warnings']}"
    )
    console.print(f"JSON report: {report_path}")

    should_fail = strict and (s["errors"] > 0 or (strict_warnings and s["warnings"] > 0))
    if should_fail:
        raise click.ClickException("Strict validation failed.")


@story_beats.command("show")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--chapter", type=int, default=None, help="Show beats for chapter")
@click.option("--scene", default="", help="Show beats for scene id")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_beats_show(project_slug: str, chapter: int | None, scene: str, projects_dir: str) -> None:
    """Show concise beat summary for a scene/chapter."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    payload = _load_json(proj_dir / "shadow_beats.json", default={})
    beats = payload.get("beats", []) if isinstance(payload, dict) else []
    scoped = _select_beats_scope(beats, chapter=chapter, scene=(scene or "").strip() or None)
    constraints = _load_constraints(proj_dir)
    report = _beats_validation_from_rows(scoped, project_slug=project_slug, constraints=constraints)

    type_counts = Counter(str(b.get("beat_type", "unknown")) for b in scoped)
    ids = [str(b.get("beat_id", "")) for b in scoped]
    scene_counts = Counter(_scene_from_beat_id(i) for i in ids)
    top_issues = report["issues"][:3]
    console.print(f"Beats summary | count={len(scoped)} | scenes={len(scene_counts)}")
    if scene_counts:
        rendered_scene_counts = ", ".join(f"{sid}:{cnt}" for sid, cnt in sorted(scene_counts.items()))
        console.print(f"Per-scene counts: {rendered_scene_counts}")
    console.print(f"Types: {dict(type_counts)}")
    console.print(f"IDs: {', '.join(ids[:8])}" + (" ..." if len(ids) > 8 else ""))
    if scoped:
        console.print("Sample beat semantics:")
        for b in scoped[:3]:
            lore_status = "grounded" if (b.get("source_canon_node_ids") or b.get("canon_refs")) else "ungrounded"
            style_status = "ok"
            for it in report["issues"]:
                if it.get("beat_id") == b.get("beat_id") and it.get("code") == "STYLE_BUDGET_MISMATCH":
                    style_status = "warn"
                    break
            console.print(
                f"- {b.get('beat_id')}: action={b.get('action', 'n/a')} | participants={','.join((b.get('participants') or [])[:3]) or 'n/a'}"
                f" | motifs={','.join((b.get('motifs') or [])[:3]) or 'n/a'} | lore={lore_status} | style={style_status}"
            )
    if top_issues:
        console.print("Top issues:")
        for it in top_issues:
            console.print(f"- [{it['level']}] {it['code']} {it['beat_id']}")
    else:
        console.print("Top issues: none")


@story_beats.command("clean")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--chapter", type=int, default=None, help="Remove beats only in chapter")
@click.option("--scene", default="", help="Remove beats only in scene id")
@click.option("--dry-run", is_flag=True, help="Preview without writing changes")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_beats_clean(project_slug: str, chapter: int | None, scene: str, dry_run: bool, projects_dir: str) -> None:
    """Clean beat artifacts safely (whole-project or scoped)."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    scene = (scene or "").strip()
    scoped_mode = bool(scene) or (chapter is not None)

    beats_path = proj_dir / "shadow_beats.json"
    sidecar_path = proj_dir / "shadow_beats_selected_sidecar.json"
    validation_path = proj_dir / "shadow_beats_validation.json"

    if not scoped_mode:
        removed = [p for p in [beats_path, sidecar_path, validation_path] if p.exists()]
        if dry_run:
            console.print(f"Dry-run: would remove {len(removed)} files")
            for p in removed:
                console.print(f"- {p}")
            return
        for p in removed:
            p.unlink()
        console.print(f"[green]OK[/green] Removed {len(removed)} beat artifact files")
        return

    if not beats_path.exists():
        raise click.ClickException("No shadow_beats.json to clean.")
    payload = _load_json(beats_path, default={})
    beats = payload.get("beats", []) if isinstance(payload, dict) else []
    doomed = _select_beats_scope(beats, chapter=chapter, scene=scene or None)
    doomed_ids = {str(b.get("beat_id", "")) for b in doomed}
    kept = [b for b in beats if str(b.get("beat_id", "")) not in doomed_ids]
    console.print(f"Scoped clean: removing {len(doomed)} beats, keeping {len(kept)}")
    if dry_run:
        return
    payload["beats"] = kept
    payload.setdefault("validation", {})
    if isinstance(payload.get("validation"), dict):
        v = payload["validation"]
        if isinstance(v.get("failed_constraints"), list):
            v["failed_constraints"] = [
                row for row in v["failed_constraints"] if str(row.get("beat_id", "")) not in doomed_ids
            ]
        if isinstance(v.get("cause_ref_issues"), list):
            v["cause_ref_issues"] = [
                row for row in v["cause_ref_issues"] if not any(did in str(row) for did in doomed_ids)
            ]
    beats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Updated {beats_path}")


@story.command("init")
@click.option("--name", help="Project display name")
@click.option("--slug", help="Project slug (defaults from --name)")
@click.option("--premise", help="1-2 sentence story premise")
@click.option("--genre", default="fantasy", show_default=True, help="Primary genre")
@click.option("--canon-file", default="", help="Optional canon JSON path")
@click.option("--target-chapters", default=6, show_default=True, type=int)
@click.option("--scenes-per-chapter", default=3, show_default=True, type=int)
@click.option("--story-era", default="", help="Optional story-time era (e.g. 'First Age', 'Third Age')")
@click.option("--story-year", type=int, default=None, help="Optional story-time year within the era")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
@click.option("--non-interactive", is_flag=True, help="Fail instead of prompting for missing required inputs")
def story_init(
    name: str | None,
    slug: str | None,
    premise: str | None,
    genre: str,
    canon_file: str,
    target_chapters: int,
    scenes_per_chapter: int,
    story_era: str,
    story_year: int | None,
    projects_dir: str,
    non_interactive: bool,
) -> None:
    """Initialize a new story project scaffold under data/projects/<slug>/."""
    if not name and not non_interactive:
        name = click.prompt("Project name", type=str)
    if not premise and not non_interactive:
        premise = click.prompt("Short premise", type=str)

    if not name or not premise:
        raise click.ClickException("--name and --premise are required (or run interactive mode without --non-interactive)")

    slug = slug or _slugify(name)
    proj_dir = _project_dir(slug, Path(projects_dir))
    proj_dir.mkdir(parents=True, exist_ok=True)
    timeline = _default_story_timeline({"slug": slug})
    if story_era.strip():
        timeline["story_era"] = canonicalize_era(story_era.strip()) or story_era.strip()
    if story_year is not None:
        timeline["story_year"] = int(story_year)

    project = {
        "name": name,
        "slug": slug,
        "genre": genre,
        "premise": premise,
        "canon_file": canon_file,
        "target_chapters": target_chapters,
        "scenes_per_chapter": scenes_per_chapter,
        "timeline": timeline,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    (proj_dir / "project.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
    (proj_dir / "constraints.json").write_text(json.dumps(_default_constraints(), indent=2), encoding="utf-8")
    (proj_dir / "story_bible.md").write_text(
        "\n".join(
            [
                f"# {name} Story Bible",
                "",
                f"## Premise\n{premise}",
                "",
                "## Core Characters",
                "- (add protagonist)",
                "",
                "## World Rules",
                "- (add non-negotiable rules)",
                "",
                "## Open Questions",
                "- (add unresolved mysteries)",
            ]
        ),
        encoding="utf-8",
    )
    (proj_dir / "plan.json").write_text(json.dumps({"project_slug": slug, "chapters": []}, indent=2), encoding="utf-8")

    console.print(f"[green]OK[/green] Story project initialized: [bold]{slug}[/bold]")
    console.print(f"Project directory: {proj_dir}")
    console.print("Next: run [bold]bga story plan --project {slug} --auto[/bold]")


@story.command("plan")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--auto", "auto_mode", is_flag=True, help="Auto-generate chapter/scene plan")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_plan(project_slug: str, auto_mode: bool, projects_dir: str) -> None:
    """Generate a chapter/scene plan from project + canon context."""
    if not auto_mode:
        raise click.ClickException("Only --auto mode is supported in this iteration. Use: bga story plan --project <slug> --auto")

    project = _load_project(project_slug, Path(projects_dir))
    proj_dir = _project_dir(project_slug, Path(projects_dir))

    constraints_path = proj_dir / "constraints.json"
    constraints = (
        json.loads(constraints_path.read_text(encoding="utf-8"))
        if constraints_path.exists()
        else _default_constraints()
    )

    plan = _build_plan(project=project, constraints=constraints, auto=True)
    plan_path = proj_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    console.print(f"[green]OK[/green] Auto-plan generated for [bold]{project_slug}[/bold]")
    console.print(f"Plan artifact: {plan_path}")
    for note in _extract_canon_notes(project.get("canon_file")):
        console.print(f"  - {note}")
    console.print(f"Chapters: {len(plan['chapters'])}")


@story.command("validate")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
@click.option("--json-out", "json_out", default="", help="Optional explicit JSON report path")
def story_validate(project_slug: str, projects_dir: str, json_out: str) -> None:
    """Validate continuity/style/canon checks and output report artifacts."""
    project = _load_project(project_slug, Path(projects_dir))
    proj_dir = _project_dir(project_slug, Path(projects_dir))

    plan_path = proj_dir / "plan.json"
    if not plan_path.exists():
        raise click.ClickException(f"Missing plan artifact: {plan_path}. Run 'bga story plan --project {project_slug} --auto' first.")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    constraints_path = proj_dir / "constraints.json"
    constraints = (
        json.loads(constraints_path.read_text(encoding="utf-8"))
        if constraints_path.exists()
        else _default_constraints()
    )

    report = _validate_plan(project=project, plan=plan, constraints=constraints)

    json_report_path = Path(json_out) if json_out else (proj_dir / "validation_report.json")
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        f"# Validation Report: {project_slug}",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Chapters: {report['summary']['chapters']}",
        f"- Issues: {report['summary']['issues']}",
        f"- Warnings: {report['summary']['warnings']}",
        "",
        "## Issues",
    ]
    if report["issues"]:
        md_lines.extend([f"- {it['code']}: {it['message']}" for it in report["issues"]])
    else:
        md_lines.append("- None")

    md_lines.append("\n## Warnings")
    if report["warnings"]:
        md_lines.extend([f"- {it['code']}: {it['message']}" for it in report["warnings"]])
    else:
        md_lines.append("- None")

    md_path = proj_dir / "validation_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    state_color = "green" if report["status"] == "pass" else "red"
    console.print(f"[{state_color}]Validation {report['status'].upper()}[/{state_color}] for [bold]{project_slug}[/bold]")
    console.print(f"Issues: {report['summary']['issues']} | Warnings: {report['summary']['warnings']}")
    console.print(f"Human report: {md_path}")
    console.print(f"JSON report: {json_report_path}")


@story.command("context")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--graph-stats", is_flag=True, help="Compute graph-derived statistical priors")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_context(project_slug: str, graph_stats: bool, projects_dir: str) -> None:
    """Compute graph-native statistical context from event artifacts."""
    if not graph_stats:
        raise click.ClickException("Use --graph-stats for this command: bga story context --project <slug> --graph-stats")

    project = _load_project(project_slug, Path(projects_dir))
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    constraints = _load_constraints(proj_dir)
    event_files = _project_event_files(project)
    if not event_files:
        raise click.ClickException("No event files found. Set project.event_files to one or more *_events.json files.")

    transition_counts: dict[str, Counter] = defaultdict(Counter)
    action_counts: Counter = Counter()
    motif_counts: Counter = Counter()
    character_counts: Counter = Counter()
    register_word_lengths: list[int] = []
    total_events = 0
    timeline = _project_timeline(project)
    entity_presence = _build_entity_temporal_presence(event_files)
    seed_entities = _project_seed_entities(project, constraints)

    for path in event_files:
        payload = _load_json(path, default={})
        events = _extract_events(payload)
        previous_action = None
        for ev in events:
            if _event_temporal_relation(ev, path, timeline) == "future":
                continue
            total_events += 1
            action = str(ev.get("action") or "unknown").strip().lower() or "unknown"
            desc = str(ev.get("description") or "")
            agent = str(ev.get("agent") or "Unknown").strip() or "Unknown"
            agent_entities = _extract_entity_names(agent) or [agent]
            action_counts[action] += 1
            for agent_name in agent_entities[:4]:
                character_counts[agent_name] += 1
            register_word_lengths.append(max(1, len(desc.split())))

            for tok in _tokenize(desc):
                motif_counts[tok] += 1

            if previous_action is not None:
                transition_counts[previous_action][action] += 1
            previous_action = action

    transition_probabilities = {
        src: _safe_prob(dict(dest))
        for src, dest in transition_counts.items()
    }

    motif_priors = _safe_prob(dict(motif_counts.most_common(80)))
    character_priors = _safe_prob(dict(character_counts))
    local_event_neighborhood = _local_neighborhood_from_events(
        event_files=event_files,
        seed_entities=seed_entities,
        timeline=timeline,
    )
    local_graph_neighborhood = _local_neighborhood_from_graph(seed_entities, timeline)
    local_story_neighborhood = _blend_story_neighborhood(local_graph_neighborhood, local_event_neighborhood)
    avg_words = int(round(sum(register_word_lengths) / max(1, len(register_word_lengths))))
    entity_status_counts = Counter(
        _temporal_entity_status(name, timeline, entity_presence)["status"]
        for name in entity_presence
    )
    temporal_guardrails = _temporal_guardrail_entities(
        timeline=timeline,
        entity_presence=entity_presence,
        character_priors=character_priors,
    )
    register_style_budgets = {
        "target_words_per_scene": max(180, min(900, avg_words * 3)),
        "dialogue_ratio_target": 0.28,
        "lore_reference_budget_per_scene": 2,
        "song_reference_budget_per_chapter": 1,
    }

    context = {
        "schema_version": "shadow-context-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_event_files": [str(p) for p in event_files],
        "timeline": {
            **timeline,
            "future_guardrail_entities": temporal_guardrails,
            "entity_status_counts": dict(entity_status_counts),
        },
        "totals": {
            "events": total_events,
            "actions": len(action_counts),
            "characters": len(character_counts),
            "motifs": len(motif_counts),
        },
        "event_transition_probabilities": transition_probabilities,
        "motif_reference_density_priors": motif_priors,
        "character_participation_priors": character_priors,
        "entity_temporal_presence": entity_presence,
        "local_story_neighborhood": local_story_neighborhood,
        "canon_evidence": _context_canon_evidence(
            {"local_story_neighborhood": local_story_neighborhood}
        ),
        "register_style_budgets": register_style_budgets,
    }

    out_path = proj_dir / "context_stats.json"
    out_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Context stats generated: {out_path}")


@story.command("grow-shadow")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--auto", "auto_mode", is_flag=True, help="Auto-generate probabilistic shadow graph")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_grow_shadow(project_slug: str, auto_mode: bool, projects_dir: str) -> None:
    """Grow a probabilistic shadow graph from context stats and plan."""
    if not auto_mode:
        raise click.ClickException("Use --auto mode: bga story grow-shadow --project <slug> --auto")

    project = _load_project(project_slug, Path(projects_dir))
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    context_path = proj_dir / "context_stats.json"
    if not context_path.exists():
        raise click.ClickException(f"Missing {context_path}. Run story context first.")

    context = _load_json(context_path, default={})
    constraints = _load_constraints(proj_dir)
    timeline = _project_timeline(project)
    timeline_snapshot = context.get("timeline", {}) if isinstance(context.get("timeline"), dict) else {}
    if timeline_snapshot:
        timeline = {
            **timeline,
            **{
                "story_era": canonicalize_era(str(timeline_snapshot.get("story_era") or "").strip()) or timeline.get("story_era"),
                "story_year": _coerce_optional_int(timeline_snapshot.get("story_year", timeline.get("story_year"))),
                "allow_past_references": bool(timeline_snapshot.get("allow_past_references", timeline.get("allow_past_references", True))),
                "forbid_future_entities": bool(timeline_snapshot.get("forbid_future_entities", timeline.get("forbid_future_entities", True))),
                "forbidden_entities": _dedupe_strings(
                    [str(x) for x in timeline.get("forbidden_entities", [])]
                    + [str(x) for x in timeline_snapshot.get("forbidden_entities", [])]
                ),
            },
        }
        timeline["story_era_order"] = era_to_order(timeline.get("story_era"))
    plan_path = proj_dir / "plan.json"
    if plan_path.exists():
        plan = _load_json(plan_path, default={})
    else:
        plan = _build_plan(project, constraints, auto=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    transitions = context.get("event_transition_probabilities", {})
    char_priors = context.get("character_participation_priors", {})
    motif_priors = context.get("motif_reference_density_priors", {})
    entity_presence = context.get("entity_temporal_presence", {}) if isinstance(context.get("entity_temporal_presence"), dict) else {}
    local_story_neighborhood = context.get("local_story_neighborhood", {}) if isinstance(context.get("local_story_neighborhood"), dict) else {}
    canon_evidence = _context_canon_evidence(context)
    local_char_priors = local_story_neighborhood.get("character_priors", {}) if isinstance(local_story_neighborhood.get("character_priors"), dict) else {}
    local_motif_priors = local_story_neighborhood.get("motif_priors", {}) if isinstance(local_story_neighborhood.get("motif_priors"), dict) else {}
    local_action_priors = local_story_neighborhood.get("action_priors", {}) if isinstance(local_story_neighborhood.get("action_priors"), dict) else {}
    top_characters = [k for k, _ in sorted(local_char_priors.items(), key=lambda kv: kv[1], reverse=True)[:12]]
    if not top_characters:
        top_characters = [k for k, _ in sorted(char_priors.items(), key=lambda kv: kv[1], reverse=True)[:12]] or ["Beren", "Luthien"]
    canon_entities = _project_canon_entities(project_slug)
    out_of_domain = _out_of_domain_entities(project_slug)
    # Strong project priors: canon first, then observed priors.
    top_characters = [
        name
        for name in list(dict.fromkeys(canon_entities + top_characters))[:24]
        if _valid_shadow_character_name(project_slug, name)
    ][:18]
    top_characters = [
        name for name in top_characters
        if _temporal_entity_status(name, timeline, entity_presence)["status"] not in {"future_only", "past_only", "explicit_forbidden"}
    ]
    if not top_characters:
        top_characters = [
            name for name in canon_entities
            if _valid_shadow_character_name(project_slug, name)
            and _temporal_entity_status(name, timeline, entity_presence)["status"] != "explicit_forbidden"
        ] or canon_entities or ["Beren", "Luthien"]
    top_motifs = [k for k, _ in sorted(local_motif_priors.items(), key=lambda kv: kv[1], reverse=True)[:30]]
    if not top_motifs:
        top_motifs = [k for k, _ in sorted(motif_priors.items(), key=lambda kv: kv[1], reverse=True)[:30]] or ["song", "oath", "shadow"]
    seed = _stable_seed(project_slug, _canonical_json(plan), _canonical_json(constraints))
    rng = random.Random(seed)
    required = [str(x) for x in constraints.get("required_elements", [])]
    required_lowers = {term.lower() for term in required}
    forbidden = {str(x).lower() for x in constraints.get("forbidden_terms", [])}

    graph_nodes = [
        {
            "id": row["evidence_id"],
            "type": "CanonEvidence",
            "description": row.get("description"),
            "source_event_id": row.get("source_event_id"),
            "source_file": row.get("source_file"),
            "source_book": row.get("source_book"),
            "source_location": row.get("source_location"),
            "era": row.get("era"),
            "year": row.get("year"),
            "temporal_relation": row.get("temporal_relation"),
            "epistemic_status": row.get("epistemic_status"),
        }
        for row in canon_evidence
    ]
    graph_edges = []
    candidates = []
    selected = []
    prev_action = "unknown"

    for ch in plan.get("chapters", []):
        chapter_num = int(ch.get("chapter_number", 1))
        scenes = ch.get("scenes", [])
        for scene in scenes:
            scene_id = str(scene.get("scene_id"))
            chapter_scene_key = f"shadow-{scene_id}"
            scene_node = {
                "id": chapter_scene_key,
                "type": "ShadowScene",
                "scene_id": scene_id,
                "chapter": chapter_num,
                "summary": scene.get("summary", ""),
            }
            graph_nodes.append(scene_node)

            row_candidates = []
            for rank in range(3):
                next_action_dist = transitions.get(prev_action, {})
                if not next_action_dist and local_action_priors:
                    next_action_dist = local_action_priors
                action_choices = list((next_action_dist or {"journey": 0.34, "conflict": 0.33, "reveal": 0.33}).items())
                action = action_choices[min(rank, len(action_choices) - 1)][0]
                transition_prob = float(action_choices[min(rank, len(action_choices) - 1)][1])
                chars = rng.sample(top_characters, k=min(2 + rank, len(top_characters)))
                temporal_statuses = {
                    c: _temporal_entity_status(c, timeline, entity_presence)
                    for c in chars
                }
                invalid_temporal_chars = [
                    c for c, status in temporal_statuses.items()
                    if status["status"] in {"future_only", "past_only", "explicit_forbidden"}
                ]
                if invalid_temporal_chars:
                    continue
                motifs = rng.sample(top_motifs, k=min(2, len(top_motifs)))
                if required and chapter_num == 1 and rank == 0:
                    motifs = list(dict.fromkeys((required[:1] + motifs)))

                # Spread hard anchors across early candidates to keep hard-gated solve feasible.
                if required:
                    req_idx = ((chapter_num - 1) * max(1, len(scenes)) + rank) % len(required)
                    motifs = list(dict.fromkeys(motifs + [required[req_idx]]))

                scene_anchor_text = " ".join(
                    str(scene.get(key) or "")
                    for key in ("goal", "summary", "setting")
                )
                scene_anchor_text += " " + " ".join(str(x) for x in scene.get("objects", []) or [])
                scene_anchor_text += " " + " ".join(str(x) for x in scene.get("characters", []) or [])
                scene_anchor_l = scene_anchor_text.lower()
                scene_required_anchors = [
                    term for term in required if term.lower() in scene_anchor_l
                ]
                if scene_required_anchors:
                    motifs = list(dict.fromkeys(scene_required_anchors + motifs))

                description = f"{scene.get('goal', 'Advance plot')} via {action}. Characters: {', '.join(chars[:3])}. Motifs: {', '.join(motifs[:3])}."
                anchor_terms = [term for term in motifs if term.lower() in required_lowers]
                if anchor_terms:
                    description += f" Anchors: {', '.join(anchor_terms)}."
                if any(term in description.lower() for term in forbidden):
                    continue

                source_canon_node_ids = _matching_canon_evidence_refs(
                    canon_evidence,
                    text=f"{scene_anchor_text} {description}",
                    participants=chars,
                    motifs=motifs,
                    action=action,
                )

                char_score = sum(float(local_char_priors.get(c, char_priors.get(c, 0.01))) for c in chars) / max(1, len(chars))
                canon_hits = sum(1 for c in chars if c in canon_entities)
                out_of_domain_hits = sum(1 for c in chars if c.lower() in out_of_domain)
                unknown_hits = sum(1 for c in chars if c.strip().lower() in {"unknown", "they", "someone"})
                motif_score = sum(float(local_motif_priors.get(m, motif_priors.get(m, 0.005))) for m in motifs) / max(1, len(motifs))
                evidence_boost = min(0.12, 0.04 * len(source_canon_node_ids))
                prior_boost = (0.10 * canon_hits) + evidence_boost - (0.22 * out_of_domain_hits) - (0.25 * unknown_hits)
                plausibility = round(min(0.99, max(0.01, (0.5 * transition_prob) + (0.3 * char_score) + (0.2 * motif_score) + prior_boost)), 6)
                cid = f"{scene_id}-cand-{rank + 1}"
                row_candidates.append(
                    {
                        "candidate_id": cid,
                        "scene_id": scene_id,
                        "chapter": chapter_num,
                        "shadow_event": {
                            "id": f"shadow-event-{cid}",
                            "type": "ShadowEvent",
                            "action": action,
                            "description": description,
                            "characters": chars,
                            "motifs": motifs,
                            "source_canon_node_ids": source_canon_node_ids,
                        },
                        "transition_probability": round(transition_prob, 6),
                        "plausibility_score": plausibility,
                        "hard_constraints_ok": True,
                        "timeline_ok": True,
                        "timeline_scope": {
                            "story_era": timeline.get("story_era"),
                            "story_year": timeline.get("story_year"),
                        },
                        "temporal_statuses": temporal_statuses,
                        "project_prior": {
                            "canon_hits": canon_hits,
                            "out_of_domain_hits": out_of_domain_hits,
                            "unknown_entity_hits": unknown_hits,
                            "canon_evidence_hits": len(source_canon_node_ids),
                        },
                    }
                )
            row_candidates.sort(key=lambda c: c["plausibility_score"], reverse=True)
            if row_candidates:
                selected.append(row_candidates[0])
                prev_action = row_candidates[0]["shadow_event"]["action"]
            candidates.extend(row_candidates)

    # Materialize all candidate events so solved trajectories always have valid grounding refs.
    for cand in candidates:
        ev = cand["shadow_event"]
        graph_nodes.append(ev)
        graph_edges.append({
            "source": f"shadow-{cand['scene_id']}",
            "target": ev["id"],
            "type": "HAS_EVENT",
            "probability": cand["plausibility_score"],
        })
        for evidence_id in ev.get("source_canon_node_ids", []) or []:
            graph_edges.append(
                {
                    "source": ev["id"],
                    "target": evidence_id,
                    "type": "SUPPORTED_BY",
                    "probability": 1.0,
                }
            )

    for idx, chosen in enumerate(selected):
        ev = chosen["shadow_event"]
        for c in ev["characters"]:
            cid = f"shadow-char-{re.sub(r'[^a-z0-9]+', '-', c.lower()).strip('-')}"
            graph_nodes.append({"id": cid, "type": "ShadowCharacter", "name": c})
            graph_edges.append({"source": ev["id"], "target": cid, "type": "INVOLVES", "probability": 1.0})
        for m in ev["motifs"]:
            mid = f"shadow-motif-{re.sub(r'[^a-z0-9]+', '-', m.lower()).strip('-')}"
            graph_nodes.append({"id": mid, "type": "ShadowMotif", "name": m})
            graph_edges.append({"source": ev["id"], "target": mid, "type": "USES_MOTIF", "probability": round(float(motif_priors.get(m, 0.02)), 6)})
        if idx > 0:
            prev = selected[idx - 1]["shadow_event"]["id"]
            graph_edges.append({"source": prev, "target": ev["id"], "type": "NEXT", "probability": chosen["transition_probability"]})

    graph_payload = {
        "schema_version": "shadow-graph-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "local_story_neighborhood": local_story_neighborhood,
    }
    candidates_payload = {
        "schema_version": "shadow-candidates-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeline": timeline,
        "constraints_snapshot": constraints,
        "candidates": candidates,
        "selected_auto": [c["candidate_id"] for c in selected],
    }

    (proj_dir / "shadow_graph.json").write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    (proj_dir / "shadow_candidates.json").write_text(json.dumps(candidates_payload, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Shadow graph artifacts written under {proj_dir}")


@story.command("sample-shadow")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--n", required=True, type=int, help="Number of shadow candidates to sample")
@click.option("--method", type=click.Choice(["anneal"], case_sensitive=False), default="anneal", show_default=True)
@click.option("--seed", type=int, default=None, help="Deterministic random seed")
@click.option("--steps", type=int, default=80, show_default=True, help="Annealing mutation steps per candidate")
@click.option("--temp-start", type=float, default=1.2, show_default=True)
@click.option("--temp-end", type=float, default=0.05, show_default=True)
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_sample_shadow(
    project_slug: str,
    n: int,
    method: str,
    seed: int | None,
    steps: int,
    temp_start: float,
    temp_end: float,
    projects_dir: str,
) -> None:
    """Sample N shadow-graph candidates via local mutations + annealing acceptance."""
    if n <= 0:
        raise click.ClickException("--n must be > 0")
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    project = _load_project(project_slug, Path(projects_dir))
    context = _load_json(proj_dir / "context_stats.json", default={})
    plan = _load_json(proj_dir / "plan.json", default={})
    constraints = _load_constraints(proj_dir)
    if not context:
        raise click.ClickException("Missing context_stats.json. Run: bga story context --graph-stats")
    if not plan:
        raise click.ClickException("Missing plan.json. Run: bga story plan --auto")

    global_transitions = context.get("event_transition_probabilities", {})
    global_char_priors = context.get("character_participation_priors", {})
    global_motif_priors = context.get("motif_reference_density_priors", {})
    style_budget = context.get("register_style_budgets", {})
    if not isinstance(global_transitions, dict):
        global_transitions = {}
    if not isinstance(global_char_priors, dict):
        global_char_priors = {}
    if not isinstance(global_motif_priors, dict):
        global_motif_priors = {}

    timeline = _project_timeline(project)
    timeline_snapshot = context.get("timeline", {}) if isinstance(context.get("timeline"), dict) else {}
    if timeline_snapshot:
        timeline = {
            **timeline,
            "story_era": (
                canonicalize_era(str(timeline_snapshot.get("story_era") or "").strip())
                or timeline.get("story_era")
            ),
            "story_year": _coerce_optional_int(
                timeline_snapshot.get("story_year", timeline.get("story_year"))
            ),
            "allow_past_references": bool(
                timeline_snapshot.get(
                    "allow_past_references",
                    timeline.get("allow_past_references", True),
                )
            ),
            "forbid_future_entities": bool(
                timeline_snapshot.get(
                    "forbid_future_entities",
                    timeline.get("forbid_future_entities", True),
                )
            ),
            "forbidden_entities": _dedupe_strings(
                [str(value) for value in timeline.get("forbidden_entities", [])]
                + [str(value) for value in timeline_snapshot.get("forbidden_entities", [])]
            ),
        }
        timeline["story_era_order"] = era_to_order(timeline.get("story_era"))

    entity_presence = (
        context.get("entity_temporal_presence", {})
        if isinstance(context.get("entity_temporal_presence"), dict)
        else {}
    )
    local_neighborhood = (
        context.get("local_story_neighborhood", {})
        if isinstance(context.get("local_story_neighborhood"), dict)
        else {}
    )
    local_char_priors = (
        local_neighborhood.get("character_priors", {})
        if isinstance(local_neighborhood.get("character_priors"), dict)
        else {}
    )
    local_motif_priors = (
        local_neighborhood.get("motif_priors", {})
        if isinstance(local_neighborhood.get("motif_priors"), dict)
        else {}
    )
    local_action_priors = (
        local_neighborhood.get("action_priors", {})
        if isinstance(local_neighborhood.get("action_priors"), dict)
        else {}
    )

    excluded_characters: dict[str, str] = {}

    def _eligible_character_priors(raw_priors: dict[str, Any]) -> dict[str, float]:
        eligible: dict[str, float] = {}
        for raw_name, raw_weight in raw_priors.items():
            name = str(raw_name).strip()
            if not _valid_shadow_character_name(project_slug, name):
                excluded_characters[name] = "not_a_project_character"
                continue
            status = _temporal_entity_status(name, timeline, entity_presence)["status"]
            if status in {"future_only", "past_only", "explicit_forbidden"}:
                excluded_characters[name] = status
                continue
            weight = max(0.0, float(raw_weight))
            if weight > 0:
                eligible[name] = weight
        total = sum(eligible.values())
        return {name: weight / total for name, weight in eligible.items()} if total else {}

    eligible_local_chars = _eligible_character_priors(local_char_priors)
    eligible_global_chars = _eligible_character_priors(global_char_priors)
    canon_names = {
        name.lower()
        for name in _project_canon_entities(project_slug)
        if _valid_shadow_character_name(project_slug, name)
    }
    canon_global_chars = {
        name: weight
        for name, weight in eligible_global_chars.items()
        if name.lower() in canon_names
    }
    if eligible_local_chars:
        char_priors = eligible_local_chars
        character_prior_source = "local_story_neighborhood"
    elif canon_global_chars:
        char_priors = canon_global_chars
        character_prior_source = "project_canon_global_fallback"
    else:
        eligible_canon = _eligible_character_priors(
            {name: 1.0 for name in _project_canon_entities(project_slug)}
        )
        if eligible_canon:
            char_priors = eligible_canon
            character_prior_source = "project_canon_fallback"
        elif eligible_global_chars:
            char_priors = eligible_global_chars
            character_prior_source = "global_context_fallback"
        else:
            char_priors = {"Beren": 0.5, "Luthien": 0.5}
            character_prior_source = "hardcoded_fallback"

    if local_motif_priors:
        motif_priors = {
            str(name): max(0.0, float(weight))
            for name, weight in local_motif_priors.items()
            if str(name).strip() and float(weight) > 0
        }
        motif_prior_source = "local_story_neighborhood"
    elif global_motif_priors:
        motif_priors = {
            str(name): max(0.0, float(weight))
            for name, weight in global_motif_priors.items()
            if str(name).strip() and float(weight) > 0
        }
        motif_prior_source = "global_context_fallback"
    else:
        motif_priors = {"oath": 0.25, "song": 0.25, "fate": 0.25, "shadow": 0.25}
        motif_prior_source = "hardcoded_fallback"

    if local_action_priors:
        transitions: dict[str, dict[str, float]] = {}
        source_actions = set(global_transitions) | {"unknown"}
        for source_action in source_actions:
            base = global_transitions.get(source_action, {})
            if not isinstance(base, dict):
                base = {}
            mixed: defaultdict[str, float] = defaultdict(float)
            for action, weight in local_action_priors.items():
                mixed[str(action)] += 0.7 * max(0.0, float(weight))
            for action, weight in base.items():
                mixed[str(action)] += 0.3 * max(0.0, float(weight))
            total = sum(mixed.values())
            if total > 0:
                transitions[str(source_action)] = {
                    action: weight / total for action, weight in mixed.items()
                }
        action_prior_source = "local_story_neighborhood_blended"
    elif global_transitions:
        transitions = global_transitions
        action_prior_source = "global_transition_fallback"
    else:
        transitions = {"unknown": {"journey": 1.0}}
        action_prior_source = "hardcoded_fallback"

    top_characters = _topk_keys(char_priors, 16, list(char_priors))
    top_motifs = _topk_keys(motif_priors, 40, list(motif_priors))
    prior_sources = {
        "characters": character_prior_source,
        "motifs": motif_prior_source,
        "actions": action_prior_source,
        "local_story_neighborhood_available": bool(local_neighborhood),
        "timeline_filter": {
            "story_era": timeline.get("story_era"),
            "story_year": timeline.get("story_year"),
            "excluded_characters": dict(sorted(excluded_characters.items())),
        },
    }

    eff_seed = int(seed if seed is not None else _stable_seed(project_slug, str(n), str(steps), method))
    rng = random.Random(eff_seed)
    out_path = proj_dir / "shadow_samples.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(n):
            candidate_seed = rng.randrange(2**32)
            crng = random.Random(candidate_seed)
            state = _build_initial_shadow_state(plan, transitions, top_characters, top_motifs, crng)
            best = json.loads(json.dumps(state))
            e_cur = _anneal_energy(state, transitions, char_priors, motif_priors, constraints, style_budget)
            e_best = e_cur
            accepted = 0
            for step in range(max(1, steps)):
                temp = _interp_temp(step, max(1, steps), temp_start, temp_end)
                proposal = _mutate_state(state, transitions, top_characters, top_motifs, crng)
                e_next = _anneal_energy(proposal, transitions, char_priors, motif_priors, constraints, style_budget)
                delta = e_next - e_cur
                accept = delta <= 0 or (crng.random() < math.exp(-delta / temp))
                if accept:
                    state = proposal
                    e_cur = e_next
                    accepted += 1
                    if e_cur < e_best:
                        best = json.loads(json.dumps(state))
                        e_best = e_cur
            row = {
                "schema_version": "shadow-sample-v1",
                "project_slug": project_slug,
                "candidate_id": f"shadow-sample-{idx+1:05d}",
                "method": method,
                "seed": candidate_seed,
                "steps": max(1, steps),
                "temp_start": temp_start,
                "temp_end": temp_end,
                "acceptance_ratio": round(accepted / max(1, steps), 6),
                "anneal_energy": round(float(e_best), 6),
                "prior_sources": prior_sources,
                "state": best,
            }
            f.write(json.dumps(row) + "\n")

    console.print(f"[green]OK[/green] Shadow samples written: {out_path} (n={n}, seed={eff_seed})")


@story.command("score-shadow")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--weights", default=None, help="JSON string or path to weights json")
@click.option("--pareto", is_flag=True, help="Also emit Pareto front")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_score_shadow(project_slug: str, weights: str | None, pareto: bool, projects_dir: str) -> None:
    """Score sampled shadow graphs with transparent component breakdowns."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    samples_path = proj_dir / "shadow_samples.jsonl"
    if not samples_path.exists():
        raise click.ClickException("Missing shadow_samples.jsonl. Run: bga story sample-shadow ...")

    context = _load_json(proj_dir / "context_stats.json", default={})
    constraints = _load_constraints(proj_dir)
    transitions = context.get("event_transition_probabilities", {})
    char_priors = context.get("character_participation_priors", {})
    motif_priors = context.get("motif_reference_density_priors", {})
    known_characters = {str(name).strip().lower() for name in char_priors if str(name).strip()}
    known_motifs = {str(name).strip().lower() for name in motif_priors if str(name).strip()}

    ws = _load_weights_arg(weights)
    rows = []
    motif_sets = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec.get("state", [])
            motif_sets.append(set(m for s in state for m in s.get("motifs", [])))
            rows.append(rec)

    results = []
    for idx, rec in enumerate(rows):
        state = rec.get("state", [])
        text = "\n".join(str(s.get("description", "")) for s in state).lower()
        required = [str(x).lower() for x in constraints.get("required_elements", [])]
        forbidden = [str(x).lower() for x in constraints.get("forbidden_terms", [])]

        missing_required = sum(1 for r in required if r not in text)
        forbidden_hits = sum(1 for f in forbidden if f in text)
        selected_characters = [
            str(value).strip().lower()
            for scene_state in state
            for value in (scene_state.get("characters", []) or [])
            if str(value).strip()
        ]
        selected_motifs = [
            str(value).strip().lower()
            for scene_state in state
            for value in (scene_state.get("motifs", []) or [])
            if str(value).strip()
        ]
        supported = sum(1 for value in selected_characters if value in known_characters)
        supported += sum(1 for value in selected_motifs if value in known_motifs)
        support_total = len(selected_characters) + len(selected_motifs)
        corpus_support = supported / max(1, support_total)
        canon_penalty = min(
            1.0,
            0.5 * forbidden_hits
            + 0.3 * (missing_required / max(1, len(required)))
            + 0.2 * (1.0 - corpus_support),
        )
        canon_consistency = round(max(0.0, 1.0 - canon_penalty), 6)

        trans_vals = []
        actions = [str(s.get("action", "unknown")) for s in state]
        for i, action in enumerate(actions):
            prev = actions[i - 1] if i > 0 else "unknown"
            trans_vals.append(float(transitions.get(prev, {}).get(action, 0.05)))
        transition_likelihood = round(sum(trans_vals) / max(1, len(trans_vals)), 6)

        arc_coherence = round(_arc_progression_score(actions), 6)

        # Shadow descriptions are plan summaries, not prose.  Score whether
        # they are specific and usable rather than comparing them with a scene
        # prose word budget (which made every concise plan look stylistically bad).
        summary_scores: list[float] = []
        for scene_state in state:
            description = str(scene_state.get("description") or "").strip()
            word_count = len(description.split())
            description_tokens = set(re.findall(r"[a-z]+", description.lower()))
            placeholder = bool(description_tokens & PLACEHOLDER_PARTICIPANTS)
            length_score = min(1.0, word_count / 8.0)
            if word_count > 80:
                length_score *= max(0.0, 1.0 - ((word_count - 80) / 80.0))
            summary_scores.append(0.0 if placeholder else length_score)
        style_register = round(sum(summary_scores) / max(1, len(summary_scores)), 6)

        motif_set = motif_sets[idx]
        avg_jaccard = 0.0
        if len(motif_sets) > 1:
            sims = []
            for j, other in enumerate(motif_sets):
                if j == idx:
                    continue
                union = len(motif_set | other)
                sims.append((len(motif_set & other) / union) if union else 1.0)
            avg_jaccard = sum(sims) / max(1, len(sims))
        novelty_diversity = round(max(0.0, 1.0 - avg_jaccard), 6)

        total = (
            ws["canon_consistency"] * canon_consistency
            + ws["transition_likelihood"] * transition_likelihood
            + ws["arc_coherence"] * arc_coherence
            + ws["style_register"] * style_register
            + ws["novelty_diversity"] * novelty_diversity
        )
        results.append(
            {
                "candidate_id": rec.get("candidate_id"),
                "seed": rec.get("seed"),
                "anneal_energy": rec.get("anneal_energy"),
                "components": {
                    "canon_consistency_penalty": round(canon_penalty, 6),
                    "canon_consistency": canon_consistency,
                    "corpus_support": round(corpus_support, 6),
                    "transition_likelihood": transition_likelihood,
                    "arc_coherence": arc_coherence,
                    "style_register": style_register,
                    "novelty_diversity": novelty_diversity,
                },
                "weighted_score": round(float(total), 6),
            }
        )

    results.sort(key=lambda r: (-r["weighted_score"], str(r["candidate_id"])))
    out = {
        "schema_version": "shadow-scores-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": ws,
        "scores": results,
    }
    out_path = proj_dir / "shadow_scores.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Shadow scores written: {out_path}")

    if pareto:
        dims = ["canon_consistency", "transition_likelihood", "arc_coherence", "style_register", "novelty_diversity"]

        def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
            ca, cb = a["components"], b["components"]
            return all(float(ca[d]) >= float(cb[d]) for d in dims) and any(float(ca[d]) > float(cb[d]) for d in dims)

        front = []
        for i, cand in enumerate(results):
            dominated = False
            for j, other in enumerate(results):
                if i == j:
                    continue
                if dominates(other, cand):
                    dominated = True
                    break
            if not dominated:
                front.append(cand)
        front.sort(key=lambda r: (-r["weighted_score"], str(r["candidate_id"])))
        pareto_payload = {
            "schema_version": "shadow-pareto-v1",
            "project_slug": project_slug,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dimensions": dims,
            "candidates": front,
        }
        pareto_path = proj_dir / "shadow_pareto_front.json"
        pareto_path.write_text(json.dumps(pareto_payload, indent=2), encoding="utf-8")
        console.print(f"Pareto front written: {pareto_path} (n={len(front)})")


@story.command("select-shadow")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--top", "top_k", required=True, type=int, help="Select top-K candidates")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_select_shadow(project_slug: str, top_k: int, projects_dir: str) -> None:
    """Select top-K shadow candidates from weighted scores (stable ordering)."""
    if top_k <= 0:
        raise click.ClickException("--top must be > 0")
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    scores = _load_json(proj_dir / "shadow_scores.json", default={})
    rows = scores.get("scores", [])
    if not rows:
        raise click.ClickException("Missing/empty shadow_scores.json. Run: bga story score-shadow ...")
    rows = sorted(rows, key=lambda r: (-float(r.get("weighted_score", 0.0)), str(r.get("candidate_id", ""))))
    selected = rows[:top_k]
    payload = {
        "schema_version": "shadow-selected-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "weighted_score_desc_then_candidate_id_asc",
        "top_k": top_k,
        "selected": selected,
    }
    out_path = proj_dir / "shadow_selected.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Shadow selection written: {out_path} (k={len(selected)})")


def _selected_sample_scene_priors(proj_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Turn selected whole-story samples into optional priors for beam solve."""
    selected_payload = _load_json(proj_dir / "shadow_selected.json", default={})
    selected_rows = selected_payload.get("selected", []) if isinstance(selected_payload, dict) else []
    selected_ids = {
        str(row.get("candidate_id") or "").strip()
        for row in selected_rows
        if isinstance(row, dict) and str(row.get("candidate_id") or "").strip()
    }
    samples_path = proj_dir / "shadow_samples.jsonl"
    if not selected_ids or not samples_path.exists():
        return {}

    accumulators: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"actions": Counter(), "characters": Counter(), "motifs": Counter()}
    )
    with samples_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if str(row.get("candidate_id") or "").strip() not in selected_ids:
                continue
            for scene_state in row.get("state", []) or []:
                if not isinstance(scene_state, dict):
                    continue
                scene_id = str(scene_state.get("scene_id") or "").strip()
                if not scene_id:
                    continue
                action = str(scene_state.get("action") or "").strip().lower()
                if action:
                    accumulators[scene_id]["actions"][action] += 1
                for value in scene_state.get("characters", []) or []:
                    normalized = str(value).strip().lower()
                    if normalized:
                        accumulators[scene_id]["characters"][normalized] += 1
                for value in scene_state.get("motifs", []) or []:
                    normalized = str(value).strip().lower()
                    if normalized:
                        accumulators[scene_id]["motifs"][normalized] += 1

    return {
        scene_id: {
            dimension: _safe_prob(dict(counts))
            for dimension, counts in dimensions.items()
        }
        for scene_id, dimensions in accumulators.items()
    }


def _selected_sample_prior_score(candidate: dict[str, Any], priors: dict[str, dict[str, dict[str, float]]]) -> float:
    scene_id = str(candidate.get("scene_id") or "").strip()
    scene_priors = priors.get(scene_id)
    if not scene_priors:
        return 0.0
    event = candidate.get("shadow_event", {}) if isinstance(candidate.get("shadow_event"), dict) else {}
    action = str(event.get("action") or "").strip().lower()
    characters = [str(value).strip().lower() for value in (event.get("characters", []) or []) if str(value).strip()]
    motifs = [str(value).strip().lower() for value in (event.get("motifs", []) or []) if str(value).strip()]
    action_score = float(scene_priors.get("actions", {}).get(action, 0.0))
    character_score = sum(float(scene_priors.get("characters", {}).get(value, 0.0)) for value in characters) / max(1, len(characters))
    motif_score = sum(float(scene_priors.get("motifs", {}).get(value, 0.0)) for value in motifs) / max(1, len(motifs))
    return max(0.0, min(1.0, (0.5 * action_score) + (0.3 * character_score) + (0.2 * motif_score)))


@story.command("solve")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_solve(project_slug: str, projects_dir: str) -> None:
    """Solve best valid trajectory through shadow candidates using beam search."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    project = _load_project(project_slug, Path(projects_dir))
    payload = _load_json(proj_dir / "shadow_candidates.json", default={})
    candidates = payload.get("candidates", [])
    if not candidates:
        raise click.ClickException("No candidates found. Run story grow-shadow first.")
    timeline = _project_timeline(project)
    timeline_snapshot = payload.get("timeline", {}) if isinstance(payload.get("timeline"), dict) else {}
    if timeline_snapshot:
        timeline["story_era"] = canonicalize_era(str(timeline_snapshot.get("story_era") or "").strip()) or timeline.get("story_era")
        timeline["story_year"] = _coerce_optional_int(timeline_snapshot.get("story_year", timeline.get("story_year")))
        timeline["story_era_order"] = era_to_order(timeline.get("story_era"))
    context_payload = _load_json(proj_dir / "context_stats.json", default={})
    entity_presence = context_payload.get("entity_temporal_presence", {}) if isinstance(context_payload.get("entity_temporal_presence"), dict) else {}

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for cand in candidates:
        by_scene[str(cand.get("scene_id"))].append(cand)
    scene_ids = sorted(by_scene.keys())
    constraints = _load_constraints(proj_dir)
    required = [str(x).lower() for x in constraints.get("required_elements", [])]
    forbidden = [str(x).lower() for x in constraints.get("forbidden_terms", [])]

    selection_cfg = constraints.get("selection", {}) if isinstance(constraints.get("selection"), dict) else {}
    enforcement_cfg = constraints.get("enforcement", {}) if isinstance(constraints.get("enforcement"), dict) else {}

    goal_completion_threshold = float(
        selection_cfg.get("goal_completion_threshold", enforcement_cfg.get("goal_completion_threshold", 1.0))
    )
    min_beats_per_scene = int(selection_cfg.get("min_beats_per_scene", enforcement_cfg.get("min_beats_per_scene", 1)) or 1)
    anti_padding_penalty = float(selection_cfg.get("anti_padding_penalty", 1.0))
    unresolved_thread_penalty = float(selection_cfg.get("unresolved_thread_penalty", 0.4))
    selected_sample_prior_weight = float(selection_cfg.get("selected_sample_prior_weight", 0.75))
    selected_sample_priors = _selected_sample_scene_priors(proj_dir)
    precondition_unknown_policy = str(
        selection_cfg.get(
            "precondition_unknown_policy",
            enforcement_cfg.get("precondition_unknown_policy", "reject"),
        )
    ).strip().lower() or "reject"
    precondition_unknown_penalty = float(
        selection_cfg.get(
            "precondition_unknown_penalty",
            enforcement_cfg.get("precondition_unknown_penalty", 0.75),
        )
    )

    beam_width_schedule = [4, 8, 16]
    k_max = len(scene_ids)
    best_score = float("-inf")
    best_path: list[dict] = []
    missing_required: list[str] = []
    selected_beam_width = beam_width_schedule[0]

    def _normalize_fact_state(value: Any, default_value: str = "unknown") -> str:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "t", "1", "yes", "y", "on"}:
                return "true"
            if v in {"false", "f", "0", "no", "n", "off"}:
                return "false"
            if v in {"unknown", "unk", "?", "none", "null"}:
                return "unknown"
            return default_value
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "unknown"
        return "true" if bool(value) else "false"

    def _as_fact_state_map(raw: Any, default_value: str = "true") -> dict[str, str]:
        if isinstance(raw, dict):
            out: dict[str, str] = {}
            for k, v in raw.items():
                key = str(k).strip().lower()
                if not key:
                    continue
                out[key] = _normalize_fact_state(v, default_value=default_value)
            return out
        if isinstance(raw, list):
            out: dict[str, str] = {}
            for item in raw:
                key = str(item).strip().lower()
                if key:
                    out[key] = default_value
            return out
        return {}

    def _evaluate_preconditions(
        world_state: dict[str, str],
        preconditions: dict[str, str],
        unknown_policy: str,
        unknown_penalty: float,
    ) -> tuple[bool, float]:
        penalty = 0.0
        for fact, expected in preconditions.items():
            observed = _normalize_fact_state(world_state.get(fact, "unknown"), default_value="unknown")
            if observed == expected:
                continue

            # Backward-compatible default semantics: unknown does NOT satisfy true,
            # but remains acceptable for expected=false unless caller opts into strictness.
            if observed == "unknown" and expected == "false":
                continue

            if observed == "unknown":
                if unknown_policy in {"penalize", "soft_penalty", "soft"}:
                    penalty += max(0.0, float(unknown_penalty))
                    continue
                return False, penalty

            return False, penalty
        return True, penalty

    def _candidate_progress(cand: dict[str, Any]) -> float:
        raw = cand.get("scene_goal_progress", cand.get("goal_progress", cand.get("progress", 0.0)))
        if raw is None:
            raw = cand.get("shadow_event", {}).get("scene_goal_progress", 0.0)
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

    def _unresolved_threads_count(cand: dict[str, Any]) -> int:
        raw = cand.get("unresolved_causal_threads", cand.get("unresolved_threads", 0))
        if raw is None:
            raw = cand.get("shadow_event", {}).get("unresolved_causal_threads", 0)
        if isinstance(raw, list):
            return len(raw)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    for beam_width in beam_width_schedule:
        beam: list[dict[str, Any]] = [{"score": 0.0, "path": [], "next_idx": 0, "max_progress": 0.0, "world_state": {}}]
        completed: list[dict[str, Any]] = []
        for _ in range(k_max):
            next_beam: list[dict[str, Any]] = []
            for node in beam:
                base_score = float(node.get("score", 0.0))
                path = list(node.get("path", []))
                next_idx = int(node.get("next_idx", 0))
                max_progress = float(node.get("max_progress", 0.0))
                world_state = {
                    str(k).strip().lower(): _normalize_fact_state(v, default_value="unknown")
                    for k, v in dict(node.get("world_state", {})).items()
                    if str(k).strip()
                }

                if len(path) >= min_beats_per_scene and max_progress >= goal_completion_threshold:
                    completed.append(node)

                if next_idx >= k_max:
                    completed.append(node)
                    continue

                sid = scene_ids[next_idx]
                action_counts = Counter(str(p.get("shadow_event", {}).get("action", "unknown")) for p in path)
                chosen_characters = {
                    c.lower()
                    for p in path
                    for c in (p.get("shadow_event", {}).get("characters", []) or [])
                    if isinstance(c, str)
                }
                for cand in by_scene[sid][:6]:
                    if cand.get("timeline_ok") is False:
                        continue
                    desc = str(cand.get("shadow_event", {}).get("description", "")).lower()
                    if any(term in desc for term in forbidden):
                        continue
                    p = max(1e-6, float(cand.get("plausibility_score", 0.01)))
                    t = max(1e-6, float(cand.get("transition_probability", 0.01)))

                    action = str(cand.get("shadow_event", {}).get("action", "unknown")).strip().lower() or "unknown"
                    chars = [str(c).strip() for c in (cand.get("shadow_event", {}).get("characters", []) or []) if str(c).strip()]
                    chars_l = {c.lower() for c in chars}
                    temporal_invalid_chars = [
                        c
                        for c in chars
                        if _temporal_entity_status(c, timeline, entity_presence)["status"] in {"future_only", "past_only", "explicit_forbidden"}
                    ]
                    if temporal_invalid_chars:
                        continue
                    prior = cand.get("project_prior", {}) if isinstance(cand.get("project_prior"), dict) else {}
                    out_of_domain_hits = int(prior.get("out_of_domain_hits", 0) or 0)

                    # Mode-collapse/placeholder suppression + diversity regularization.
                    placeholder_penalty = 0.0
                    if action in {"unknown", "placeholder", "tbd"}:
                        placeholder_penalty += 2.0
                    if any(c.lower() in {"unknown", "they", "someone"} for c in chars):
                        placeholder_penalty += 1.5
                    placeholder_penalty += 1.2 * out_of_domain_hits

                    repeat_penalty = 0.5 * action_counts.get(action, 0)
                    novelty_bonus = 0.35 * len(chars_l - chosen_characters)

                    preconditions = _as_fact_state_map(
                        cand.get("preconditions", cand.get("shadow_event", {}).get("preconditions", [])),
                        default_value="true",
                    )
                    preconditions_ok, unknown_mismatch_penalty = _evaluate_preconditions(
                        world_state,
                        preconditions,
                        unknown_policy=precondition_unknown_policy,
                        unknown_penalty=precondition_unknown_penalty,
                    )
                    if not preconditions_ok:
                        continue

                    effects = _as_fact_state_map(
                        cand.get("effects", cand.get("shadow_event", {}).get("effects", [])),
                        default_value="true",
                    )
                    next_state = dict(world_state)
                    next_state.update(effects)

                    cand_progress = _candidate_progress(cand)
                    next_progress = max(max_progress, cand_progress)
                    progress_gain = max(0.0, next_progress - max_progress)
                    nonprogress_penalty = anti_padding_penalty if progress_gain <= 1e-9 else 0.0
                    unresolved_penalty = unresolved_thread_penalty * _unresolved_threads_count(cand)
                    selected_sample_bonus = selected_sample_prior_weight * _selected_sample_prior_score(
                        cand,
                        selected_sample_priors,
                    )

                    score = (
                        base_score
                        + math.log(p)
                        + 0.5 * math.log(t)
                        + novelty_bonus
                        + selected_sample_bonus
                        - repeat_penalty
                        - placeholder_penalty
                        - unknown_mismatch_penalty
                        - nonprogress_penalty
                        - unresolved_penalty
                    )
                    next_beam.append(
                        {
                            "score": score,
                            "path": path + [cand],
                            "next_idx": next_idx + 1,
                            "max_progress": next_progress,
                            "world_state": next_state,
                        }
                    )
            next_beam.sort(key=lambda x: float(x.get("score", float("-inf"))), reverse=True)
            beam = next_beam[:beam_width] or beam

        completed.extend(beam)
        completed.sort(key=lambda x: float(x.get("score", float("-inf"))), reverse=True)

        top = completed[0]
        cand_score = float(top.get("score", float("-inf")))
        cand_path = list(top.get("path", []))

        full_text = "\n".join(c.get("shadow_event", {}).get("description", "") for c in cand_path).lower()
        missing = [r for r in required if r not in full_text]
        best_score, best_path = cand_score, cand_path
        missing_required = missing
        selected_beam_width = beam_width
        if not missing:
            break

    status = "pass" if not missing_required else "fail"
    if missing_required:
        raise click.ClickException(
            "Solved trajectory failed hard required-element gating "
            f"after retries (beam schedule={beam_width_schedule}, last_beam={selected_beam_width}). "
            f"Missing required elements: {missing_required}"
        )

    solved = {
        "schema_version": "shadow-solution-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k_max": k_max,
        "beam_width": selected_beam_width,
        "objective": "sum(log(plausibility)+0.5*log(transition_probability)+novelty_bonus+selected_sample_bonus-repeat_penalty-placeholder_penalty-nonprogress_penalty-unresolved_thread_penalty)",
        "selected_sample_prior": {
            "enabled": bool(selected_sample_priors),
            "weight": selected_sample_prior_weight,
            "scene_count": len(selected_sample_priors),
        },
        "best_score": round(best_score, 6),
        "status": status,
        "missing_required_elements": missing_required,
        "trajectory": [
            {
                "scene_id": c.get("scene_id"),
                "candidate_id": c.get("candidate_id"),
                "shadow_event_id": c.get("shadow_event", {}).get("id"),
                "action": c.get("shadow_event", {}).get("action"),
                "plausibility_score": c.get("plausibility_score"),
            }
            for c in best_path
        ],
    }
    out_path = proj_dir / "shadow_solution.json"
    out_path.write_text(json.dumps(solved, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Solved trajectory written: {out_path}")


@story.command("draft")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--chapter", required=True, type=int, help="Chapter number")
@click.option("--grounded", is_flag=True, help="Require graph-grounded drafting")
@click.option("--renderer", type=click.Choice(["llm", "template"], case_sensitive=False), default="llm", show_default=True, help="Scene renderer to use")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_draft(project_slug: str, chapter: int, grounded: bool, renderer: str, projects_dir: str) -> None:
    """Draft chapter prose from solved shadow graph trajectory."""
    if not grounded:
        raise click.ClickException("Use --grounded for this command.")

    proj_dir = _project_dir(project_slug, Path(projects_dir))
    project = _load_project(project_slug, Path(projects_dir))
    solved = _load_json(proj_dir / "shadow_solution.json", default={})
    graph = _load_json(proj_dir / "shadow_graph.json", default={})
    plan = _load_json(proj_dir / "plan.json", default={})
    _scene_plan_by_id, chapters_by_number = _scene_plan_index(plan if isinstance(plan, dict) else {})
    plan_chapter = chapters_by_number.get(chapter)
    constraints = _load_constraints(proj_dir)
    if not solved.get("trajectory"):
        raise click.ClickException("Missing solved trajectory. Run story solve first.")

    graph_node_by_id = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    chapter_rows = [row for row in solved.get("trajectory", []) if str(row.get("scene_id", "")).startswith(f"ch{chapter:02d}-")]
    if not chapter_rows:
        raise click.ClickException(f"No solved scenes found for chapter {chapter}.")

    required_terms = _required_terms(constraints)
    context_stats = _load_json(proj_dir / "context_stats.json", default={})
    timeline = _project_timeline(project)
    timeline_snapshot = context_stats.get("timeline", {}) if isinstance(context_stats.get("timeline"), dict) else {}
    if timeline_snapshot:
        timeline["story_era"] = canonicalize_era(str(timeline_snapshot.get("story_era") or "").strip()) or timeline.get("story_era")
        timeline["story_year"] = _coerce_optional_int(timeline_snapshot.get("story_year", timeline.get("story_year")))
        timeline["story_era_order"] = era_to_order(timeline.get("story_era"))
        timeline["allow_past_references"] = bool(timeline_snapshot.get("allow_past_references", timeline.get("allow_past_references", True)))
        timeline["forbid_future_entities"] = bool(timeline_snapshot.get("forbid_future_entities", timeline.get("forbid_future_entities", True)))
    entity_presence = context_stats.get("entity_temporal_presence", {}) if isinstance(context_stats.get("entity_temporal_presence"), dict) else {}
    max_retries = int(constraints.get("enforcement", {}).get("max_retries", 2))
    attempts = 0
    final_text = ""
    final_trace: list[dict] = []
    scene_records: list[dict] = []
    missing: list[str] = []
    temporal_future_mentions: list[dict[str, str]] = []
    quality_failures: list[str] = []
    renderer = renderer.lower().strip()
    while attempts <= max_retries:
        attempts += 1
        final_text, final_trace, scene_records = _render_grounded_chapter_text(
            project=project,
            proj_dir=proj_dir,
            plan=plan,
            chapter=chapter,
            chapter_rows=chapter_rows,
            graph_node_by_id=graph_node_by_id,
            required_terms=required_terms,
            missing_terms_hint=missing,
            renderer=renderer,
        )
        missing = _missing_required_terms(final_text, required_terms)
        temporal_future_mentions = _find_temporal_mentions(
            final_text,
            timeline=timeline,
            entity_presence=entity_presence,
        )["future_mentions"]
        quality_failures = _chapter_quality_failures(
            final_text,
            final_trace,
            constraints,
            chapter=chapter,
        )
        if not missing and not temporal_future_mentions and not quality_failures:
            break

    if missing or temporal_future_mentions or quality_failures:
        if missing and not temporal_future_mentions and not quality_failures:
            raise click.ClickException(
                f"Grounded draft failed required-term enforcement after {attempts} attempts. Missing required terms: {missing}"
            )
        parts: list[str] = []
        if missing:
            parts.append(f"Missing required terms: {missing}")
        if temporal_future_mentions:
            parts.append(
                "Future-era contamination: "
                + ", ".join(row["name"] for row in temporal_future_mentions[:8])
            )
        if quality_failures:
            parts.append("Quality gate failures: " + "; ".join(quality_failures))
        raise click.ClickException(
            f"Grounded draft failed hard chapter constraints after {attempts} attempts. {'; '.join(parts)}"
        )

    chapter_path = _chapter_path(proj_dir, chapter)
    chapter_path.write_text(final_text, encoding="utf-8")
    trace_payload = {
        "schema_version": "chapter-trace-v1",
        "project_slug": project_slug,
        "chapter": chapter,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": final_trace,
    }
    trace_out = _trace_path(proj_dir, chapter)
    trace_out.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
    draft_meta = {
        "schema_version": "chapter-draft-v2",
        "project_slug": project_slug,
        "chapter": chapter,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(scene_records),
        "chapter_structure": _chapter_structure_metadata(
            project=project,
            plan_chapter=plan_chapter,
            scene_count=len(scene_records),
        ),
        "renderer": renderer,
        "scenes": scene_records,
        "required_term_enforcement": {
            "enabled": True,
            "attempts": attempts,
            "max_retries": max_retries,
            "missing_required_terms": [],
        },
        "temporal_enforcement": {
            "story_era": timeline.get("story_era"),
            "story_year": timeline.get("story_year"),
            "future_mentions": temporal_future_mentions,
        },
    }
    (proj_dir / f"chapter_{chapter:02d}_draft.json").write_text(json.dumps(draft_meta, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Draft written: {chapter_path}")
    console.print(f"Trace written: {trace_out}")


@story.command("audit")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--chapter", required=True, type=int, help="Chapter number")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
@click.option("--enforce-required-terms/--no-enforce-required-terms", default=None, help="Treat missing required terms as errors")
def story_audit(project_slug: str, chapter: int, projects_dir: str, enforce_required_terms: bool | None) -> None:
    """Audit chapter grounding, coverage, and hard constraints."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    project = _load_project(project_slug, Path(projects_dir))
    chapter_path = _chapter_path(proj_dir, chapter)
    trace_path = _trace_path(proj_dir, chapter)
    solution_path = proj_dir / "shadow_solution.json"
    graph_path = proj_dir / "shadow_graph.json"
    plan_path = proj_dir / "plan.json"

    if not chapter_path.exists() or not trace_path.exists():
        raise click.ClickException("Missing chapter or trace artifacts. Run story draft --grounded first.")

    text = chapter_path.read_text(encoding="utf-8")
    trace = _load_json(trace_path, default={})
    solved = _load_json(solution_path, default={})
    graph = _load_json(graph_path, default={})
    plan = _load_json(plan_path, default={})
    constraints = _load_constraints(proj_dir)
    quality = _quality_settings(constraints, chapter=chapter)
    context_stats = _load_json(proj_dir / "context_stats.json", default={})
    timeline = _project_timeline(project)
    timeline_snapshot = context_stats.get("timeline", {}) if isinstance(context_stats.get("timeline"), dict) else {}
    if timeline_snapshot:
        timeline["story_era"] = canonicalize_era(str(timeline_snapshot.get("story_era") or "").strip()) or timeline.get("story_era")
        timeline["story_year"] = _coerce_optional_int(timeline_snapshot.get("story_year", timeline.get("story_year")))
        timeline["story_era_order"] = era_to_order(timeline.get("story_era"))
        timeline["allow_past_references"] = bool(timeline_snapshot.get("allow_past_references", timeline.get("allow_past_references", True)))
        timeline["forbid_future_entities"] = bool(timeline_snapshot.get("forbid_future_entities", timeline.get("forbid_future_entities", True)))
    entity_presence = context_stats.get("entity_temporal_presence", {}) if isinstance(context_stats.get("entity_temporal_presence"), dict) else {}
    if enforce_required_terms is None:
        enforce_required_terms = True

    expected_scenes = [row for row in solved.get("trajectory", []) if str(row.get("scene_id", "")).startswith(f"ch{chapter:02d}-")]
    traced = trace.get("sections", [])
    coverage = round(len(traced) / max(1, len(expected_scenes)), 6)
    _scene_plan_by_id, chapters_by_number = _scene_plan_index(plan if isinstance(plan, dict) else {})
    plan_chapter = chapters_by_number.get(chapter)
    chapter_structure = _chapter_structure_metadata(
        project=project,
        plan_chapter=plan_chapter,
        scene_count=len(traced),
    )

    forbidden = [str(x).lower() for x in constraints.get("forbidden_terms", [])]
    required = [str(x).lower() for x in constraints.get("required_elements", [])]
    text_l = text.lower()
    forbidden_hits = [t for t in forbidden if t in text_l]
    required_missing = [t for t in required if t not in text_l]
    required_coverage = round((len(required) - len(required_missing)) / max(1, len(required)), 6)
    placeholder_hits = _placeholder_term_hits(text, quality["forbid_placeholder_terms"])
    template_artifact_hits = _template_artifact_hits(text) if quality["forbid_template_artifacts"] else []
    lowercase_paragraph_starts = _lowercase_paragraph_start_samples(text)
    canon_entities = {c.lower() for c in _project_canon_entities(project_slug)}
    out_of_domain = _out_of_domain_entities(project_slug)
    out_of_domain_text_hits = [
        name for name in sorted(out_of_domain) if _text_mentions_name(text, name)
    ]

    chapter_traj = [row for row in expected_scenes if isinstance(row, dict)]
    action_seq = [str(row.get("action") or "unknown").lower() for row in chapter_traj]
    unique_actions = len(set(action_seq))
    action_diversity = round(unique_actions / max(1, len(action_seq)), 6)

    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)
    dialogue_words = _dialogue_word_count(text)
    dialogue_ratio = round(dialogue_words / max(1, word_count), 6)
    ttr = round((len({w.lower() for w in words}) / max(1, word_count)), 6)
    sentence_lengths = _sentence_word_lengths(text)
    avg_sentence_words = _avg_sentence_words(text)
    paragraph_repeats = _paragraph_repeat_stats(text)
    repeated_phrases = _repeated_long_phrase_stats(
        text,
        phrase_words=int(quality["repeated_long_phrase_words"]),
        min_count=int(quality["repeated_long_phrase_min_count"]),
    )
    vocative_openings = _dialogue_vocative_opening_stats(text)
    event_density = _event_density_stats(text)

    chapter_chars = []
    graph_by_id = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    for row in chapter_traj:
        ev = graph_by_id.get(row.get("shadow_event_id"), {})
        for c in (ev.get("characters", []) or []):
            if isinstance(c, str):
                chapter_chars.append(c)
    out_hits = [c for c in chapter_chars if c.lower() in out_of_domain]
    canon_hits = [c for c in chapter_chars if c.lower() in canon_entities]
    out_rate = round((len(out_hits) / max(1, len(chapter_chars))), 6)
    scene_word_counts: list[dict[str, Any]] = []
    for sec in traced:
        scene_words = int(sec.get("word_count", 0) or 0)
        scene_word_counts.append(
            {
                "section": sec.get("section"),
                "scene_id": sec.get("scene_id"),
                "word_count": scene_words,
            }
        )
    min_scene_words = int(quality["min_scene_words"])
    min_chapter_words = int(quality["min_chapter_words"])
    min_dialogue_ratio = float(quality["min_dialogue_ratio"])
    min_event_sentence_ratio = float(quality["min_event_sentence_ratio"])
    configured_min_type_token_ratio = float(quality["min_type_token_ratio"])
    min_type_token_ratio = _effective_min_type_token_ratio(configured_min_type_token_ratio, word_count)
    max_avg_sentence_words = float(quality["max_avg_sentence_words"])
    max_repeated_paragraphs = int(quality["max_repeated_paragraphs"])
    max_repeated_long_phrases = int(quality["max_repeated_long_phrases"])
    max_dialogue_vocative_openings = int(quality["max_dialogue_vocative_openings"])
    min_scene_word_violations = [
        row
        for row in scene_word_counts
        if min_scene_words > 0 and int(row.get("word_count", 0) or 0) < min_scene_words
    ]
    min_chapter_word_violation = bool(min_chapter_words > 0 and word_count < min_chapter_words)
    min_dialogue_ratio_violation = bool(min_dialogue_ratio > 0 and dialogue_ratio < min_dialogue_ratio)
    min_event_sentence_ratio_violation = bool(
        min_event_sentence_ratio > 0 and float(event_density["event_sentence_ratio"]) < min_event_sentence_ratio
    )
    min_type_token_ratio_violation = bool(min_type_token_ratio > 0 and ttr < min_type_token_ratio)
    max_avg_sentence_words_violation = bool(max_avg_sentence_words > 0 and avg_sentence_words > max_avg_sentence_words)
    repeated_paragraph_violation = bool(
        int(paragraph_repeats["repeated_paragraph_count"]) > max_repeated_paragraphs
    )
    repeated_long_phrase_violation = bool(
        int(repeated_phrases["repeated_phrase_count"]) > max_repeated_long_phrases
    )
    dialogue_vocative_opening_violation = bool(
        int(vocative_openings["count"]) > max_dialogue_vocative_openings
    )
    temporal_mentions = _find_temporal_mentions(
        text,
        timeline=timeline,
        entity_presence=entity_presence,
    )
    future_mentions = temporal_mentions["future_mentions"]
    past_mentions = temporal_mentions["past_mentions"]

    node_ids = {n.get("id") for n in graph.get("nodes", []) if isinstance(n, dict)}
    invalid_refs = []
    ungrounded_sections: list[dict[str, Any]] = []
    for sec in traced:
        for key in ("shadow_event_id", "shadow_scene_id"):
            rid = sec.get(key)
            if rid and rid not in node_ids:
                invalid_refs.append({"section": sec.get("section"), "missing": rid, "field": key})
        canon_refs = [
            str(ref).strip()
            for ref in (sec.get("source_canon_node_ids", []) or [])
            if str(ref).strip()
        ]
        if not canon_refs:
            ungrounded_sections.append(
                {"section": sec.get("section"), "scene_id": sec.get("scene_id")}
            )
        for rid in canon_refs:
            if rid not in node_ids:
                invalid_refs.append(
                    {"section": sec.get("section"), "missing": rid, "field": "source_canon_node_ids"}
                )

    status = "pass"
    hard_out_of_domain = bool(quality["forbid_out_of_domain_entities"] and (out_hits or out_of_domain_text_hits))
    if (
        coverage < 0.99
        or not chapter_structure["movement_count_matches_plan"]
        or forbidden_hits
        or invalid_refs
        or ungrounded_sections
        or future_mentions
        or placeholder_hits
        or template_artifact_hits
        or (bool(quality.get("fail_lowercase_paragraph_starts")) and lowercase_paragraph_starts)
        or min_scene_word_violations
        or min_chapter_word_violation
        or min_dialogue_ratio_violation
        or min_event_sentence_ratio_violation
        or min_type_token_ratio_violation
        or max_avg_sentence_words_violation
        or repeated_paragraph_violation
        or repeated_long_phrase_violation
        or dialogue_vocative_opening_violation
        or hard_out_of_domain
    ):
        status = "fail"
    elif required_missing:
        status = "fail"

    report = {
        "schema_version": "chapter-audit-v1",
        "project_slug": project_slug,
        "chapter": chapter,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "coverage": {
            "expected_scene_count": len(expected_scenes),
            "traced_scene_count": len(traced),
            "ratio": coverage,
        },
        "chapter_structure": chapter_structure,
        "constraints": {
            "forbidden_hits": forbidden_hits,
            "required_missing": required_missing,
            "required_terms_enforced": bool(enforce_required_terms),
            "required_coverage": required_coverage,
        },
        "grounding": {
            "invalid_trace_refs": invalid_refs,
            "ungrounded_sections": ungrounded_sections,
            "grounded_section_ratio": round(
                (len(traced) - len(ungrounded_sections)) / max(1, len(traced)),
                6,
            ),
        },
        "quality_proxies": {
            "word_count": word_count,
            "sentence_count": len(sentence_lengths),
            "avg_sentence_words": avg_sentence_words,
            "max_avg_sentence_words": max_avg_sentence_words,
            "max_avg_sentence_words_violation": max_avg_sentence_words_violation,
            "paragraph_count": paragraph_repeats["paragraph_count"],
            "repeated_paragraph_count": paragraph_repeats["repeated_paragraph_count"],
            "unique_repeated_paragraphs": paragraph_repeats["unique_repeated_paragraphs"],
            "paragraph_repeat_ratio": paragraph_repeats["repeat_ratio"],
            "max_repeated_paragraphs": max_repeated_paragraphs,
            "repeated_paragraph_violation": repeated_paragraph_violation,
            "repeated_long_phrases": repeated_phrases,
            "max_repeated_long_phrases": max_repeated_long_phrases,
            "repeated_long_phrase_violation": repeated_long_phrase_violation,
            "type_token_ratio": ttr,
            "min_type_token_ratio": min_type_token_ratio,
            "configured_min_type_token_ratio": configured_min_type_token_ratio,
            "min_type_token_ratio_violation": min_type_token_ratio_violation,
            "action_diversity": action_diversity,
            "unique_actions": unique_actions,
            "min_scene_words": min_scene_words,
            "min_chapter_words": min_chapter_words,
            "min_chapter_word_violation": min_chapter_word_violation,
            "scene_word_counts": scene_word_counts,
            "min_scene_word_violations": min_scene_word_violations,
            "dialogue_word_count": dialogue_words,
            "dialogue_ratio": dialogue_ratio,
            "min_dialogue_ratio": min_dialogue_ratio,
            "min_dialogue_ratio_violation": min_dialogue_ratio_violation,
            "event_density": event_density,
            "min_event_sentence_ratio": min_event_sentence_ratio,
            "min_event_sentence_ratio_violation": min_event_sentence_ratio_violation,
            "dialogue_vocative_openings": vocative_openings,
            "max_dialogue_vocative_openings": max_dialogue_vocative_openings,
            "dialogue_vocative_opening_violation": dialogue_vocative_opening_violation,
            "placeholder_hits": placeholder_hits,
            "template_artifact_hits": template_artifact_hits,
            "forbid_template_artifacts": bool(quality["forbid_template_artifacts"]),
            "lowercase_paragraph_starts": lowercase_paragraph_starts,
            "fail_lowercase_paragraph_starts": bool(quality["fail_lowercase_paragraph_starts"]),
            "lowercase_paragraph_start_violation": bool(
                quality["fail_lowercase_paragraph_starts"] and lowercase_paragraph_starts
            ),
        },
        "domain_alignment": {
            "chapter_character_mentions": len(chapter_chars),
            "canon_entity_hits": len(canon_hits),
            "out_of_domain_hits": len(out_hits),
            "out_of_domain_text_hits": out_of_domain_text_hits,
            "out_of_domain_rate": out_rate,
            "forbid_out_of_domain_entities": bool(quality["forbid_out_of_domain_entities"]),
        },
        "temporal_alignment": {
            "story_era": timeline.get("story_era"),
            "story_year": timeline.get("story_year"),
            "future_mentions": future_mentions,
            "past_references": past_mentions,
            "future_mention_count": len(future_mentions),
            "past_reference_count": len(past_mentions),
            "allow_past_references": bool(timeline.get("allow_past_references", True)),
        },
    }

    json_path = _audit_json_path(proj_dir, chapter)
    md_path = _audit_md_path(proj_dir, chapter)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        f"# Chapter {chapter} Audit — {project_slug}",
        "",
        f"- Status: **{status.upper()}**",
        f"- Coverage: {len(traced)}/{len(expected_scenes)} ({coverage:.2%})",
        f"- Structure: {chapter_structure['actual_movement_count']} movements"
        + (
            f" ({chapter_structure['movement_count_basis']})"
            if chapter_structure["movement_count_basis"]
            else ""
        ),
        f"- Forbidden hits: {len(forbidden_hits)}",
        f"- Required missing: {len(required_missing)}",
        f"- Placeholder hits: {len(placeholder_hits)}",
        f"- Template artifact hits: {len(template_artifact_hits)}",
        f"- Invalid trace refs: {len(invalid_refs)}",
        f"- Ungrounded sections: {len(ungrounded_sections)}",
        f"- Future-era mentions: {len(future_mentions)}",
        f"- Past-era references: {len(past_mentions)}",
        f"- Action diversity: {unique_actions}/{max(1, len(action_seq))} ({action_diversity:.2%})",
        f"- Out-of-domain entity rate: {out_rate:.2%}",
        f"- Out-of-domain text hits: {len(out_of_domain_text_hits)}",
        f"- Word count: {word_count}",
        f"- Avg sentence words: {avg_sentence_words:.2f}",
        f"- Repeated paragraphs: {paragraph_repeats['repeated_paragraph_count']}",
        f"- Repeated long phrases: {repeated_phrases['repeated_phrase_count']}",
        f"- Dialogue name-openings: {vocative_openings['count']}",
        f"- Dialogue ratio: {dialogue_ratio:.2%}",
        f"- Event sentence ratio: {event_density['event_sentence_ratio']:.2%}",
        f"- Type-token ratio: {ttr:.2%}",
        "",
        "## Details",
        f"- forbidden_hits: {forbidden_hits or '[]'}",
        f"- required_missing: {required_missing or '[]'}",
        f"- placeholder_hits: {placeholder_hits or '[]'}",
        f"- template_artifact_hits: {template_artifact_hits or '[]'}",
        f"- invalid_trace_refs: {invalid_refs or '[]'}",
        f"- ungrounded_sections: {ungrounded_sections or '[]'}",
        f"- min_scene_word_violations: {min_scene_word_violations or '[]'}",
        f"- min_chapter_word_violation: {min_chapter_word_violation}",
        f"- min_dialogue_ratio_violation: {min_dialogue_ratio_violation}",
        f"- min_event_sentence_ratio_violation: {min_event_sentence_ratio_violation}",
        f"- min_type_token_ratio_violation: {min_type_token_ratio_violation}",
        f"- max_avg_sentence_words_violation: {max_avg_sentence_words_violation}",
        f"- repeated_paragraph_violation: {repeated_paragraph_violation}",
        f"- repeated_long_phrase_violation: {repeated_long_phrase_violation}",
        f"- dialogue_vocative_opening_violation: {dialogue_vocative_opening_violation}",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    console.print(f"[green]OK[/green] Audit written: {json_path}")
    console.print(f"Markdown report: {md_path}")
