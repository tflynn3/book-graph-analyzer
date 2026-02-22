from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .lineage import parse_lineage


_TOKEN_RE = re.compile(r"[^a-z0-9\s\-']+")


@dataclass(frozen=True)
class LineageThreshold:
    min_lineages: int
    min_forms: int
    min_derivations: int
    min_join_rate: float = 0.95


BOOK_THRESHOLDS: dict[str, LineageThreshold] = {
    "hobbit": LineageThreshold(min_lineages=8, min_forms=18, min_derivations=10),
    "fellowship_of_ring": LineageThreshold(min_lineages=8, min_forms=18, min_derivations=10),
    "two_towers": LineageThreshold(min_lineages=8, min_forms=18, min_derivations=10),
    "return_of_king": LineageThreshold(min_lineages=8, min_forms=18, min_derivations=10),
    "silmarillion": LineageThreshold(min_lineages=9, min_forms=24, min_derivations=14),
}


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = _TOKEN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_entity_id(seed_kind: str, seed_id: str) -> str:
    prefix = {"characters": "char", "places": "place", "objects": "obj"}[seed_kind]
    return f"{prefix}_{seed_id}"


def load_seed_alias_catalog(repo_root: Path) -> dict[str, set[str]]:
    catalog: dict[str, set[str]] = {}
    for seed_kind in ("characters", "places", "objects"):
        payload = json.loads((repo_root / "data" / "seeds" / f"{seed_kind}.json").read_text(encoding="utf-8"))
        for row in payload:
            seed_id = str(row.get("id") or "").strip()
            canonical_name = str(row.get("canonical_name") or "").strip()
            if not seed_id or not canonical_name:
                continue
            eid = canonical_entity_id(seed_kind, seed_id)
            forms = {_norm(canonical_name)}
            for alias in row.get("aliases", []) or []:
                alias_s = _norm(str(alias))
                if alias_s:
                    forms.add(alias_s)
            catalog[eid] = forms
    return catalog


def load_book_surface_text(events_path: Path, lore_depth_path: Path | None = None) -> str:
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    events = payload.get("events", {})
    rows = list(events.values()) if isinstance(events, dict) else list(events)
    parts: list[str] = []
    for row in rows:
        for key in ("description", "agent", "patient", "action", "source_text"):
            value = str(row.get(key) or "").strip()
            if value:
                parts.append(value)

    if lore_depth_path and lore_depth_path.exists():
        lore = json.loads(lore_depth_path.read_text(encoding="utf-8"))
        for art in lore.get("artifacts", []) or []:
            name = str(art.get("name") or "").strip()
            desc = str(art.get("description") or "").strip()
            if name:
                parts.append(name)
            if desc:
                parts.append(desc)

    return "\n".join(parts)


def _count_alias_hits(corpus: str, aliases: set[str]) -> tuple[int, list[str]]:
    found: Counter[str] = Counter()
    for alias in aliases:
        if len(alias) < 3:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
        hits = len(pattern.findall(corpus))
        if hits:
            found[alias] += hits

    if not found:
        return 0, []
    ordered = [a for a, _ in found.most_common()]
    return sum(found.values()), ordered


def infer_language(form: str, rank: int) -> str:
    token = _norm(form)
    if rank == 0:
        return "Common Speech"
    if any(x in token for x in ("dur", "khaz", "gund", "goblin")):
        return "Khuzdul"
    if any(x in token for x in ("û", "â", "ô", "adun", "numen")):
        return "Adûnaic"
    if any(x in token for x in ("th", "nd", "gl", "ril", "dor", "loth", "ien")):
        return "Sindarin"
    return "Quenya"


def build_lineages_from_corpus(corpus: str, catalog: dict[str, set[str]], source_passage_id: str, min_mentions: int = 2, max_entities: int = 12) -> list[dict]:
    corpus_norm = _norm(corpus)
    scored: list[tuple[str, int, list[str]]] = []
    for entity_id, aliases in catalog.items():
        hits, ranked_aliases = _count_alias_hits(corpus_norm, aliases)
        if hits >= min_mentions and ranked_aliases:
            scored.append((entity_id, hits, ranked_aliases))

    scored.sort(key=lambda t: t[1], reverse=True)
    lineages: list[dict] = []
    for entity_id, _hits, ranked_aliases in scored[:max_entities]:
        selected = ranked_aliases[:3]
        forms = []
        for i, form_text in enumerate(selected):
            forms.append(
                {
                    "id": f"legacy_{entity_id}_{i}",
                    "form": form_text.title(),
                    "language": infer_language(form_text, i),
                    "source_passage_id": source_passage_id,
                }
            )

        derivations = []
        for i in range(1, len(forms)):
            derivations.append(
                {
                    "source_form_id": forms[0]["id"],
                    "target_form_id": forms[i]["id"],
                    "derivation_type": "translation" if i == 1 else "adaptation",
                }
            )

        if len(forms) >= 2:
            lineages.append({"entity_id": entity_id, "forms": forms, "derivations": derivations})

    return lineages


def compute_lineage_metrics(lineages: list[dict], valid_entity_ids: set[str]) -> dict[str, float | int]:
    lineage_count = len(lineages)
    forms = sum(len(l.get("forms", [])) for l in lineages)
    derivations = sum(len(l.get("derivations", [])) for l in lineages)
    joined = 0
    for lin in lineages:
        entity_ok = lin.get("entity_id") in valid_entity_ids
        for _ in lin.get("forms", []):
            if entity_ok:
                joined += 1
    join_rate = (joined / forms) if forms else 1.0
    return {
        "lineages": lineage_count,
        "forms": forms,
        "derivations": derivations,
        "join_rate": join_rate,
    }


def threshold_pass(metrics: dict[str, float | int], threshold: LineageThreshold) -> bool:
    return (
        int(metrics["lineages"]) >= threshold.min_lineages
        and int(metrics["forms"]) >= threshold.min_forms
        and int(metrics["derivations"]) >= threshold.min_derivations
        and float(metrics["join_rate"]) >= threshold.min_join_rate
    )


def parse_lineages_payload(raw_lineages: list[dict]) -> list:
    return [parse_lineage(item) for item in raw_lineages]
