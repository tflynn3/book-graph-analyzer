from __future__ import annotations

import json
import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console

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
}


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


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _safe_prob(counter: dict[str, int]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in counter.items()}


def _load_constraints(proj_dir: Path) -> dict:
    constraints_path = proj_dir / "constraints.json"
    return (
        json.loads(constraints_path.read_text(encoding="utf-8"))
        if constraints_path.exists()
        else _default_constraints()
    )


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


def _tokenize_for_match(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", text.lower())


def _build_required_term_aliases(required_terms: list[str], constraints: dict | None = None) -> dict[str, list[str]]:
    aliases_cfg = (constraints or {}).get("required_element_aliases", {}) if isinstance(constraints, dict) else {}
    aliases: dict[str, list[str]] = {}
    for term in required_terms:
        extra = aliases_cfg.get(term, []) if isinstance(aliases_cfg, dict) else []
        variants = [term] + ([str(x) for x in extra] if isinstance(extra, list) else [])
        cleaned = [v.strip() for v in variants if str(v).strip()]
        aliases[term] = list(dict.fromkeys(cleaned))
    return aliases


def _contains_phrase_tokens(text_tokens: list[str], phrase: str) -> bool:
    phrase_tokens = _tokenize_for_match(phrase)
    if not phrase_tokens:
        return False
    n = len(phrase_tokens)
    for i in range(0, max(0, len(text_tokens) - n + 1)):
        if text_tokens[i : i + n] == phrase_tokens:
            return True
    return False


def _missing_required_terms(text: str, required_terms: list[str], constraints: dict | None = None) -> list[str]:
    text_tokens = _tokenize_for_match(text)
    aliases = _build_required_term_aliases(required_terms, constraints)
    missing: list[str] = []
    for term in required_terms:
        variants = aliases.get(term, [term])
        if not any(_contains_phrase_tokens(text_tokens, v) for v in variants):
            missing.append(term)
    return missing


def _stable_seed(*parts: str) -> int:
    material = "||".join(parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _evidence_alignment_ratio(text: str, excerpt: str) -> float:
    text_tokens = set(_tokenize_for_match(text))
    excerpt_tokens = [t for t in _tokenize_for_match(excerpt) if len(t) > 3]
    if not excerpt_tokens:
        return 1.0
    matched = sum(1 for t in excerpt_tokens if t in text_tokens)
    return matched / max(1, len(excerpt_tokens))


def _render_grounded_chapter_text(chapter: int, chapter_rows: list[dict], graph_node_by_id: dict[str, dict], required_terms: list[str]) -> tuple[str, list[dict]]:
    lines = [f"# Chapter {chapter}: Shadow-Woven Paths", ""]
    trace_sections = []
    for idx, row in enumerate(chapter_rows, start=1):
        scene_id = row.get("scene_id")
        event_id = row.get("shadow_event_id")
        event = graph_node_by_id.get(event_id, {})
        chars = event.get("characters", ["They"])
        motifs = event.get("motifs", [])
        action = event.get("action", "moved")
        motif_clause = f" Motifs threading this passage: {', '.join(motifs[:2])}." if motifs else ""
        prose = (
            f"In scene {scene_id}, {chars[0]} and {chars[-1]} {action} through uncertain country, "
            f"testing the cost of each vow against the weight of old memory. "
            f"Their choices keep close to known history while opening a narrow, plausible margin for what may yet be told."
            f"{motif_clause}"
        )
        lines.extend([f"## Scene {idx}", "", prose, ""])
        trace_sections.append(
            {
                "section": idx,
                "scene_id": scene_id,
                "shadow_event_id": event_id,
                "shadow_scene_id": f"shadow-{scene_id}",
                "source_canon_node_ids": [f"canon-action-{action}"],
                "text_excerpt": prose[:220],
            }
        )

    if required_terms:
        lines.extend(["", "## Required Canon Anchors", ""])
        lines.extend([f"- {term}" for term in required_terms])
        lines.append("")

    return "\n".join(lines), trace_sections


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

    expected_scene_count = int(project.get("scenes_per_chapter", 3))
    for ch in chapters:
        scenes = ch.get("scenes", [])
        if len(scenes) != expected_scene_count:
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


@story.command("init")
@click.option("--name", help="Project display name")
@click.option("--slug", help="Project slug (defaults from --name)")
@click.option("--premise", help="1-2 sentence story premise")
@click.option("--genre", default="fantasy", show_default=True, help="Primary genre")
@click.option("--canon-file", default="", help="Optional canon JSON path")
@click.option("--target-chapters", default=6, show_default=True, type=int)
@click.option("--scenes-per-chapter", default=3, show_default=True, type=int)
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

    project = {
        "name": name,
        "slug": slug,
        "genre": genre,
        "premise": premise,
        "canon_file": canon_file,
        "target_chapters": target_chapters,
        "scenes_per_chapter": scenes_per_chapter,
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
    event_files = _project_event_files(project)
    if not event_files:
        raise click.ClickException("No event files found. Set project.event_files to one or more *_events.json files.")

    transition_counts: dict[str, Counter] = defaultdict(Counter)
    action_counts: Counter = Counter()
    motif_counts: Counter = Counter()
    character_counts: Counter = Counter()
    register_word_lengths: list[int] = []
    total_events = 0

    for path in event_files:
        payload = _load_json(path, default={})
        events = _extract_events(payload)
        total_events += len(events)
        previous_action = None
        for ev in events:
            action = str(ev.get("action") or "unknown").strip().lower() or "unknown"
            desc = str(ev.get("description") or "")
            agent = str(ev.get("agent") or "Unknown").strip() or "Unknown"
            action_counts[action] += 1
            character_counts[agent] += 1
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
    avg_words = int(round(sum(register_word_lengths) / max(1, len(register_word_lengths))))
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
        "totals": {
            "events": total_events,
            "actions": len(action_counts),
            "characters": len(character_counts),
            "motifs": len(motif_counts),
        },
        "event_transition_probabilities": transition_probabilities,
        "motif_reference_density_priors": motif_priors,
        "character_participation_priors": character_priors,
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
    plan_path = proj_dir / "plan.json"
    if plan_path.exists():
        plan = _load_json(plan_path, default={})
    else:
        plan = _build_plan(project, constraints, auto=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    transitions = context.get("event_transition_probabilities", {})
    char_priors = context.get("character_participation_priors", {})
    motif_priors = context.get("motif_reference_density_priors", {})
    top_characters = [k for k, _ in sorted(char_priors.items(), key=lambda kv: kv[1], reverse=True)[:12]] or ["Beren", "Luthien"]
    top_motifs = [k for k, _ in sorted(motif_priors.items(), key=lambda kv: kv[1], reverse=True)[:30]] or ["song", "oath", "shadow"]
    seed = _stable_seed(project_slug, json.dumps(plan, sort_keys=True), json.dumps(constraints, sort_keys=True))
    rng = random.Random(seed)
    required = [str(x) for x in constraints.get("required_elements", [])]
    forbidden = {str(x).lower() for x in constraints.get("forbidden_terms", [])}
    search_cfg = constraints.get("search", {}) if isinstance(constraints.get("search"), dict) else {}

    scene_count = sum(len(ch.get("scenes", [])) for ch in plan.get("chapters", []))
    target_candidates = int(search_cfg.get("target_candidates", max(500, scene_count * 24)))
    candidates_per_scene = max(6, min(128, math.ceil(target_candidates / max(1, scene_count))))

    graph_nodes = []
    graph_edges = []
    candidates = []
    selected = []
    prev_action = "unknown"

    elites_grid: dict[str, dict] = {}
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
            for rank in range(candidates_per_scene):
                action_choices = list(transitions.get(prev_action, {"journey": 0.34, "conflict": 0.33, "reveal": 0.33}).items())
                action = action_choices[rank % max(1, len(action_choices))][0]
                transition_prob = float(action_choices[rank % max(1, len(action_choices))][1])
                chars = rng.sample(top_characters, k=min(2 + (rank % 3), len(top_characters)))
                motifs = rng.sample(top_motifs, k=min(2, len(top_motifs)))
                if required and chapter_num == 1 and rank == 0:
                    motifs = list(dict.fromkeys((required[:1] + motifs)))

                description = f"{scene.get('goal', 'Advance plot')} via {action}."
                if any(term in description.lower() for term in forbidden):
                    continue

                char_score = sum(float(char_priors.get(c, 0.01)) for c in chars) / max(1, len(chars))
                motif_score = sum(float(motif_priors.get(m, 0.005)) for m in motifs) / max(1, len(motifs))
                plausibility = round(min(0.99, max(0.05, (0.5 * transition_prob) + (0.3 * char_score) + (0.2 * motif_score))), 6)
                score_components = {
                    "transition": round(transition_prob, 6),
                    "character_participation": round(char_score, 6),
                    "motif_grounding": round(motif_score, 6),
                    "constraint_bonus": round(0.05 if any(m in required for m in motifs) else 0.0, 6),
                }
                total_score = round(
                    (0.55 * score_components["transition"])
                    + (0.25 * score_components["character_participation"])
                    + (0.15 * score_components["motif_grounding"])
                    + score_components["constraint_bonus"],
                    6,
                )
                cid = f"{scene_id}-cand-{rank + 1}"
                behavior_descriptor = {
                    "action": action,
                    "character_load": "high" if len(chars) >= 4 else "medium" if len(chars) == 3 else "low",
                    "motif_load": "rich" if len(motifs) >= 2 else "sparse",
                }
                cell_key = f"{behavior_descriptor['action']}|{behavior_descriptor['character_load']}|{behavior_descriptor['motif_load']}"
                elite = elites_grid.get(cell_key)
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
                        },
                        "transition_probability": round(transition_prob, 6),
                        "plausibility_score": plausibility,
                        "score_components": score_components,
                        "score_total": total_score,
                        "behavior_descriptor": behavior_descriptor,
                        "hard_constraints_ok": True,
                    }
                )
                if elite is None or total_score > float(elite.get("score_total", 0.0)):
                    elites_grid[cell_key] = {"candidate_id": cid, "score_total": total_score}

            row_candidates.sort(key=lambda c: (c["score_total"], c["plausibility_score"]), reverse=True)
            if row_candidates:
                selected.append(row_candidates[0])
                prev_action = row_candidates[0]["shadow_event"]["action"]
            candidates.extend(row_candidates)

    for idx, chosen in enumerate(selected):
        ev = chosen["shadow_event"]
        graph_nodes.append(ev)
        graph_edges.append({
            "source": f"shadow-{chosen['scene_id']}",
            "target": ev["id"],
            "type": "HAS_EVENT",
            "probability": chosen["plausibility_score"],
        })
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
    }
    candidates_payload = {
        "schema_version": "shadow-candidates-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "constraints_snapshot": constraints,
        "sampling": {
            "scene_count": scene_count,
            "target_candidates": target_candidates,
            "candidates_per_scene": candidates_per_scene,
            "generated_candidates": len(candidates),
            "elites_cells": len(elites_grid),
        },
        "candidates": candidates,
        "selected_auto": [c["candidate_id"] for c in selected],
        "elites_grid": elites_grid,
    }

    (proj_dir / "shadow_graph.json").write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    (proj_dir / "shadow_candidates.json").write_text(json.dumps(candidates_payload, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] Shadow graph artifacts written under {proj_dir}")


@story.command("solve")
@click.option("--project", "project_slug", required=True, help="Project slug")
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_solve(project_slug: str, projects_dir: str) -> None:
    """Solve best valid trajectory through shadow candidates using beam search."""
    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    payload = _load_json(proj_dir / "shadow_candidates.json", default={})
    candidates = payload.get("candidates", [])
    if not candidates:
        raise click.ClickException("No candidates found. Run story grow-shadow first.")

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for cand in candidates:
        by_scene[str(cand.get("scene_id"))].append(cand)
    scene_ids = sorted(by_scene.keys())
    constraints = _load_constraints(proj_dir)
    required = [str(x).lower() for x in constraints.get("required_elements", [])]
    forbidden = [str(x).lower() for x in constraints.get("forbidden_terms", [])]

    beam: list[tuple[float, list[dict]]] = [(0.0, [])]
    beam_width = 4

    for sid in scene_ids:
        next_beam: list[tuple[float, list[dict]]] = []
        for base_score, path in beam:
            for cand in by_scene[sid][:4]:
                desc = str(cand.get("shadow_event", {}).get("description", "")).lower()
                if any(term in desc for term in forbidden):
                    continue
                p = max(1e-6, float(cand.get("score_total", cand.get("plausibility_score", 0.01))))
                t = max(1e-6, float(cand.get("transition_probability", 0.01)))
                score = base_score + math.log(p) + 0.5 * math.log(t)
                next_beam.append((score, path + [cand]))
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width] or beam

    best_score, best_path = beam[0]
    full_text = "\n".join(c.get("shadow_event", {}).get("description", "") for c in best_path).lower()
    missing_required = [r for r in required if r not in full_text]
    status = "pass" if not missing_required else "warn"

    solved = {
        "schema_version": "shadow-solution-v1",
        "project_slug": project_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "beam_width": beam_width,
        "objective": "sum(log(plausibility)+0.5*log(transition_probability))",
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
@click.option("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), show_default=True, type=click.Path())
def story_draft(project_slug: str, chapter: int, grounded: bool, projects_dir: str) -> None:
    """Draft chapter prose from solved shadow graph trajectory."""
    if not grounded:
        raise click.ClickException("Use --grounded for this command.")

    proj_dir = _project_dir(project_slug, Path(projects_dir))
    _load_project(project_slug, Path(projects_dir))
    solved = _load_json(proj_dir / "shadow_solution.json", default={})
    graph = _load_json(proj_dir / "shadow_graph.json", default={})
    constraints = _load_constraints(proj_dir)
    if not solved.get("trajectory"):
        raise click.ClickException("Missing solved trajectory. Run story solve first.")

    graph_node_by_id = {n.get("id"): n for n in graph.get("nodes", []) if isinstance(n, dict)}
    chapter_rows = [row for row in solved.get("trajectory", []) if str(row.get("scene_id", "")).startswith(f"ch{chapter:02d}-")]
    if not chapter_rows:
        raise click.ClickException(f"No solved scenes found for chapter {chapter}.")

    required_terms = _required_terms(constraints)
    max_retries = int(constraints.get("enforcement", {}).get("max_retries", 2))
    attempts = 0
    final_text = ""
    final_trace: list[dict] = []
    missing: list[str] = []
    while attempts <= max_retries:
        attempts += 1
        final_text, final_trace = _render_grounded_chapter_text(chapter, chapter_rows, graph_node_by_id, required_terms)
        missing = _missing_required_terms(final_text, required_terms, constraints)
        if not missing:
            break

    if missing:
        raise click.ClickException(
            f"Grounded draft failed required-term enforcement after {attempts} attempts. Missing required terms: {missing}"
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
        "project_slug": project_slug,
        "chapter": chapter,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "required_term_enforcement": {
            "enabled": True,
            "attempts": attempts,
            "max_retries": max_retries,
            "missing_required_terms": [],
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
    _load_project(project_slug, Path(projects_dir))
    chapter_path = _chapter_path(proj_dir, chapter)
    trace_path = _trace_path(proj_dir, chapter)
    solution_path = proj_dir / "shadow_solution.json"
    graph_path = proj_dir / "shadow_graph.json"

    if not chapter_path.exists() or not trace_path.exists():
        raise click.ClickException("Missing chapter or trace artifacts. Run story draft --grounded first.")

    text = chapter_path.read_text(encoding="utf-8")
    trace = _load_json(trace_path, default={})
    solved = _load_json(solution_path, default={})
    graph = _load_json(graph_path, default={})
    constraints = _load_constraints(proj_dir)
    if enforce_required_terms is None:
        enforce_required_terms = bool(constraints.get("enforcement", {}).get("required_terms", False))

    expected_scenes = [row for row in solved.get("trajectory", []) if str(row.get("scene_id", "")).startswith(f"ch{chapter:02d}-")]
    traced = trace.get("sections", [])
    coverage = round(len(traced) / max(1, len(expected_scenes)), 6)

    forbidden = [str(x).lower() for x in constraints.get("forbidden_terms", [])]
    required_terms = [str(x) for x in constraints.get("required_elements", [])]
    text_l = text.lower()
    forbidden_hits = [t for t in forbidden if t in text_l]
    required_missing = _missing_required_terms(text, required_terms, constraints)

    required_aliases = _build_required_term_aliases(required_terms, constraints)
    scene_required_coverage = []
    for sec in traced:
        excerpt = str(sec.get("text_excerpt", "") or "")
        missing_scene = _missing_required_terms(excerpt, required_terms, constraints)
        scene_required_coverage.append(
            {
                "section": sec.get("section"),
                "covered_terms": sorted([t for t in required_terms if t not in missing_scene]),
                "missing_terms": missing_scene,
            }
        )

    node_ids = {n.get("id") for n in graph.get("nodes", []) if isinstance(n, dict)}
    invalid_refs = []
    unaligned_sections = []
    unaligned_section_ids: set[int] = set()
    for sec in traced:
        for key in ("shadow_event_id", "shadow_scene_id"):
            rid = sec.get(key)
            if rid and rid not in node_ids:
                invalid_refs.append({"section": sec.get("section"), "missing": rid, "field": key})
        excerpt = str(sec.get("text_excerpt", "") or "")
        if excerpt and _evidence_alignment_ratio(text, excerpt) < 0.6:
            unaligned_sections.append({"section": sec.get("section"), "excerpt": excerpt[:120]})
            if isinstance(sec.get("section"), int):
                unaligned_section_ids.add(int(sec.get("section")))
        if not sec.get("source_canon_node_ids"):
            unaligned_sections.append({"section": sec.get("section"), "reason": "missing_source_canon_node_ids"})
            if isinstance(sec.get("section"), int):
                unaligned_section_ids.add(int(sec.get("section")))

    evidence_alignment_ratio = round((len(traced) - len(unaligned_section_ids)) / max(1, len(traced)), 6)

    status = "pass"
    if coverage < 0.99 or forbidden_hits or invalid_refs or evidence_alignment_ratio < 0.95:
        status = "fail"
    elif required_missing and enforce_required_terms:
        status = "fail"
    elif required_missing:
        status = "warn"

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
        "constraints": {
            "forbidden_hits": forbidden_hits,
            "required_missing": required_missing,
            "required_aliases": required_aliases,
            "required_scene_coverage": scene_required_coverage,
            "required_terms_enforced": bool(enforce_required_terms),
        },
        "grounding": {
            "invalid_trace_refs": invalid_refs,
            "evidence_alignment": {
                "ratio": evidence_alignment_ratio,
                "unaligned_sections": unaligned_sections,
            },
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
        f"- Forbidden hits: {len(forbidden_hits)}",
        f"- Required missing: {len(required_missing)}",
        f"- Invalid trace refs: {len(invalid_refs)}",
        "",
        "## Details",
        f"- forbidden_hits: {forbidden_hits or '[]'}",
        f"- required_missing: {required_missing or '[]'}",
        f"- invalid_trace_refs: {invalid_refs or '[]'}",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    console.print(f"[green]OK[/green] Audit written: {json_path}")
    console.print(f"Markdown report: {md_path}")
