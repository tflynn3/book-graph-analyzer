from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _topk_keys(weights: dict[str, float], k: int, fallback: list[str]) -> list[str]:
    keys = [k for k, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    return keys or fallback


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

    arc_changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    arc_coherence = 1.0 - (arc_changes / max(1, len(actions) - 1))
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
        + 1.5 * arc_coherence
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


def _project_canon_entities(project_slug: str) -> list[str]:
    slug = (project_slug or "").lower()
    if "beren" in slug or "luthien" in slug:
        return [
            "Beren", "Luthien", "Lúthien", "Thingol", "Melian", "Sauron", "Morgoth",
            "Finrod", "Celegorm", "Curufin", "Huan", "Tol-in-Gaurhoth", "Doriath", "Nargothrond",
        ]
    return []


def _out_of_domain_entities(project_slug: str) -> set[str]:
    slug = (project_slug or "").lower()
    if "beren" in slug or "luthien" in slug:
        return {
            "frodo", "sam", "gandalf", "aragorn", "legolas", "gimli", "boromir", "faramir",
            "pippin", "merry", "gollum", "smeagol", "saruman", "eowyn", "theoden", "denethor",
        }
    return set()


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
    canon_entities = _project_canon_entities(project_slug)
    out_of_domain = _out_of_domain_entities(project_slug)
    # Strong project priors: canon first, then observed priors.
    top_characters = list(dict.fromkeys(canon_entities + top_characters))[:18]
    top_motifs = [k for k, _ in sorted(motif_priors.items(), key=lambda kv: kv[1], reverse=True)[:30]] or ["song", "oath", "shadow"]
    seed = abs(hash(project_slug)) % (2**32)
    rng = random.Random(seed)
    required = [str(x) for x in constraints.get("required_elements", [])]
    forbidden = {str(x).lower() for x in constraints.get("forbidden_terms", [])}

    graph_nodes = []
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
                action_choices = list(transitions.get(prev_action, {"journey": 0.34, "conflict": 0.33, "reveal": 0.33}).items())
                action = action_choices[min(rank, len(action_choices) - 1)][0]
                transition_prob = float(action_choices[min(rank, len(action_choices) - 1)][1])
                chars = rng.sample(top_characters, k=min(2 + rank, len(top_characters)))
                motifs = rng.sample(top_motifs, k=min(2, len(top_motifs)))
                if required and chapter_num == 1 and rank == 0:
                    motifs = list(dict.fromkeys((required[:1] + motifs)))

                # Spread hard anchors across early candidates to keep hard-gated solve feasible.
                if required:
                    req_idx = ((chapter_num - 1) * max(1, len(scenes)) + rank) % len(required)
                    motifs = list(dict.fromkeys(motifs + [required[req_idx]]))

                description = f"{scene.get('goal', 'Advance plot')} via {action}. Characters: {', '.join(chars[:3])}. Motifs: {', '.join(motifs[:3])}."
                if any(term in description.lower() for term in forbidden):
                    continue

                char_score = sum(float(char_priors.get(c, 0.01)) for c in chars) / max(1, len(chars))
                canon_hits = sum(1 for c in chars if c in canon_entities)
                out_of_domain_hits = sum(1 for c in chars if c.lower() in out_of_domain)
                unknown_hits = sum(1 for c in chars if c.strip().lower() in {"unknown", "they", "someone"})
                motif_score = sum(float(motif_priors.get(m, 0.005)) for m in motifs) / max(1, len(motifs))
                prior_boost = (0.10 * canon_hits) - (0.22 * out_of_domain_hits) - (0.25 * unknown_hits)
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
                        },
                        "transition_probability": round(transition_prob, 6),
                        "plausibility_score": plausibility,
                        "hard_constraints_ok": True,
                        "project_prior": {
                            "canon_hits": canon_hits,
                            "out_of_domain_hits": out_of_domain_hits,
                            "unknown_entity_hits": unknown_hits,
                        },
                    }
                )
            row_candidates.sort(key=lambda c: c["plausibility_score"], reverse=True)
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
    _load_project(project_slug, Path(projects_dir))
    context = _load_json(proj_dir / "context_stats.json", default={})
    plan = _load_json(proj_dir / "plan.json", default={})
    constraints = _load_constraints(proj_dir)
    if not context:
        raise click.ClickException("Missing context_stats.json. Run: bga story context --graph-stats")
    if not plan:
        raise click.ClickException("Missing plan.json. Run: bga story plan --auto")

    transitions = context.get("event_transition_probabilities", {})
    char_priors = context.get("character_participation_priors", {})
    motif_priors = context.get("motif_reference_density_priors", {})
    style_budget = context.get("register_style_budgets", {})
    top_characters = _topk_keys(char_priors, 16, ["Beren", "Luthien", "Thingol"])
    top_motifs = _topk_keys(motif_priors, 40, ["oath", "song", "fate", "shadow"])

    eff_seed = int(seed if seed is not None else (abs(hash(f"{project_slug}:{n}:{steps}:{method}")) % (2**32)))
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
    style_budget = context.get("register_style_budgets", {})

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
        canon_penalty = min(1.0, 0.6 * forbidden_hits + 0.4 * (missing_required / max(1, len(required))))
        canon_consistency = round(max(0.0, 1.0 - canon_penalty), 6)

        trans_vals = []
        actions = [str(s.get("action", "unknown")) for s in state]
        for i, action in enumerate(actions):
            prev = actions[i - 1] if i > 0 else "unknown"
            trans_vals.append(float(transitions.get(prev, {}).get(action, 0.05)))
        transition_likelihood = round(sum(trans_vals) / max(1, len(trans_vals)), 6)

        action_switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
        arc_coherence = round(1.0 - (action_switches / max(1, len(actions) - 1)), 6)

        target_words = float(style_budget.get("target_words_per_scene", 300))
        words = [len(str(s.get("description", "")).split()) for s in state]
        mean_words = sum(words) / max(1, len(words))
        style_register = round(max(0.0, 1.0 - min(1.0, abs(mean_words - target_words) / max(1.0, target_words))), 6)

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

    beam_width_schedule = [4, 8, 16]
    best_score = float("-inf")
    best_path: list[dict] = []
    missing_required: list[str] = []
    selected_beam_width = beam_width_schedule[0]

    for beam_width in beam_width_schedule:
        beam: list[tuple[float, list[dict]]] = [(0.0, [])]
        for sid in scene_ids:
            next_beam: list[tuple[float, list[dict]]] = []
            for base_score, path in beam:
                action_counts = Counter(str(p.get("shadow_event", {}).get("action", "unknown")) for p in path)
                chosen_characters = {
                    c.lower()
                    for p in path
                    for c in (p.get("shadow_event", {}).get("characters", []) or [])
                    if isinstance(c, str)
                }
                for cand in by_scene[sid][:6]:
                    desc = str(cand.get("shadow_event", {}).get("description", "")).lower()
                    if any(term in desc for term in forbidden):
                        continue
                    p = max(1e-6, float(cand.get("plausibility_score", 0.01)))
                    t = max(1e-6, float(cand.get("transition_probability", 0.01)))

                    action = str(cand.get("shadow_event", {}).get("action", "unknown")).strip().lower() or "unknown"
                    chars = [str(c).strip() for c in (cand.get("shadow_event", {}).get("characters", []) or []) if str(c).strip()]
                    chars_l = {c.lower() for c in chars}
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

                    score = base_score + math.log(p) + 0.5 * math.log(t) + novelty_bonus - repeat_penalty - placeholder_penalty
                    next_beam.append((score, path + [cand]))
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:beam_width] or beam

        cand_score, cand_path = beam[0]
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
        "beam_width": selected_beam_width,
        "objective": "sum(log(plausibility)+0.5*log(transition_probability)+novelty_bonus-repeat_penalty-placeholder_penalty)",
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
        missing = _missing_required_terms(final_text, required_terms)
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
        enforce_required_terms = True

    expected_scenes = [row for row in solved.get("trajectory", []) if str(row.get("scene_id", "")).startswith(f"ch{chapter:02d}-")]
    traced = trace.get("sections", [])
    coverage = round(len(traced) / max(1, len(expected_scenes)), 6)

    forbidden = [str(x).lower() for x in constraints.get("forbidden_terms", [])]
    required = [str(x).lower() for x in constraints.get("required_elements", [])]
    text_l = text.lower()
    forbidden_hits = [t for t in forbidden if t in text_l]
    required_missing = [t for t in required if t not in text_l]
    canon_entities = {c.lower() for c in _project_canon_entities(project_slug)}
    out_of_domain = _out_of_domain_entities(project_slug)

    chapter_traj = [row for row in expected_scenes if isinstance(row, dict)]
    action_seq = [str(row.get("action") or "unknown").lower() for row in chapter_traj]
    unique_actions = len(set(action_seq))
    action_diversity = round(unique_actions / max(1, len(action_seq)), 6)

    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)
    ttr = round((len({w.lower() for w in words}) / max(1, word_count)), 6)

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

    node_ids = {n.get("id") for n in graph.get("nodes", []) if isinstance(n, dict)}
    invalid_refs = []
    for sec in traced:
        for key in ("shadow_event_id", "shadow_scene_id"):
            rid = sec.get(key)
            if rid and rid not in node_ids:
                invalid_refs.append({"section": sec.get("section"), "missing": rid, "field": key})

    status = "pass"
    if coverage < 0.99 or forbidden_hits or invalid_refs:
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
        "constraints": {
            "forbidden_hits": forbidden_hits,
            "required_missing": required_missing,
            "required_terms_enforced": bool(enforce_required_terms),
        },
        "grounding": {
            "invalid_trace_refs": invalid_refs,
        },
        "quality_proxies": {
            "word_count": word_count,
            "type_token_ratio": ttr,
            "action_diversity": action_diversity,
            "unique_actions": unique_actions,
        },
        "domain_alignment": {
            "chapter_character_mentions": len(chapter_chars),
            "canon_entity_hits": len(canon_hits),
            "out_of_domain_hits": len(out_hits),
            "out_of_domain_rate": out_rate,
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
        f"- Action diversity: {unique_actions}/{max(1, len(action_seq))} ({action_diversity:.2%})",
        f"- Out-of-domain entity rate: {out_rate:.2%}",
        f"- Word count: {word_count}",
        "",
        "## Details",
        f"- forbidden_hits: {forbidden_hits or '[]'}",
        f"- required_missing: {required_missing or '[]'}",
        f"- invalid_trace_refs: {invalid_refs or '[]'}",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    console.print(f"[green]OK[/green] Audit written: {json_path}")
    console.print(f"Markdown report: {md_path}")
