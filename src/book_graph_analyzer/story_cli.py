from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console

console = Console()


DEFAULT_PROJECTS_DIR = Path("data") / "projects"


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


def _default_constraints() -> dict:
    return {
        "required_elements": [],
        "forbidden_terms": [],
        "style": {
            "tone": "Consistent with project premise",
            "target_words_per_scene": 900,
        },
    }


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
