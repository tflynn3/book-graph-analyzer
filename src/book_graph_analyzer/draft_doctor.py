"""Revision-oriented diagnostics for generated draft chapters."""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ABSTRACT_TERMS = {
    "burden",
    "certainty",
    "doubt",
    "duty",
    "fear",
    "hope",
    "knowledge",
    "labour",
    "mercy",
    "need",
    "patience",
    "service",
    "shadow",
    "silence",
    "truth",
    "warning",
    "wisdom",
}

PHYSICAL_TERMS = {
    "ash",
    "bank",
    "blade",
    "blood",
    "bone",
    "branch",
    "cloak",
    "cord",
    "ditch",
    "door",
    "fish",
    "ford",
    "hand",
    "lamp",
    "letter",
    "map",
    "mud",
    "net",
    "rain",
    "reed",
    "river",
    "road",
    "rope",
    "stone",
    "track",
    "water",
}

OBJECT_TERMS = {
    "branches",
    "branch",
    "cloak",
    "fish bones",
    "folded letters",
    "lamp",
    "lamps",
    "letter",
    "letters",
    "map",
    "maps",
    "muddy water",
    "rope",
    "staff",
    "weathered cloak",
}

CAUSAL_VERBS = {
    "altered",
    "answered",
    "blocked",
    "broke",
    "burned",
    "carried",
    "caught",
    "changed",
    "chose",
    "crossed",
    "cut",
    "decided",
    "drew",
    "dropped",
    "delayed",
    "entered",
    "escaped",
    "exposed",
    "fastened",
    "fled",
    "forced",
    "found",
    "gave",
    "guarded",
    "handled",
    "held",
    "hid",
    "kept",
    "learned",
    "left",
    "marked",
    "matter",
    "mattered",
    "misled",
    "moved",
    "named",
    "opened",
    "placed",
    "questioned",
    "read",
    "reached",
    "revealed",
    "sent",
    "served",
    "set",
    "spoke",
    "tested",
    "tightened",
    "tied",
    "turned",
    "used",
    "warned",
    "weighed",
}

STATE_CHANGE_TERMS = {
    "knowledge": {
        "asked",
        "answered",
        "discovered",
        "heard",
        "learned",
        "named",
        "questioned",
        "revealed",
        "told",
        "warned",
    },
    "location": {
        "arrived",
        "brought",
        "came",
        "crossed",
        "entered",
        "left",
        "reached",
        "returned",
        "rode",
        "turned",
        "went",
    },
    "danger": {
        "escaped",
        "failed",
        "fled",
        "followed",
        "hunted",
        "loose",
        "lost",
        "watched",
    },
    "custody": {
        "bound",
        "caught",
        "guarded",
        "held",
        "prisoner",
        "seized",
        "tied",
    },
    "relationship": {
        "accepted",
        "agreed",
        "parted",
        "promised",
        "refused",
        "trusted",
        "vowed",
    },
}

PLACEHOLDER_PATTERNS: list[tuple[str, str, str]] = [
    ("brace_placeholder", r"\{[^}\n]+\}", "Unreplaced template token."),
    ("in_road", r"\bIn\s+Road\b", "Malformed bare location phrase."),
    ("about_road", r"\bAbout\s+Road\b", "Malformed bare location phrase."),
    ("in_westward_road", r"\bIn\s+the\s+westward\s+road\b", "Malformed generated road phrase."),
    ("about_westward_road", r"\bAbout\s+the\s+westward\s+road\b", "Malformed generated road phrase."),
    ("maps_and_road", r"\bmaps?\s+and\s+Road\b", "Mixed object/location placeholder."),
    (
        "object_and_road",
        r"\b(?:branches?|cloak|fish bones?|folded letters?|lamps?|maps?|muddy water|"
        r"rope|staff|weathered cloak)\s+and\s+Road\b",
        "Mixed object/location placeholder.",
    ),
    (
        "road_and_object",
        r"\bRoad\s+and\s+(?:branches?|cloak|fish bones?|folded letters?|lamps?|maps?|"
        r"muddy water|rope|staff|weathered cloak)\b",
        "Mixed location/object placeholder.",
    ),
    ("meta_end_of_chapter", r"\bBy the end of the chapter\b", "Editorial process leaked into the fiction."),
    ("meta_final_page", r"\bthe final page\b", "Editorial process leaked into the fiction."),
    ("meta_chapter_could_not", r"\bThe chapter could not\b", "Editorial process leaked into the fiction."),
    ("meta_three_movements", r"\bthree movements of the tale\b", "Editorial process leaked into the fiction."),
    (
        "meta_imagination_supplied",
        r"\bimagination supplied (?:one|it|them)\b",
        "Authorial reasoning leaked into the viewpoint.",
    ),
    (
        "meta_borrowed_before_time",
        r"\bborrowed before (?:its|their) time\b",
        "Canon-management language leaked into the fiction.",
    ),
]

MODERN_ANALYTICAL_PATTERNS: list[tuple[str, str]] = [
    ("stopping_rule", r"\bstopping rule\b"),
    ("independent_uncertainties", r"\bindependent uncertainties\b"),
    ("revise_fear_downward", r"\brevise (?:the )?fear downward\b"),
    ("failed_experiment", r"\bfailed experiment\b"),
    ("controlled_risk", r"\bcontrolled risk\b"),
    ("trauma_response", r"\btrauma response\b"),
    ("alternative_causes", r"\balternative causes\b"),
    ("forensic_process", r"\b(?:forensic|provenance|compliance)\b"),
    ("formal_process", r"\b(?:protocol|procedure)\b"),
]

ENDING_PATTERNS: list[tuple[str, str]] = [
    ("it_had_to", r"\bIt had to\.?$"),
    ("for_now_enough", r"\bFor now,? it was enough\.?$"),
    ("road_opened", r"\bThe road opened\b"),
    ("warning_took_road", r"\bSo warning took the road\b"),
    ("shape_of_hope", r"\bshape of hope\b"),
    ("that_was_enough", r"\bthat was enough\.?$"),
    ("road_would_remember", r"\bThe road would remember\.?$"),
]

APHORISM_PATTERNS = [
    re.compile(r"^(a|an|the|no|every|some|there|it|so|hope|mercy|truth)\b", re.I),
    re.compile(r"\b(may|must|cannot|can|should|need not|does not|would)\b", re.I),
]

STRICT_BLOCKING_CATEGORIES = {
    "draft_fullness",
    "placeholder_continuity_lint",
    "repetition_graph",
    "scene_causality",
    "tolkien_register_balance",
    "voice_differentiation",
    "object_causality",
    "ending_cadence_overload",
}

# "Strict" is the publication/readiness gate.  Medium findings in the
# categories above describe pervasive structural or voice defects (rather than
# optional polish), so treating them as advisory made the gate report PASS for
# drafts with hundreds of material findings.
STRICT_BLOCKING_SEVERITIES = {"high", "medium"}

PROFILE_FULLNESS_DEFAULTS = {
    "tolkien": {
        "min_chapter_words": 3000,
        "min_scene_words": 450,
        "min_total_words": 36000,
    }
}


@dataclass(frozen=True)
class ParagraphRef:
    chapter: int
    chapter_title: str
    scene: int
    scene_title: str
    paragraph: int
    text: str


@dataclass(frozen=True)
class SceneRef:
    chapter: int
    chapter_title: str
    scene: int
    scene_title: str
    paragraphs: tuple[ParagraphRef, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


@dataclass(frozen=True)
class DialogueRef:
    chapter: int
    scene: int
    paragraph: int
    speaker: str
    line: str


def analyze_draft(
    path: Path,
    profile: str = "tolkien",
    *,
    min_chapter_words: int | None = None,
    min_scene_words: int | None = None,
    min_total_words: int | None = None,
) -> dict[str, Any]:
    """Analyze markdown draft chapters and return a revision report."""
    if profile != "tolkien":
        raise ValueError(f"Unsupported draft doctor profile: {profile}")

    fullness = _fullness_settings(
        profile,
        min_chapter_words=min_chapter_words,
        min_scene_words=min_scene_words,
        min_total_words=min_total_words,
    )
    chapters = _load_chapters(path)
    scenes = [scene for chapter in chapters for scene in chapter["scenes"]]
    paragraphs = [p for scene in scenes for p in scene.paragraphs]
    issues: list[dict[str, Any]] = []

    issues.extend(_fullness_issues(chapters, scenes, fullness))
    issues.extend(_placeholder_issues(paragraphs))
    issues.extend(_paragraph_repetition_issues(paragraphs))
    issues.extend(_scene_opening_repetition_issues(scenes))
    issues.extend(_maxim_repetition_issues(paragraphs))
    issues.extend(_scene_causality_issues(scenes))
    issues.extend(_register_balance_issues(paragraphs, scenes))

    dialogue = _extract_dialogue(paragraphs)
    issues.extend(_dialogue_repetition_issues(dialogue))
    issues.extend(_voice_differentiation_issues(dialogue))
    issues.extend(_object_causality_issues(paragraphs))
    issues.extend(_ending_cadence_issues(chapters))

    issues = _rank_issues(issues)
    salvage = _salvage_passages(scenes)
    summary = _summary(chapters, scenes, paragraphs, dialogue, issues, fullness)
    plan = _ranked_repair_plan(issues, salvage)
    strict_validation = _strict_validation(issues)

    return {
        "schema_version": "draft-doctor-v1",
        "profile": profile,
        "source": str(path),
        "summary": summary,
        "strict_validation": strict_validation,
        "ranked_repair_plan": plan,
        "issues_by_chapter": _issues_by_chapter(issues),
        "issues_by_scene": _issues_by_scene(issues),
        "issues_by_severity": _issues_by_severity(issues),
        "issues": issues,
        "salvage_passages": salvage,
    }


def write_report(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Write JSON and sibling Markdown reports."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return output, markdown_path


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    fullness = summary.get("fullness", {}) if isinstance(summary.get("fullness"), dict) else {}
    lines = [
        "# Draft Quality Repair Report",
        "",
        f"- Profile: `{report.get('profile', 'unknown')}`",
        f"- Chapters: {summary.get('chapter_count', 0)}",
        f"- Scenes: {summary.get('scene_count', 0)}",
        f"- Words: {summary.get('word_count', 0)}",
        f"- Fullness gate: total >= {fullness.get('min_total_words', 0)}, "
        f"chapter >= {fullness.get('min_chapter_words', 0)}, "
        f"scene >= {fullness.get('min_scene_words', 0)}",
        f"- Paragraphs: {summary.get('paragraph_count', 0)}",
        f"- Dialogue lines: {summary.get('dialogue_line_count', 0)}",
        f"- Issues: {summary.get('issue_count', 0)} "
        f"(high {summary.get('high_severity_count', 0)}, medium {summary.get('medium_severity_count', 0)})",
        f"- Strict validation: {'PASS' if report.get('strict_validation', {}).get('pass') else 'FAIL'}",
        "",
        "## Ranked Repair Plan",
    ]
    for item in report.get("ranked_repair_plan", []):
        lines.extend(
            [
                "",
                f"### {item['priority']}. {item['title']}",
                f"- Severity: {item['severity']}",
                f"- Issues: {item['issue_count']}",
                f"- Action: {item['action']}",
            ]
        )
        if item.get("example"):
            lines.append(f"- Example: {item['example']}")
    lines.extend(["", "## Strongest Salvage Passages"])
    for passage in report.get("salvage_passages", []):
        lines.extend(
            [
                "",
                f"### {passage['label']}",
                f"- Location: Chapter {passage['chapter']}, Scene {passage['scene']} "
                f"({passage['scene_title']})",
                f"- Why it works: {passage['why']}",
                f"> {_one_line(passage['excerpt'], 360)}",
            ]
        )
    lines.extend(["", "## Issues"])
    for issue in report.get("issues", [])[:80]:
        loc = issue.get("location", {})
        lines.extend(
            [
                "",
                f"### {issue['id']} · {issue['category']} · {issue['severity']}",
                f"- Location: Chapter {loc.get('chapter')}, Scene {loc.get('scene')}, "
                f"Paragraph {loc.get('paragraph')}",
                f"- Finding: {issue['message']}",
                f"- Suggested action: {issue['suggestion']}",
            ]
        )
        if issue.get("examples"):
            lines.append(f"- Example: {_one_line(issue['examples'][0], 360)}")
        if issue.get("related_locations"):
            related = ", ".join(
                f"Ch {r.get('chapter')} S{r.get('scene')} P{r.get('paragraph')}"
                for r in issue["related_locations"][:6]
            )
            lines.append(f"- Related: {related}")
    return "\n".join(lines) + "\n"


def _load_chapters(path: Path) -> list[dict[str, Any]]:
    files = _chapter_files(path)
    chapters = []
    for idx, file_path in enumerate(files, start=1):
        text = file_path.read_text(encoding="utf-8")
        title = _chapter_title(text, file_path, idx)
        scenes = _split_scenes(text, idx, title)
        chapters.append({"number": idx, "title": title, "path": str(file_path), "text": text, "scenes": scenes})
    return chapters


def _chapter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    chapter_files = sorted(
        p
        for p in path.glob("chapter_*.md")
        if not p.name.endswith("_audit.md") and not p.name.endswith("_reader.md")
    )
    if not chapter_files:
        chapter_files = sorted(p for p in path.glob("*.md") if "audit" not in p.stem and "reader" not in p.stem)
    if not chapter_files:
        raise FileNotFoundError(f"No markdown chapter files found under {path}")
    return chapter_files


def _chapter_title(text: str, file_path: Path, fallback: int) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return _plain(line.lstrip("#").strip()) or file_path.stem
    return file_path.stem.replace("_", " ").title() or f"Chapter {fallback}"


def _split_scenes(text: str, chapter: int, chapter_title: str) -> list[SceneRef]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    scenes: list[SceneRef] = []
    current_title = "Opening"
    current_blocks: list[str] = []

    def flush() -> None:
        nonlocal current_blocks
        paragraphs: list[ParagraphRef] = []
        for block in current_blocks:
            if block.startswith("#"):
                continue
            cleaned = _plain(block)
            if cleaned:
                paragraphs.append(
                    ParagraphRef(
                        chapter=chapter,
                        chapter_title=chapter_title,
                        scene=len(scenes) + 1,
                        scene_title=current_title,
                        paragraph=len(paragraphs) + 1,
                        text=cleaned,
                    )
                )
        if paragraphs:
            scenes.append(
                SceneRef(
                    chapter=chapter,
                    chapter_title=chapter_title,
                    scene=len(scenes) + 1,
                    scene_title=current_title,
                    paragraphs=tuple(paragraphs),
                )
            )
        current_blocks = []

    for block in blocks:
        if block.startswith("## "):
            flush()
            current_title = _plain(block.lstrip("#").strip()) or f"Scene {len(scenes) + 1}"
            continue
        if re.fullmatch(r"(\* *\* *\*|-{3,}|<hr[^>]*>)", block, re.I):
            flush()
            current_title = f"Scene {len(scenes) + 1}"
            continue
        current_blocks.append(block)
    flush()
    if not scenes:
        cleaned = _plain(text)
        if cleaned:
            paragraph = ParagraphRef(chapter, chapter_title, 1, "Opening", 1, cleaned)
            scenes.append(SceneRef(chapter, chapter_title, 1, "Opening", (paragraph,)))
    return scenes


def _plain(markdown: str) -> str:
    text = html.unescape(markdown)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`>#]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fullness_settings(
    profile: str,
    *,
    min_chapter_words: int | None,
    min_scene_words: int | None,
    min_total_words: int | None,
) -> dict[str, int]:
    defaults = PROFILE_FULLNESS_DEFAULTS.get(profile, {})
    settings = {
        "min_chapter_words": int(defaults.get("min_chapter_words", 0)),
        "min_scene_words": int(defaults.get("min_scene_words", 0)),
        "min_total_words": int(defaults.get("min_total_words", 0)),
    }
    overrides = {
        "min_chapter_words": min_chapter_words,
        "min_scene_words": min_scene_words,
        "min_total_words": min_total_words,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = max(0, int(value))
    return settings


def _fullness_issues(
    chapters: list[dict[str, Any]],
    scenes: list[SceneRef],
    fullness: dict[str, int],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    first_ref = _first_paragraph_ref(chapters)
    total_words = sum(_word_count(scene.text) for scene in scenes)
    min_total = int(fullness.get("min_total_words", 0))
    if first_ref and min_total and total_words < min_total:
        issues.append(
            _issue(
                category="draft_fullness",
                severity="high",
                ref=first_ref,
                message=f"Draft is under full-book scale: {total_words} words, expected at least {min_total}.",
                suggestion="Expand the draft with plot-bearing scenes before style repair; do not let a synopsis-length book pass strict validation.",
                examples=[first_ref.text],
                evidence={"word_count": total_words, "min_total_words": min_total, "kind": "under_min_total_words"},
            )
        )

    min_chapter = int(fullness.get("min_chapter_words", 0))
    for chapter in chapters:
        chapter_scenes = chapter.get("scenes", [])
        chapter_words = sum(_word_count(scene.text) for scene in chapter_scenes)
        ref = chapter_scenes[0].paragraphs[0] if chapter_scenes and chapter_scenes[0].paragraphs else first_ref
        if ref and min_chapter and chapter_words < min_chapter:
            issues.append(
                _issue(
                    category="draft_fullness",
                    severity="high",
                    ref=ref,
                    message=(
                        f"Chapter {chapter.get('number')} is under chapter scale: "
                        f"{chapter_words} words, expected at least {min_chapter}."
                    ),
                    suggestion="Expand the chapter around irreversible events, travel pressure, discovery, conflict, and consequence.",
                    examples=[ref.text],
                    evidence={
                        "chapter_word_count": chapter_words,
                        "min_chapter_words": min_chapter,
                        "kind": "under_min_chapter_words",
                    },
                )
            )

    min_scene = int(fullness.get("min_scene_words", 0))
    for scene in scenes:
        scene_words = _word_count(scene.text)
        if scene.paragraphs and min_scene and scene_words < min_scene:
            issues.append(
                _issue(
                    category="draft_fullness",
                    severity="medium",
                    ref=scene.paragraphs[0],
                    message=(
                        f"Scene is under developed-scene scale: {scene_words} words, "
                        f"expected at least {min_scene}."
                    ),
                    suggestion="Add concrete turns inside the scene: setup, pressure, action, information gained, and changed conditions.",
                    examples=[_one_line(scene.text, 420)],
                    evidence={
                        "scene_word_count": scene_words,
                        "min_scene_words": min_scene,
                        "kind": "under_min_scene_words",
                    },
                )
            )
    return issues


def _first_paragraph_ref(chapters: list[dict[str, Any]]) -> ParagraphRef | None:
    for chapter in chapters:
        for scene in chapter.get("scenes", []):
            if scene.paragraphs:
                return scene.paragraphs[0]
    return None


def _placeholder_issues(paragraphs: list[ParagraphRef]) -> list[dict[str, Any]]:
    issues = []
    for p in paragraphs:
        for name, pattern, reason in PLACEHOLDER_PATTERNS:
            if re.search(pattern, p.text):
                issues.append(
                    _issue(
                        category="placeholder_continuity_lint",
                        severity="high",
                        ref=p,
                        message=f"{name}: {reason}",
                        suggestion="Repair the generated location/object phrase before revising style; this is a mechanical believability break.",
                        examples=[p.text],
                        evidence={"pattern": name},
                    )
                )
        if re.match(r"^[a-z]", p.text) and _object_hits(p.text):
            issues.append(
                _issue(
                    category="placeholder_continuity_lint",
                    severity="high",
                    ref=p,
                    message="Lowercase object-list paragraph opening suggests an unpolished template fragment.",
                    suggestion="Rewrite the paragraph as a concrete action beat with a grammatical subject.",
                    examples=[p.text],
                    evidence={"pattern": "lowercase_object_list_opening"},
                )
            )
    return issues


def _paragraph_repetition_issues(paragraphs: list[ParagraphRef]) -> list[dict[str, Any]]:
    by_norm: dict[str, list[ParagraphRef]] = defaultdict(list)
    candidates = [p for p in paragraphs if _word_count(p.text) >= 14]
    for p in candidates:
        by_norm[_normalize(p.text)].append(p)

    issues = []
    seen_pairs: set[tuple[int, int, int, int, int, int]] = set()
    for group in by_norm.values():
        if len(group) > 1:
            base = group[0]
            issues.append(
                _issue(
                    category="repetition_graph",
                    severity="high",
                    ref=base,
                    message=f"Duplicated paragraph appears {len(group)} times.",
                    suggestion="Keep one use only if it marks escalation; otherwise replace repeats with new plot information.",
                    examples=[base.text],
                    related_locations=[_loc(g) for g in group[1:]],
                    evidence={"repeat_count": len(group), "kind": "duplicate_paragraph"},
                )
            )

    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            if left.chapter == right.chapter and left.scene == right.scene:
                continue
            key = (left.chapter, left.scene, left.paragraph, right.chapter, right.scene, right.paragraph)
            if key in seen_pairs:
                continue
            if _length_ratio(left.text, right.text) < 0.62:
                continue
            score = SequenceMatcher(None, _normalize(left.text), _normalize(right.text)).ratio()
            if score >= 0.86:
                seen_pairs.add(key)
                issues.append(
                    _issue(
                        category="repetition_graph",
                        severity="medium",
                        ref=left,
                        message=f"Near-duplicated paragraph recurs with {score:.0%} textual similarity.",
                        suggestion="Merge the repeated thematic function or turn the later recurrence into a concrete escalation.",
                        examples=[left.text, right.text],
                        related_locations=[_loc(right)],
                        evidence={"similarity": round(score, 4), "kind": "near_duplicate_paragraph"},
                    )
                )
                if len(issues) > 80:
                    return issues
    return issues


def _scene_opening_repetition_issues(scenes: list[SceneRef]) -> list[dict[str, Any]]:
    openings = [(scene, scene.paragraphs[0]) for scene in scenes if scene.paragraphs]
    issues = []
    for i, (scene_a, para_a) in enumerate(openings):
        for scene_b, para_b in openings[i + 1 :]:
            if _length_ratio(para_a.text, para_b.text) < 0.55:
                continue
            score = SequenceMatcher(None, _normalize(para_a.text[:240]), _normalize(para_b.text[:240])).ratio()
            if score >= 0.72:
                issues.append(
                    _issue(
                        category="repetition_graph",
                        severity="medium",
                        ref=para_a,
                        message=f"Scene opening resembles another scene opening ({score:.0%} similarity).",
                        suggestion="Make the opening name the scene's unique pressure: who acts, what changes, and what cannot be undone.",
                        examples=[para_a.text, para_b.text],
                        related_locations=[_loc(para_b)],
                        evidence={"kind": "repeated_scene_opening", "similarity": round(score, 4)},
                    )
                )
                break
    return issues


def _maxim_repetition_issues(paragraphs: list[ParagraphRef]) -> list[dict[str, Any]]:
    sentence_refs: list[tuple[ParagraphRef, str]] = []
    for p in paragraphs:
        for sent in _sentences(p.text):
            if _is_aphoristic(sent):
                sentence_refs.append((p, sent))

    issues = []
    normalized = defaultdict(list)
    for ref, sentence in sentence_refs:
        normalized[_maxim_key(sentence)].append((ref, sentence))
    for group in normalized.values():
        if len(group) < 2:
            continue
        ref, sentence = group[0]
        issues.append(
            _issue(
                category="repetition_graph",
                severity="medium",
                ref=ref,
                message=f"Aphoristic/thematic sentence pattern recurs {len(group)} times.",
                suggestion="Convert repeated maxims into scene-specific action, contradiction, or cost.",
                examples=[sentence],
                related_locations=[_loc(g[0]) for g in group[1:]],
                evidence={"kind": "repeated_maxim", "repeat_count": len(group)},
            )
        )

    by_chapter = Counter(ref.chapter for ref, _ in sentence_refs)
    para_by_chapter = Counter(p.chapter for p in paragraphs)
    for chapter, count in by_chapter.items():
        if count >= max(6, para_by_chapter[chapter] // 5):
            ref = next(ref for ref, _ in sentence_refs if ref.chapter == chapter)
            issues.append(
                _issue(
                    category="repetition_graph",
                    severity="low",
                    ref=ref,
                    message=f"Chapter contains a high load of aphoristic sentences ({count}).",
                    suggestion="Reserve the elevated cadence for earned turns; replace filler maxims with decisions, pursuit, or changed information.",
                    examples=[sent for r, sent in sentence_refs if r.chapter == chapter][:2],
                    evidence={"kind": "maxim_density", "count": count},
                )
            )
    return issues


def _scene_causality_issues(scenes: list[SceneRef]) -> list[dict[str, Any]]:
    issues = []
    for scene in scenes:
        stats = _scene_causality_stats(scene.text)
        if stats["score"] < 0.28:
            ref = scene.paragraphs[0]
            issues.append(
                _issue(
                    category="scene_causality",
                    severity="high" if not stats["state_changes"] else "medium",
                    ref=ref,
                    message="Scene does not establish a clear irreversible narrative effect.",
                    suggestion="Add or foreground one concrete change: new knowledge, changed location, raised danger, custody shift, warning sent, or relationship altered.",
                    examples=[_one_line(scene.text, 420)],
                    evidence=stats,
                )
            )
        elif stats["abstract_density"] > 0.04 and stats["event_sentence_ratio"] < 0.34:
            ref = scene.paragraphs[0]
            issues.append(
                _issue(
                    category="scene_causality",
                    severity="medium",
                    ref=ref,
                    message="Scene has some state change, but abstract explanation is doing more work than events.",
                    suggestion="Move the theme behind observable pressure: tracks found, witness contradicted, prisoner moved, warning delayed, or danger worsened.",
                    examples=[_one_line(scene.text, 420)],
                    evidence=stats,
                )
            )
    return issues


def _register_balance_issues(paragraphs: list[ParagraphRef], scenes: list[SceneRef]) -> list[dict[str, Any]]:
    issues = []
    for p in paragraphs:
        analytical_hits = [
            name
            for name, pattern in MODERN_ANALYTICAL_PATTERNS
            if re.search(pattern, p.text, re.IGNORECASE)
        ]
        if analytical_hits:
            issues.append(
                _issue(
                    category="tolkien_register_balance",
                    severity="medium",
                    ref=p,
                    message="Modern analytical or process language breaks the secondary-world register.",
                    suggestion=(
                        "Express the reasoning through observed signs, remembered lore, disagreement, "
                        "or a concrete decision made under pressure."
                    ),
                    examples=[p.text],
                    evidence={"kind": "modern_analytical_diction", "patterns": analytical_hits},
                )
            )
        words = _tokens(p.text)
        if len(words) < 16:
            continue
        abstract_count = sum(1 for w in words if w in ABSTRACT_TERMS)
        physical_count = sum(1 for w in words if w in PHYSICAL_TERMS)
        causal_count = sum(1 for w in words if w in CAUSAL_VERBS)
        if abstract_count >= 4 and abstract_count > physical_count + causal_count:
            issues.append(
                _issue(
                    category="tolkien_register_balance",
                    severity="medium",
                    ref=p,
                    message="Abstract thematic cluster arrives before enough physical causality earns it.",
                    suggestion="Keep the register elevated, but let weather, craft, motion, speech, or damage carry the moral weight.",
                    examples=[p.text],
                    evidence={
                        "abstract_terms": _term_counts(words, ABSTRACT_TERMS),
                        "physical_terms": _term_counts(words, PHYSICAL_TERMS),
                        "causal_verbs": _term_counts(words, CAUSAL_VERBS),
                    },
                )
            )

    for scene in scenes:
        stats = _scene_causality_stats(scene.text)
        if stats["abstract_density"] >= 0.055 and stats["concrete_density"] < 0.035:
            issues.append(
                _issue(
                    category="tolkien_register_balance",
                    severity="medium",
                    ref=scene.paragraphs[0],
                    message="Scene-level register leans heavily on moral nouns with little concrete craft or world detail.",
                    suggestion="Add specific place, tool, track, witness, weather, or bodily action; remove explanatory theme sentences that repeat the same burden.",
                    examples=[_one_line(scene.text, 360)],
                    evidence=stats,
                )
            )
    return issues


def _extract_dialogue(paragraphs: list[ParagraphRef]) -> list[DialogueRef]:
    refs: list[DialogueRef] = []
    quote_re = re.compile(r"(?:[\"“](?P<double>[^\"”]+)[\"”]|(?<!\w)'(?P<single>[^']{3,})'(?!\w))")
    for p in paragraphs:
        for match in quote_re.finditer(p.text):
            line = (match.group("double") or match.group("single") or "").strip()
            if not line:
                continue
            speaker = _infer_speaker(p.text, match.start(), match.end())
            refs.append(DialogueRef(p.chapter, p.scene, p.paragraph, speaker, line))
    return refs


def _infer_speaker(paragraph: str, start: int, end: int) -> str:
    window = paragraph[max(0, start - 90) : min(len(paragraph), end + 120)]
    for speaker in ("Gandalf", "Aragorn", "Gollum"):
        if re.search(rf"\b(said|asked|answered|whispered|muttered|cried)\s+{speaker}\b", window, re.I):
            return speaker
        if re.search(rf"\b{speaker}\s+(said|asked|answered|whispered|muttered|cried)\b", window, re.I):
            return speaker
    return "Unknown"


def _dialogue_repetition_issues(dialogue: list[DialogueRef]) -> list[dict[str, Any]]:
    issues = []
    by_norm: dict[str, list[DialogueRef]] = defaultdict(list)
    for ref in dialogue:
        if _word_count(ref.line) >= 5:
            by_norm[_normalize(ref.line)].append(ref)
    for group in by_norm.values():
        if len(group) >= 2:
            first = group[0]
            issues.append(
                _dialogue_issue(
                    category="voice_differentiation",
                    severity="medium",
                    ref=first,
                    message=f"Dialogue line is recycled {len(group)} times.",
                    suggestion="Only repeat dialogue when the repetition escalates meaning; otherwise give the speaker a scene-specific want or tactic.",
                    examples=[first.line],
                    related_locations=[_dialogue_loc(g) for g in group[1:]],
                    evidence={"kind": "recycled_dialogue", "repeat_count": len(group)},
                )
            )
    for i, left in enumerate(dialogue):
        if _word_count(left.line) < 7:
            continue
        for right in dialogue[i + 1 :]:
            if left.speaker == right.speaker and left.chapter == right.chapter:
                continue
            if _length_ratio(left.line, right.line) < 0.62:
                continue
            score = SequenceMatcher(None, _normalize(left.line), _normalize(right.line)).ratio()
            if score >= 0.82:
                issues.append(
                    _dialogue_issue(
                        category="voice_differentiation",
                        severity="medium",
                        ref=left,
                        message=f"Dialogue is near-recycled ({score:.0%} similarity).",
                        suggestion="Rewrite the second line to reveal a new tactic, emotion, or piece of knowledge.",
                        examples=[left.line, right.line],
                        related_locations=[_dialogue_loc(right)],
                        evidence={"kind": "near_recycled_dialogue", "similarity": round(score, 4)},
                    )
                )
                break
    return issues


def _voice_differentiation_issues(dialogue: list[DialogueRef]) -> list[dict[str, Any]]:
    issues = []
    for ref in dialogue:
        if ref.speaker in {"Gandalf", "Aragorn"} and _is_aphoristic(ref.line):
            words = _tokens(ref.line)
            abstract_count = sum(1 for w in words if w in ABSTRACT_TERMS)
            if abstract_count >= 1 or _word_count(ref.line) >= 9:
                issues.append(
                    _dialogue_issue(
                        category="voice_differentiation",
                        severity="low",
                        ref=ref,
                        message=f"{ref.speaker} speaks in a portable maxim that could fit another wise character.",
                        suggestion="Anchor the line in the speaker's role: Gandalf presses hidden knowledge and urgency; Aragorn weighs craft, endurance, and practical risk.",
                        examples=[ref.line],
                        evidence={"kind": "interchangeable_aphorism", "abstract_terms": _term_counts(words, ABSTRACT_TERMS)},
                    )
                )

    gollum_formulas: dict[str, list[DialogueRef]] = defaultdict(list)
    for ref in dialogue:
        if ref.speaker != "Gollum":
            continue
        key = None
        lower = ref.line.lower()
        if "baggins" in lower and ("bites" in lower or "thief" in lower or "shire" in lower):
            key = "baggins_shire_wound"
        if "kind hands" in lower and ("tie" in lower or "knots" in lower):
            key = "kind_hands_tie_knots"
        if key:
            gollum_formulas[key].append(ref)
    for key, group in gollum_formulas.items():
        if len(group) >= 2:
            issues.append(
                _dialogue_issue(
                    category="voice_differentiation",
                    severity="medium",
                    ref=group[0],
                    message=f"Gollum formula `{key}` repeats without clear escalation.",
                    suggestion="Keep compulsive repetition only when the scene changes what the repetition costs or reveals.",
                    examples=[g.line for g in group[:3]],
                    related_locations=[_dialogue_loc(g) for g in group[1:]],
                    evidence={"kind": "gollum_formula_repeat", "formula": key, "repeat_count": len(group)},
                )
            )
    return issues


def _object_causality_issues(paragraphs: list[ParagraphRef]) -> list[dict[str, Any]]:
    by_object_set: dict[tuple[str, ...], list[ParagraphRef]] = defaultdict(list)
    issues = []
    for p in paragraphs:
        hits = _canonical_object_hits(p.text)
        if len(hits) < 2:
            continue
        tokens = set(_tokens(p.text))
        causal_hits = tokens & CAUSAL_VERBS
        if not causal_hits:
            by_object_set[tuple(sorted(hits))].append(p)
            issues.append(
                _issue(
                    category="object_causality",
                    severity="medium",
                    ref=p,
                    message=f"Object inventory appears decorative rather than causal: {', '.join(sorted(hits))}.",
                    suggestion="Make one object change the scene: block escape, expose a lie, alter a route, trigger a question, or force a decision.",
                    examples=[p.text],
                    evidence={"objects": sorted(hits), "causal_verbs": []},
                )
            )
    for objects, group in by_object_set.items():
        if len(group) >= 3:
            issues.append(
                _issue(
                    category="object_causality",
                    severity="medium",
                    ref=group[0],
                    message=f"Same symbolic object set recurs {len(group)} times: {', '.join(objects)}.",
                    suggestion="Retire repeated inventory lists unless the objects perform a new plot function.",
                    examples=[g.text for g in group[:2]],
                    related_locations=[_loc(g) for g in group[1:]],
                    evidence={"objects": list(objects), "repeat_count": len(group)},
                )
            )
    return issues


def _ending_cadence_issues(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []
    for chapter in chapters:
        paragraphs = [p for scene in chapter["scenes"] for p in scene.paragraphs]
        all_sentences: list[tuple[int, ParagraphRef, str, str]] = []
        for p in paragraphs:
            for sentence in _sentences(p.text):
                for name, pattern in ENDING_PATTERNS:
                    if re.search(pattern, sentence, re.I):
                        all_sentences.append((len(all_sentences), p, sentence, name))
        if len(all_sentences) <= 1:
            continue
        early = [
            (idx, p, sentence, name)
            for idx, p, sentence, name in all_sentences
            if p.paragraph < max(1, len(paragraphs) - 3)
        ]
        if early or len(all_sentences) >= 3:
            idx, ref, sentence, name = all_sentences[0]
            issues.append(
                _issue(
                    category="ending_cadence_overload",
                    severity="medium",
                    ref=ref,
                    message=f"Chapter uses {len(all_sentences)} final-sounding cadence markers before the true close.",
                    suggestion="Save closure cadence for the final turn; earlier scenes should end on changed action, withheld information, or immediate pressure.",
                    examples=[item[2] for item in all_sentences[:4]],
                    evidence={
                        "cadence_count": len(all_sentences),
                        "early_cadence_count": len(early),
                        "patterns": [item[3] for item in all_sentences],
                    },
                )
            )
    return issues


def _scene_causality_stats(text: str) -> dict[str, Any]:
    words = _tokens(text)
    word_count = max(1, len(words))
    sentences = _sentences(text)
    state_changes = []
    for name, terms in STATE_CHANGE_TERMS.items():
        hits = sorted(set(words) & terms)
        if hits:
            state_changes.append({"type": name, "terms": hits[:8]})
    eventful = [s for s in sentences if set(_tokens(s)) & CAUSAL_VERBS]
    abstract = sum(1 for w in words if w in ABSTRACT_TERMS)
    concrete = sum(1 for w in words if w in PHYSICAL_TERMS or w in CAUSAL_VERBS)
    score = min(
        1.0,
        0.18 * len(state_changes)
        + 0.45 * (len(eventful) / max(1, len(sentences)))
        + 0.18 * min(1.0, concrete / 18)
        - 0.16 * min(1.0, abstract / 16),
    )
    return {
        "score": round(max(0.0, score), 4),
        "state_changes": state_changes,
        "event_sentence_ratio": round(len(eventful) / max(1, len(sentences)), 4),
        "abstract_density": round(abstract / word_count, 4),
        "concrete_density": round(concrete / word_count, 4),
        "abstract_terms": _term_counts(words, ABSTRACT_TERMS),
        "concrete_terms": _term_counts(words, PHYSICAL_TERMS | CAUSAL_VERBS),
    }


def _salvage_passages(scenes: list[SceneRef]) -> list[dict[str, Any]]:
    scored: list[tuple[float, str, SceneRef, dict[str, Any]]] = []
    for scene in scenes:
        stats = _scene_causality_stats(scene.text)
        lower = scene.text.lower() + " " + scene.scene_title.lower()
        label = "Concrete state-change scene"
        bonus = 0.0
        if "marsh" in lower and any(t in lower for t in ("caught", "bound", "capture", "rope", "seized")):
            label = "Marsh capture"
            bonus = 0.35
        elif any(t in lower for t in ("question", "asked", "answered", "testimony")) and any(
            t in lower for t in ("gollum", "gandalf", "wood-elves", "mirkwood")
        ):
            label = "Guarded questioning"
            bonus = 0.3
        score = stats["score"] + bonus
        scored.append((score, label, scene, stats))
    scored.sort(key=lambda item: item[0], reverse=True)

    passages = []
    used_labels: set[str] = set()
    for score, label, scene, stats in scored:
        if len(passages) >= 6:
            break
        if label in used_labels and label != "Concrete state-change scene":
            continue
        used_labels.add(label)
        changes = ", ".join(change["type"] for change in stats["state_changes"]) or "observable pressure"
        passages.append(
            {
                "label": label,
                "chapter": scene.chapter,
                "scene": scene.scene,
                "scene_title": scene.scene_title,
                "score": round(score, 4),
                "why": f"Strongest local material because it carries {changes} through concrete action.",
                "excerpt": _one_line(scene.text, 520),
                "evidence": stats,
            }
        )
    return passages


def _summary(
    chapters: list[dict[str, Any]],
    scenes: list[SceneRef],
    paragraphs: list[ParagraphRef],
    dialogue: list[DialogueRef],
    issues: list[dict[str, Any]],
    fullness: dict[str, int],
) -> dict[str, Any]:
    by_category = Counter(issue["category"] for issue in issues)
    by_severity = Counter(issue["severity"] for issue in issues)
    chapter_word_counts = {
        f"chapter_{int(chapter.get('number') or 0):02d}": sum(_word_count(scene.text) for scene in chapter.get("scenes", []))
        for chapter in chapters
    }
    scene_word_counts = {
        f"chapter_{scene.chapter:02d}/scene_{scene.scene:02d}": _word_count(scene.text)
        for scene in scenes
    }
    word_count = sum(scene_word_counts.values())
    return {
        "chapter_count": len(chapters),
        "scene_count": len(scenes),
        "paragraph_count": len(paragraphs),
        "dialogue_line_count": len(dialogue),
        "word_count": word_count,
        "avg_chapter_words": round(word_count / max(1, len(chapters)), 2),
        "avg_scene_words": round(word_count / max(1, len(scenes)), 2),
        "chapter_word_counts": chapter_word_counts,
        "scene_word_counts": scene_word_counts,
        "fullness": dict(fullness),
        "issue_count": len(issues),
        "high_severity_count": by_severity["high"],
        "medium_severity_count": by_severity["medium"],
        "low_severity_count": by_severity["low"],
        "issue_counts_by_category": dict(sorted(by_category.items())),
    }


def _strict_validation(issues: list[dict[str, Any]]) -> dict[str, Any]:
    blocking = [
        issue
        for issue in issues
        if issue["category"] in STRICT_BLOCKING_CATEGORIES
        and issue["severity"] in STRICT_BLOCKING_SEVERITIES
    ]
    counts_by_category = Counter(issue["category"] for issue in blocking)
    counts_by_severity = Counter(issue["severity"] for issue in blocking)
    return {
        "pass": not blocking,
        "blocking_issue_count": len(blocking),
        "blocking_counts_by_category": dict(sorted(counts_by_category.items())),
        "blocking_counts_by_severity": dict(sorted(counts_by_severity.items())),
        "blocking_issue_ids": [issue["id"] for issue in blocking[:50]],
        "rules": {
            "blocking_categories": sorted(STRICT_BLOCKING_CATEGORIES),
            "blocking_severities": sorted(STRICT_BLOCKING_SEVERITIES),
            "low_severity_findings": "reported as revision guidance but non-blocking",
        },
    }


def _ranked_repair_plan(issues: list[dict[str, Any]], salvage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = [
        (
            "draft_fullness",
            "Expand synopsis-length chapters into real draft scale",
            "Do not revise around a loophole: bring total, chapter, and scene word counts above strict thresholds with plot-bearing material.",
        ),
        (
            "placeholder_continuity_lint",
            "Fix mechanical placeholders and location artifacts",
            "Repair unreplaced tokens and malformed Road/object phrases before any line edit; they break secondary-world belief immediately.",
        ),
        (
            "repetition_graph",
            "Collapse duplicated thematic blocks",
            "Remove duplicate paragraphs, repeated scene openings, recycled maxims, and repeated dialogue unless each recurrence visibly escalates plot.",
        ),
        (
            "scene_causality",
            "Give low-causality scenes an irreversible effect",
            "For each flagged scene, state the concrete change in knowledge, location, danger, custody, warning, or relationship, then rewrite around that change.",
        ),
        (
            "voice_differentiation",
            "Separate Gandalf, Aragorn, and Gollum in dialogue",
            "Cut interchangeable aphorisms; give Gandalf urgent hidden knowledge, Aragorn practical craft and endurance, and Gollum compulsive but escalating repetitions.",
        ),
        (
            "object_causality",
            "Make symbolic objects alter plot",
            "Keep repeated objects only when they block, reveal, mislead, force a route, or change a decision.",
        ),
        (
            "tolkien_register_balance",
            "Earn elevated register through physical causality",
            "Replace clusters of mercy/hope/warning/truth explanation with action, weather, craft, place, and cost.",
        ),
        (
            "ending_cadence_overload",
            "Reserve closure cadence for actual endings",
            "Remove final-sounding sentences from mid-scene and mid-chapter positions; end those passages on pressure or changed state.",
        ),
    ]
    severity_weight = {"high": 3, "medium": 2, "low": 1}
    plan = []
    for category, title, action in definitions:
        cat_issues = [i for i in issues if i["category"] == category]
        if not cat_issues:
            continue
        score = sum(severity_weight.get(i["severity"], 1) for i in cat_issues)
        first = cat_issues[0]
        plan.append(
            {
                "priority": 0,
                "category": category,
                "title": title,
                "severity": first["severity"],
                "issue_count": len(cat_issues),
                "score": score,
                "action": action,
                "example": _one_line((first.get("examples") or [""])[0], 260),
                "issue_ids": [i["id"] for i in cat_issues[:12]],
            }
        )
    if salvage:
        plan.append(
            {
                "priority": 0,
                "category": "salvage_passages",
                "title": "Use strongest passages as revision templates",
                "severity": "info",
                "issue_count": len(salvage),
                "score": 1,
                "action": "Model weak scenes on the strongest passages: concrete pressure first, theme as consequence.",
                "example": f"{salvage[0]['label']}: {salvage[0]['excerpt'][:220]}",
                "issue_ids": [],
            }
        )
    plan.sort(key=lambda item: (-item["score"], item["category"]))
    for idx, item in enumerate(plan, start=1):
        item["priority"] = idx
    return plan


def _issues_by_chapter(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        grouped[str(issue["location"]["chapter"])].append(issue)
    return dict(sorted(grouped.items(), key=lambda item: int(item[0])))


def _issues_by_scene(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        loc = issue["location"]
        grouped[f"chapter_{int(loc.get('chapter') or 0):02d}/scene_{int(loc.get('scene') or 0):02d}"].append(issue)
    return dict(sorted(grouped.items()))


def _issues_by_severity(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        grouped[str(issue["severity"])].append(issue)
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    return dict(sorted(grouped.items(), key=lambda item: order.get(item[0], 99)))


def _rank_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weight = {"high": 0, "medium": 1, "low": 2, "info": 3}
    issues = sorted(
        issues,
        key=lambda issue: (
            weight.get(issue["severity"], 4),
            issue["location"].get("chapter") or 0,
            issue["location"].get("scene") or 0,
            issue["location"].get("paragraph") or 0,
            issue["category"],
        ),
    )
    for idx, issue in enumerate(issues, start=1):
        issue["id"] = f"DQ{idx:04d}"
    return issues


def _issue(
    *,
    category: str,
    severity: str,
    ref: ParagraphRef,
    message: str,
    suggestion: str,
    examples: list[str],
    evidence: dict[str, Any],
    related_locations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "",
        "category": category,
        "severity": severity,
        "location": _loc(ref),
        "message": message,
        "suggestion": suggestion,
        "examples": [_one_line(e, 520) for e in examples if e],
        "related_locations": related_locations or [],
        "evidence": evidence,
    }


def _dialogue_issue(
    *,
    category: str,
    severity: str,
    ref: DialogueRef,
    message: str,
    suggestion: str,
    examples: list[str],
    evidence: dict[str, Any],
    related_locations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "",
        "category": category,
        "severity": severity,
        "location": _dialogue_loc(ref),
        "message": message,
        "suggestion": suggestion,
        "examples": [_one_line(e, 520) for e in examples if e],
        "related_locations": related_locations or [],
        "evidence": evidence,
    }


def _loc(ref: ParagraphRef) -> dict[str, Any]:
    return {
        "chapter": ref.chapter,
        "chapter_title": ref.chapter_title,
        "scene": ref.scene,
        "scene_title": ref.scene_title,
        "paragraph": ref.paragraph,
    }


def _dialogue_loc(ref: DialogueRef) -> dict[str, Any]:
    return {
        "chapter": ref.chapter,
        "scene": ref.scene,
        "paragraph": ref.paragraph,
        "speaker": ref.speaker,
    }


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text)]


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _normalize(text: str) -> str:
    tokens = _tokens(text)
    return " ".join(tokens)


def _maxim_key(sentence: str) -> str:
    tokens = _tokens(sentence)
    stripped = [
        "THING"
        if token in ABSTRACT_TERMS or token in PHYSICAL_TERMS or token in {"road", "shire", "gollum", "baggins"}
        else token
        for token in tokens[:18]
    ]
    return " ".join(stripped)


def _is_aphoristic(sentence: str) -> bool:
    words = _tokens(sentence)
    if len(words) < 6 or len(words) > 36:
        return False
    abstract_count = sum(1 for w in words if w in ABSTRACT_TERMS)
    if abstract_count >= 2:
        return True
    return all(pattern.search(sentence) for pattern in APHORISM_PATTERNS)


def _term_counts(words: list[str], terms: set[str]) -> dict[str, int]:
    counts = Counter(w for w in words if w in terms)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12])


def _object_hits(text: str) -> set[str]:
    lower = text.lower()
    hits = set()
    for term in OBJECT_TERMS:
        if re.search(r"(?<![\w-])" + re.escape(term) + r"(?![\w-])", lower):
            hits.add(term)
    return hits


def _canonical_object_hits(text: str) -> set[str]:
    canonical = {
        "branches": "branch",
        "folded letters": "letter",
        "letters": "letter",
        "maps": "map",
        "lamps": "lamp",
        "weathered cloak": "cloak",
        "muddy water": "water",
    }
    return {canonical.get(hit, hit) for hit in _object_hits(text)}


def _length_ratio(a: str, b: str) -> float:
    la = max(1, len(a))
    lb = max(1, len(b))
    return min(la, lb) / max(la, lb)


def _one_line(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"
