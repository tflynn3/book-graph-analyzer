"""Lore-depth extraction helpers for artifacts and unresolved references."""

from __future__ import annotations

import re

from ..models.lore_depth import (
    BrokenReference,
    LoreArtifact,
    LoreArtifactType,
    LoreDepthExtractionResult,
)


_ARTIFACT_RE = re.compile(
    r"\b(?P<kind>song|poem|lay|chant|verse|artifact|relic|amulet|ring|sword|crown)\s+(?:of\s+)?(?P<name>[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,4})"
)

_BROKEN_REF_BRACKETS_RE = re.compile(r"\[\[([^\]]+)\]\]")
_BROKEN_REF_QUAL_RE = re.compile(
    r"\b(?P<text>(unknown|unnamed|forgotten)\s+(artifact|relic|song|poem|name|heir))\b",
    re.IGNORECASE,
)


def extract_lore_depth(
    text: str,
    source_book: str | None = None,
    passage_id: str | None = None,
) -> LoreDepthExtractionResult:
    """Extract lore artifacts and unresolved references from raw text."""
    artifacts: list[LoreArtifact] = []
    broken: list[BrokenReference] = []

    for i, m in enumerate(_ARTIFACT_RE.finditer(text)):
        kind = m.group("kind").lower()
        if kind in {"song", "lay", "chant", "verse"}:
            artifact_type = LoreArtifactType.SONG
        elif kind == "poem":
            artifact_type = LoreArtifactType.POEM
        else:
            artifact_type = LoreArtifactType.ARTIFACT

        artifacts.append(
            LoreArtifact(
                id=f"artifact_{passage_id or 'inline'}_{i}",
                name=m.group("name").strip(),
                artifact_type=artifact_type,
                description=m.group(0),
                source_book=source_book,
                passage_id=passage_id,
            )
        )

    for i, m in enumerate(_BROKEN_REF_BRACKETS_RE.finditer(text)):
        mention = m.group(1).strip()
        broken.append(
            BrokenReference(
                id=f"broken_{passage_id or 'inline'}_b{i}",
                mention_text=mention,
                context_text=m.group(0),
                expected_type="unknown",
                source_book=source_book,
                passage_id=passage_id,
                confidence=0.85,
            )
        )

    for i, m in enumerate(_BROKEN_REF_QUAL_RE.finditer(text)):
        broken.append(
            BrokenReference(
                id=f"broken_{passage_id or 'inline'}_q{i}",
                mention_text=m.group("text"),
                context_text=m.group(0),
                expected_type="unresolved_qualifier",
                source_book=source_book,
                passage_id=passage_id,
                confidence=0.65,
            )
        )

    return LoreDepthExtractionResult(artifacts=artifacts, broken_references=broken)
