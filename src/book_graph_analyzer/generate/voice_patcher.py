"""Targeted dialogue voice patching for generated scenes."""

from __future__ import annotations

import re
from collections import defaultdict

from .models import Scene
from ..voice.dialogue import extract_dialogue
from ..voice.profile import CharacterVoiceProfile


class VoicePatcher:
    """Patch character dialogue against canonical voice profiles.

    This component only rewrites quoted dialogue lines and never rewrites narration.
    """

    DIALOGUE_PATCH_PROMPT = """The following dialogue lines were written for {character}.
Their established voice profile:
- Formality: {formality_score:.2f}/1.00
- Contraction rate: {contraction_rate:.3f}
- Typical sentence length: {avg_sentence_len:.1f} words
- Signature markers: {signature_markers}

Rewrite ONLY these dialogue lines to match this voice profile.
Do not change any narration. Keep line order. Return exactly one rewritten line per input line.

DIALOGUE TO PATCH:
{dialogue_blocks}
"""

    _QUOTE_PATTERN = re.compile(r'"([^"]+)"|\u201c([^\u201d]+)\u201d')

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def estimate_max_deviation(
        self,
        scene: Scene,
        voice_profiles: dict[str, CharacterVoiceProfile],
    ) -> float:
        """Estimate max dialogue-profile deviation for the scene."""
        extraction = extract_dialogue(scene.text, passage_id=scene.id)
        by_speaker = defaultdict(list)
        for line in extraction.dialogue_lines:
            if line.speaker:
                by_speaker[line.speaker].append(line)

        max_deviation = 0.0
        for speaker, lines in by_speaker.items():
            profile = voice_profiles.get(speaker)
            if not profile or not lines:
                continue
            max_deviation = max(max_deviation, self._deviation_for_lines(lines, profile))

        return max_deviation

    def patch(
        self,
        scene: Scene,
        voice_profiles: dict[str, CharacterVoiceProfile],
        threshold: float,
    ) -> Scene:
        """Patch scene dialogue when character voice deviation exceeds threshold."""
        if not self.llm:
            return scene

        extraction = extract_dialogue(scene.text, passage_id=scene.id)
        by_speaker = defaultdict(list)
        for line in extraction.dialogue_lines:
            if line.speaker:
                by_speaker[line.speaker].append(line)

        quote_matches = list(self._QUOTE_PATTERN.finditer(scene.text))
        if not quote_matches:
            return scene

        updated_text = scene.text
        replacements: dict[int, str] = {}

        for speaker, lines in by_speaker.items():
            profile = voice_profiles.get(speaker)
            if not profile:
                continue

            deviation = self._deviation_for_lines(lines, profile)
            if deviation < threshold:
                continue

            source_lines = [line.text.strip() for line in lines]
            prompt = self.DIALOGUE_PATCH_PROMPT.format(
                character=speaker,
                formality_score=profile.formality_score,
                contraction_rate=profile.contraction_ratio,
                avg_sentence_len=profile.avg_utterance_length,
                signature_markers=", ".join(profile.distinctive_words[:6]) or "None",
                dialogue_blocks="\n".join(f"- {t}" for t in source_lines),
            )
            patched_raw = self.llm.generate(prompt, temperature=0.4)
            patched_lines = self._parse_patched_lines(patched_raw, expected=len(source_lines))

            for line, patched in zip(lines, patched_lines):
                if line.position < len(quote_matches):
                    replacements[line.position] = patched

        if not replacements:
            return scene

        rebuilt = []
        cursor = 0
        for idx, match in enumerate(quote_matches):
            rebuilt.append(updated_text[cursor:match.start()])
            old_inner = match.group(1) if match.group(1) is not None else match.group(2)
            new_inner = replacements.get(idx, old_inner)
            raw = match.group(0)
            if raw.startswith('"'):
                rebuilt.append(f'"{new_inner}"')
            else:
                rebuilt.append(f"\u201c{new_inner}\u201d")
            cursor = match.end()
        rebuilt.append(updated_text[cursor:])

        scene.text = "".join(rebuilt)
        scene.word_count = len(scene.text.split())
        return scene

    def _deviation_for_lines(self, lines, profile: CharacterVoiceProfile) -> float:
        words = []
        contractions = 0
        for line in lines:
            line_words = [w.strip('.,!?"\'').lower() for w in line.text.split() if w.strip()]
            words.extend(line_words)
            contractions += sum(1 for w in line_words if "'" in w)

        if not words:
            return 0.0

        contraction_rate = contractions / max(1, len(words))
        avg_len = sum(len(line.text.split()) for line in lines) / max(1, len(lines))

        contraction_delta = abs(contraction_rate - profile.contraction_ratio)
        formality_delta = abs((1.0 - min(1.0, contraction_rate * 10.0)) - profile.formality_score)
        length_delta = abs(avg_len - profile.avg_utterance_length) / max(1.0, profile.avg_utterance_length)

        return max(contraction_delta * 2.5, formality_delta, min(1.0, length_delta))

    @staticmethod
    def _parse_patched_lines(raw: str, expected: int) -> list[str]:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            ln = re.sub(r"^[-*\d.\)\s]+", "", ln).strip()
            ln = ln.strip('"\u201c\u201d')
            if ln:
                cleaned.append(ln)
        if len(cleaned) < expected:
            cleaned.extend([cleaned[-1] if cleaned else ""] * (expected - len(cleaned)))
        return cleaned[:expected]
