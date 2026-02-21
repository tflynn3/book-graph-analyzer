"""Sociolinguistic register profiling and drift detection (Issue #47 slice 1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class SociolinguisticRegister(str, Enum):
    NOBLE = "noble"
    RITUAL = "ritual"
    MARTIAL = "martial"
    SCHOLARLY = "scholarly"
    FOLK = "folk"
    PROPHETIC = "prophetic"


_REGISTER_KEYWORDS: dict[str, set[str]] = {
    SociolinguisticRegister.NOBLE: {"lord", "lady", "house", "lineage", "honor", "court", "heir"},
    SociolinguisticRegister.RITUAL: {"vow", "oath", "rite", "sacred", "hallowed", "consecrate", "swear"},
    SociolinguisticRegister.MARTIAL: {"blade", "shield", "march", "captain", "host", "banner", "command"},
    SociolinguisticRegister.SCHOLARLY: {"lore", "chronicle", "tongue", "glyph", "record", "annal", "study"},
    SociolinguisticRegister.FOLK: {"ale", "bread", "home", "garden", "friend", "road", "supper"},
    SociolinguisticRegister.PROPHETIC: {"doom", "fate", "shadow", "star", "vision", "foretold", "destiny"},
}

_ARCHAIC_WORDS = {"thou", "thee", "thy", "hath", "shalt", "ere", "nay", "ye"}


@dataclass
class RegisterProfile:
    dominant_register: str
    confidence: float
    register_scores: dict[str, float]
    formality_score: float
    archaism_rate: float
    contraction_rate: float
    avg_sentence_length: float
    token_count: int


@dataclass
class RegisterDrift:
    baseline_register: str
    current_register: str
    register_shift: float
    formality_shift: float
    archaism_shift: float

    @property
    def severity(self) -> str:
        s = max(self.register_shift, abs(self.formality_shift), abs(self.archaism_shift))
        if s >= 0.45:
            return "high"
        if s >= 0.25:
            return "medium"
        return "low"


class SociolinguisticRegisterClassifier:
    """MVP rule-first classifier for Tolkien sociolinguistic register."""

    _word_re = re.compile(r"[A-Za-z']+")

    def classify(self, text: str) -> RegisterProfile:
        text = text or ""
        words = [w.lower() for w in self._word_re.findall(text)]
        token_count = len(words)
        if token_count == 0:
            return RegisterProfile(
                dominant_register=SociolinguisticRegister.FOLK.value,
                confidence=0.0,
                register_scores={},
                formality_score=0.0,
                archaism_rate=0.0,
                contraction_rate=0.0,
                avg_sentence_length=0.0,
                token_count=0,
            )

        scores: dict[str, float] = {}
        word_set = set(words)
        for reg, kw in _REGISTER_KEYWORDS.items():
            overlap = len(word_set & kw)
            scores[reg.value] = overlap / max(1.0, min(10.0, token_count / 8.0))

        archaism_hits = sum(1 for w in words if w in _ARCHAIC_WORDS)
        contractions = sum(1 for w in words if "'" in w)
        archaism_rate = archaism_hits / token_count
        contraction_rate = contractions / token_count

        sentence_parts = [p for p in re.split(r"[.!?]+", text) if p.strip()]
        avg_sentence_length = token_count / max(1, len(sentence_parts))
        formality_score = max(0.0, min(1.0, (archaism_rate * 5.0) + (1.0 - min(1.0, contraction_rate * 10.0)) * 0.6))

        scores[SociolinguisticRegister.NOBLE.value] += formality_score * 0.2
        scores[SociolinguisticRegister.RITUAL.value] += archaism_rate * 0.8
        scores[SociolinguisticRegister.FOLK.value] += contraction_rate * 0.8
        scores[SociolinguisticRegister.MARTIAL.value] += 0.15 if avg_sentence_length < 14 else 0.0
        scores[SociolinguisticRegister.SCHOLARLY.value] += 0.15 if avg_sentence_length > 18 else 0.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        dominant, top_score = ranked[0]
        total = sum(max(0.0, s) for _, s in ranked)
        confidence = (top_score / total) if total > 0 else 0.0

        return RegisterProfile(
            dominant_register=dominant,
            confidence=round(confidence, 4),
            register_scores={k: round(v, 4) for k, v in ranked},
            formality_score=round(formality_score, 4),
            archaism_rate=round(archaism_rate, 4),
            contraction_rate=round(contraction_rate, 4),
            avg_sentence_length=round(avg_sentence_length, 2),
            token_count=token_count,
        )


def detect_register_drift(baseline: RegisterProfile, current: RegisterProfile) -> RegisterDrift:
    baseline_current_score = current.register_scores.get(baseline.dominant_register, 0.0)
    register_shift = max(0.0, current.confidence - baseline_current_score)
    return RegisterDrift(
        baseline_register=baseline.dominant_register,
        current_register=current.dominant_register,
        register_shift=round(register_shift, 4),
        formality_shift=round(current.formality_score - baseline.formality_score, 4),
        archaism_shift=round(current.archaism_rate - baseline.archaism_rate, 4),
    )

