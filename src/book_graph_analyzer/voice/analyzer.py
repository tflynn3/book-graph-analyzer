"""
Voice Analyzer

Main entry point for character voice analysis.
Extracts dialogue, builds voice profiles per character.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
from collections import Counter
import json

from ..extract.normalizer import normalize_text
from .dialogue import (
    DialogueLine,
    extract_dialogue,
    merge_dialogue_extractions,
)
from .profile import CharacterVoiceProfile


@dataclass
class VoiceAnalysisResult:
    """Complete result of voice analysis."""
    source_file: str
    total_dialogue_lines: int = 0
    total_characters: int = 0
    
    # All dialogue by speaker
    dialogue_by_speaker: dict[str, list[DialogueLine]] = field(default_factory=dict)
    
    # Voice profiles
    profiles: dict[str, CharacterVoiceProfile] = field(default_factory=dict)
    
    # Unattributed dialogue
    unattributed_lines: int = 0
    attribution_rate: float = 0.0
    
    def get_profile(self, character: str) -> Optional[CharacterVoiceProfile]:
        """Get profile for a character (case-insensitive search)."""
        char_lower = character.lower()
        for name, profile in self.profiles.items():
            if name.lower() == char_lower:
                return profile
        return None
    
    def top_speakers(self, n: int = 10) -> list[tuple[str, int]]:
        """Get top N speakers by line count."""
        counts = [(name, len(lines)) for name, lines in self.dialogue_by_speaker.items()
                 if name != "UNKNOWN"]
        return sorted(counts, key=lambda x: -x[1])[:n]


class VoiceAnalyzer:
    """
    Analyzes character voices from text.
    
    Usage:
        analyzer = VoiceAnalyzer()
        result = analyzer.analyze_file("book.txt")
        
        # Get Gandalf's voice profile
        gandalf = result.get_profile("Gandalf")
        print(gandalf.summary())
    """
    
    def __init__(
        self,
        min_lines_for_profile: int = 3,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize the voice analyzer.
        
        Args:
            min_lines_for_profile: Minimum dialogue lines to create a profile
            progress_callback: Optional progress callback
        """
        self.min_lines_for_profile = min_lines_for_profile
        self.progress_callback = progress_callback
    
    def _report_progress(self, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def analyze_text(
        self,
        text: str,
        source_name: str = "text",
    ) -> VoiceAnalysisResult:
        """
        Analyze dialogue in a text.
        
        Args:
            text: The full text to analyze
            source_name: Name of the source
            
        Returns:
            VoiceAnalysisResult with profiles for each character
        """
        result = VoiceAnalysisResult(source_file=source_name)
        text = normalize_text(text)
        
        self._report_progress("Extracting dialogue...")
        
        # Split into paragraphs for processing
        paragraphs = text.split('\n\n')
        
        all_extractions = []
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 0:
                extraction = extract_dialogue(para, passage_id=f"para_{i}")
                if extraction.dialogue_lines:
                    all_extractions.append(extraction)
        
        self._report_progress(f"Found {len(all_extractions)} passages with dialogue")
        
        # Merge by speaker
        dialogue_by_speaker = merge_dialogue_extractions(all_extractions)
        result.dialogue_by_speaker = dialogue_by_speaker
        
        # Count totals
        result.total_dialogue_lines = sum(len(lines) for lines in dialogue_by_speaker.values())
        result.unattributed_lines = len(dialogue_by_speaker.get("UNKNOWN", []))
        
        attributed = result.total_dialogue_lines - result.unattributed_lines
        result.attribution_rate = attributed / result.total_dialogue_lines if result.total_dialogue_lines > 0 else 0
        
        self._report_progress(f"Attribution rate: {result.attribution_rate*100:.1f}%")
        
        # Build word counts for all characters (for distinctive word analysis)
        all_char_words = {}
        for speaker, lines in dialogue_by_speaker.items():
            if speaker == "UNKNOWN":
                continue
            word_counter = Counter()
            for line in lines:
                words = line.text.lower().split()
                word_counter.update(words)
            all_char_words[speaker] = word_counter
        
        # Build profiles
        self._report_progress("Building voice profiles...")
        
        for speaker, lines in dialogue_by_speaker.items():
            if speaker == "UNKNOWN":
                continue
            if len(lines) < self.min_lines_for_profile:
                continue
            
            profile = CharacterVoiceProfile.from_dialogue_lines(
                character_name=speaker,
                lines=lines,
                all_character_words=all_char_words,
            )
            result.profiles[speaker] = profile
        
        result.total_characters = len(result.profiles)
        
        self._report_progress(f"Created {result.total_characters} voice profiles")
        
        return result
    
    def analyze_file(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
    ) -> VoiceAnalysisResult:
        """
        Analyze dialogue in a text file.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
            
        Returns:
            VoiceAnalysisResult
        """
        path = Path(file_path)
        
        self._report_progress(f"Loading {path.name}...")
        
        with open(path, 'r', encoding=encoding) as f:
            text = f.read()
        
        return self.analyze_text(text, source_name=path.name)
    
    def compare_voices(
        self,
        profile1: CharacterVoiceProfile,
        profile2: CharacterVoiceProfile,
    ) -> dict:
        """
        Compare two character voice profiles.
        
        Returns dict with comparison metrics.
        """
        comparison = {
            "character1": profile1.character_name,
            "character2": profile2.character_name,
            "metrics": {},
        }
        
        # Utterance length
        comparison["metrics"]["utterance_length"] = {
            "char1": profile1.avg_utterance_length,
            "char2": profile2.avg_utterance_length,
            "difference": abs(profile1.avg_utterance_length - profile2.avg_utterance_length),
        }
        
        # Question ratio
        comparison["metrics"]["question_ratio"] = {
            "char1": profile1.question_ratio,
            "char2": profile2.question_ratio,
            "difference": abs(profile1.question_ratio - profile2.question_ratio),
        }
        
        # Vocabulary diversity
        comparison["metrics"]["type_token_ratio"] = {
            "char1": profile1.type_token_ratio,
            "char2": profile2.type_token_ratio,
            "difference": abs(profile1.type_token_ratio - profile2.type_token_ratio),
        }
        
        # Formality (contractions)
        comparison["metrics"]["contraction_ratio"] = {
            "char1": profile1.contraction_ratio,
            "char2": profile2.contraction_ratio,
            "difference": abs(profile1.contraction_ratio - profile2.contraction_ratio),
        }
        
        # Shared distinctive words
        shared_distinctive = set(profile1.distinctive_words) & set(profile2.distinctive_words)
        comparison["shared_distinctive_words"] = list(shared_distinctive)
        
        # Overall similarity score (simple average of normalized differences)
        diffs = [
            comparison["metrics"]["utterance_length"]["difference"] / 20,  # Normalize to ~0-1
            comparison["metrics"]["question_ratio"]["difference"],
            comparison["metrics"]["type_token_ratio"]["difference"],
            comparison["metrics"]["contraction_ratio"]["difference"],
        ]
        comparison["similarity_score"] = 1 - (sum(diffs) / len(diffs))
        
        return comparison
    
    def identify_speaker(
        self,
        text: str,
        profiles: dict[str, CharacterVoiceProfile],
        top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Given an unmarked quote, identify the most likely speaker.
        
        The Blindspot Test: can we identify the speaker from the profile alone?
        
        Args:
            text: The dialogue text to identify
            profiles: Dict of character name -> CharacterVoiceProfile
            top_n: Number of candidates to return
            
        Returns:
            List of (character_name, confidence) sorted by confidence desc
        """
        if not profiles:
            return []
        
        # Compute text metrics
        words = text.lower().split()
        word_count = len(words)
        
        # Archaisms
        archaisms_list = {
            "thee", "thou", "thy", "thine", "ye", "hath", "doth", "art", "wast",
            "wherefore", "hither", "thither", "whither", "hence", "thence",
            "ere", "nay", "aye", "yea", "behold", "lo", "alas", "forsooth",
            "methinks", "mayhap", "perchance", "betwixt", "amongst", "whilst",
            "verily", "hark", "hearken", "tarry", "prithee",
        }
        text_archaism_count = sum(1 for w in words if w.strip('.,!?"\'') in archaisms_list)
        text_archaism_rate = (text_archaism_count / word_count * 100) if word_count > 0 else 0
        
        # Contraction ratio
        contraction_patterns = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
        text_contractions = sum(1 for w in words if any(p in w for p in contraction_patterns))
        text_contraction_ratio = text_contractions / word_count if word_count > 0 else 0
        
        # Question / exclamation cadence for single-line voice matching.
        text_is_question = float(text.rstrip().endswith('?'))
        text_is_exclamation = float(text.rstrip().endswith('!'))
        
        # Word length
        clean_words = [w.strip('.,!?"\'') for w in words if w.strip('.,!?"\'')]
        text_avg_word_length = sum(len(w) for w in clean_words) / len(clean_words) if clean_words else 0
        
        # Formality estimate for the text
        arch_norm = min(text_archaism_rate / 5.0, 1.0)
        contr_inv = max(0.0, 1.0 - text_contraction_ratio * 10)
        wl_norm = min((text_avg_word_length - 3.0) / 3.0, 1.0) if text_avg_word_length > 3 else 0.0
        text_formality = arch_norm * 0.5 + contr_inv * 0.3 + wl_norm * 0.2
        
        # Distinctive word overlap (strip punctuation for matching)
        text_word_set = set(w.strip('.,!?"\'-') for w in words)
        
        scores: list[tuple[str, float]] = []
        
        for char_name, profile in profiles.items():
            score = 0.0
            weight_total = 0.0
            
            # 1. Formality score proximity (weight: 0.25)
            formality_diff = abs(profile.formality_score - text_formality)
            formality_score = max(0.0, 1.0 - formality_diff * 2)
            score += formality_score * 0.25
            weight_total += 0.25
            
            # 2. Archaism rate proximity (weight: 0.25)
            arch_diff = abs(profile.archaism_rate - text_archaism_rate)
            arch_score = max(0.0, 1.0 - arch_diff / 5.0)
            score += arch_score * 0.25
            weight_total += 0.25
            
            # 3. Contraction ratio proximity (weight: 0.15)
            contr_diff = abs(profile.contraction_ratio - text_contraction_ratio)
            contr_score = max(0.0, 1.0 - contr_diff * 5)
            score += contr_score * 0.15
            weight_total += 0.15
            
            # 4. Distinctive word overlap (weight: 0.20)
            if profile.distinctive_words:
                dist_set = set(profile.distinctive_words)
                overlap = len(text_word_set & dist_set) / max(len(dist_set), 1)
                score += min(overlap * 5, 1.0) * 0.20  # Amplify (usually few overlap)
            weight_total += 0.20
            
            # 5. Signature phrase match (weight: 0.10)
            if profile.signature_phrases:
                text_lower = text.lower()
                phrase_hits = sum(1 for phrase in profile.signature_phrases if phrase in text_lower)
                if phrase_hits > 0:
                    score += min(phrase_hits / 2, 1.0) * 0.10
            weight_total += 0.10

            # 6. Utterance cadence proximity (weight: 0.05)
            punctuation_score = 1.0 - (
                abs(text_is_question - profile.question_ratio)
                + abs(text_is_exclamation - profile.exclamation_ratio)
            ) / 2.0
            score += max(0.0, punctuation_score) * 0.05
            weight_total += 0.05
            
            # 7. Penalty: text uses "never_says" words (weight: -0.05 per hit)
            if profile.never_says:
                never_set = set(profile.never_says)
                penalty = sum(1 for w in text_word_set if w in never_set)
                score -= penalty * 0.05
            
            # Normalize
            normalized = max(0.0, score / weight_total) if weight_total > 0 else 0.0
            scores.append((char_name, round(normalized, 3)))
        
        # Sort by confidence desc
        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]
    
    def check_voice_violations(
        self,
        text: str,
        profile: CharacterVoiceProfile,
    ) -> list[dict]:
        """
        Check generated dialogue for voice consistency violations.
        
        Returns list of violation dicts with keys:
          - type: 'wrong_formality' | 'uses_never_says' | 'missing_signature' | 'anachronism'
          - severity: 'hard' | 'soft'
          - message: Human-readable description
        """
        violations = []
        words = text.lower().split()
        clean_words = set(w.strip('.,!?"\'') for w in words)
        
        # Check 1: Wrong formality level
        archaisms_list = {
            "thee", "thou", "thy", "thine", "ye", "hath", "doth",
        }
        text_archaism_count = sum(1 for w in clean_words if w in archaisms_list)
        text_contraction_count = sum(
            1 for w in words if any(p in w for p in ["n't", "'re", "'ve", "'ll", "'d", "'m"])
        )
        word_count = len(words) or 1
        text_formality = (
            min(text_archaism_count / word_count * 100 / 5.0, 1.0) * 0.5
            + max(0.0, 1.0 - (text_contraction_count / word_count) * 10) * 0.5
        )
        
        if abs(text_formality - profile.formality_score) > 0.4:
            direction = "too formal" if text_formality > profile.formality_score else "too informal"
            violations.append({
                "type": "wrong_formality",
                "severity": "hard" if abs(text_formality - profile.formality_score) > 0.6 else "soft",
                "message": (
                    f"Formality mismatch: text is {direction} for {profile.character_name} "
                    f"(text={text_formality:.2f}, expected≈{profile.formality_score:.2f})"
                ),
            })
        
        # Check 2: Uses words from never_says list
        if profile.never_says:
            never_set = set(profile.never_says)
            banned_words = clean_words & never_set
            if banned_words:
                violations.append({
                    "type": "uses_never_says",
                    "severity": "soft",
                    "message": (
                        f"{profile.character_name} would never say: {', '.join(banned_words)}"
                    ),
                })
        
        # Check 3: Modern/anachronistic vocabulary
        modern_words = {
            "okay", "ok", "alright", "yeah", "yep", "nope", "sure", "cool",
            "awesome", "totally", "literally", "basically", "actually",
            "whatever", "stuff", "things", "guys", "hey", "hi", "bye",
        }
        anachronisms = clean_words & modern_words
        if anachronisms:
            violations.append({
                "type": "anachronism",
                "severity": "hard",
                "message": (
                    f"Anachronistic vocabulary for {profile.character_name}: "
                    f"{', '.join(anachronisms)}"
                ),
            })
        
        # Check 4: Missing signature patterns (only if many phrases defined)
        if len(profile.signature_phrases) >= 3:
            text_lower = text.lower()
            any_sig = any(phrase in text_lower for phrase in profile.signature_phrases)
            if not any_sig and len(words) > 10:
                violations.append({
                    "type": "missing_signature",
                    "severity": "soft",
                    "message": (
                        f"No signature patterns detected for {profile.character_name}. "
                        f"Consider: {', '.join(profile.signature_phrases[:3])}"
                    ),
                })
        
        return violations

    def save_results(
        self,
        result: VoiceAnalysisResult,
        output_path: str | Path,
    ):
        """Save analysis results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "source_file": result.source_file,
            "stats": {
                "total_dialogue_lines": result.total_dialogue_lines,
                "total_characters": result.total_characters,
                "unattributed_lines": result.unattributed_lines,
                "attribution_rate": result.attribution_rate,
            },
            "top_speakers": result.top_speakers(20),
            "profiles": {
                name: profile.to_dict()
                for name, profile in result.profiles.items()
            },
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    
    def load_results(self, input_path: str | Path) -> VoiceAnalysisResult:
        """Load analysis results from JSON."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = VoiceAnalysisResult(source_file=data["source_file"])
        result.total_dialogue_lines = data["stats"]["total_dialogue_lines"]
        result.total_characters = data["stats"]["total_characters"]
        result.unattributed_lines = data["stats"]["unattributed_lines"]
        result.attribution_rate = data["stats"]["attribution_rate"]
        
        for name, profile_dict in data.get("profiles", {}).items():
            result.profiles[name] = CharacterVoiceProfile.from_dict(profile_dict)
        
        return result
