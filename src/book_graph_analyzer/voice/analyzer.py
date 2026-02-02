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

from .dialogue import (
    DialogueExtraction,
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
