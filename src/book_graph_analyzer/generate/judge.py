"""LLM-as-judge for narrative quality evaluation."""

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..llm import LLMClient
from .models import SceneScores


@dataclass
class NarrativeJudgment:
    """Result of narrative quality judgment."""
    engagement: float
    pacing: float
    dialogue: float
    imagery: float
    emotional_weight: float
    
    overall: float
    critique: str
    strengths: list[str]
    weaknesses: list[str]


class NarrativeJudge:
    """Evaluates narrative quality using LLM-as-judge."""
    
    JUDGE_PROMPT = '''You are a harsh literary critic specializing in fantasy fiction, particularly Tolkien's style.

Rate this passage on a scale of 1-10 for each dimension. Be critical - Tolkien's bar is high.

PASSAGE:
"""
{passage}
"""

CONTEXT (previous events):
{context}

Rate each dimension and explain briefly:

1. ENGAGEMENT (1-10): Would a reader want to keep reading? Is there tension, mystery, or stakes?
2. PACING (1-10): Does the scene breathe? Not too rushed, not dragging?
3. DIALOGUE (1-10): Natural speech? Reveals character? Appropriate formality for the setting?
4. IMAGERY (1-10): Vivid and evocative? Or flat, generic description?
5. EMOTIONAL_WEIGHT (1-10): Does the scene land emotionally? Do we feel something?

Also provide:
- STRENGTHS: What works well (2-3 points)
- WEAKNESSES: What needs improvement (2-3 points)
- OVERALL: Single score 1-10 weighing all factors

Respond in JSON format:
{{
    "engagement": <score>,
    "pacing": <score>,
    "dialogue": <score>,
    "imagery": <score>,
    "emotional_weight": <score>,
    "overall": <score>,
    "strengths": ["...", "..."],
    "weaknesses": ["...", "..."],
    "critique": "<one paragraph summary>"
}}'''

    STYLE_PROMPT = '''Compare this passage to Tolkien's writing style.

PASSAGE:
"""
{passage}
"""

Consider:
- Sentence structure and rhythm (Tolkien favors longer, flowing sentences with Anglo-Saxon cadence)
- Vocabulary (archaic/formal register, nature imagery, compound words)
- Narrative voice (omniscient, mythic tone, occasional direct address)
- Dialogue style (formal, poetic, characters speak differently by race/status)

Rate STYLE_MATCH from 1-10 where:
- 10 = Indistinguishable from Tolkien
- 7-9 = Clearly Tolkien-influenced, minor tells
- 4-6 = Generic fantasy, not distinctly Tolkien
- 1-3 = Clearly modern/wrong register

Respond in JSON:
{{
    "style_score": <score>,
    "tolkien_elements": ["...", "..."],
    "non_tolkien_elements": ["...", "..."],
    "suggestions": "..."
}}'''

    def __init__(self):
        self.llm = LLMClient()
    
    def judge_narrative(self, passage: str, context: str = "") -> NarrativeJudgment:
        """Judge narrative quality of a passage."""
        prompt = self.JUDGE_PROMPT.format(
            passage=passage,
            context=context or "No prior context provided."
        )
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            return NarrativeJudgment(
                engagement=float(data.get("engagement", 5)) / 10,
                pacing=float(data.get("pacing", 5)) / 10,
                dialogue=float(data.get("dialogue", 5)) / 10,
                imagery=float(data.get("imagery", 5)) / 10,
                emotional_weight=float(data.get("emotional_weight", 5)) / 10,
                overall=float(data.get("overall", 5)) / 10,
                critique=data.get("critique", ""),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to neutral scores
            print(f"Warning: Failed to parse judge response: {e}")
            return NarrativeJudgment(
                engagement=0.5,
                pacing=0.5,
                dialogue=0.5,
                imagery=0.5,
                emotional_weight=0.5,
                overall=0.5,
                critique="Failed to parse judgment",
                strengths=[],
                weaknesses=[],
            )
    
    def judge_style(self, passage: str) -> tuple[float, list[str], list[str]]:
        """Judge Tolkien style match.
        
        Returns:
            (style_score, tolkien_elements, non_tolkien_elements)
        """
        prompt = self.STYLE_PROMPT.format(passage=passage)
        response = self.llm.generate(prompt, temperature=0.3)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
            
            return (
                float(data.get("style_score", 5)) / 10,
                data.get("tolkien_elements", []),
                data.get("non_tolkien_elements", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return (0.5, [], [])
    
    def full_evaluation(self, passage: str, context: str = "") -> SceneScores:
        """Run full evaluation returning SceneScores."""
        # Narrative quality
        narrative = self.judge_narrative(passage, context)
        
        # Style match
        style_score, _, non_tolkien = self.judge_style(passage)
        
        scores = SceneScores(
            narrative_score=narrative.overall,
            style_score=style_score,
            engagement=narrative.engagement,
            pacing=narrative.pacing,
            dialogue=narrative.dialogue,
            imagery=narrative.imagery,
            emotional_weight=narrative.emotional_weight,
        )
        
        return scores, narrative.critique, narrative.weaknesses
