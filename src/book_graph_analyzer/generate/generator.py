"""Scene generator with Neo4j context and constitutional critique."""

import json
import re
import uuid
from typing import Optional

from ..llm import LLMClient
from ..graph.connection import get_driver
from ..worldbible import WorldBible
from .models import Scene, SceneScores, GenerationConfig, GenerationStatus
from .judge import NarrativeJudge


class SceneGenerator:
    """Generates scenes grounded in Neo4j knowledge graph."""
    
    GENERATION_PROMPT = '''You are writing a scene in the style of J.R.R. Tolkien.

SETTING: {setting}
CHARACTERS PRESENT: {characters}
OBJECTS OF NOTE: {objects}

PREVIOUS EVENTS:
{previous_context}

SCENE GOAL: {scene_goal}

WORLD RULES TO RESPECT:
{world_rules}

Write the scene in Tolkien's style:
- Flowing, rhythmic prose with Anglo-Saxon cadence
- Rich nature imagery and attention to landscape
- Formal dialogue appropriate to each character's race and status
- Mythic, omniscient narrative voice
- Show don't tell - let actions and dialogue reveal character

Write 400-800 words. Begin the scene directly, no preamble.'''

    CRITIQUE_PROMPT = '''Review this passage for lore violations and inconsistencies.

PASSAGE:
"""
{passage}
"""

WORLD RULES:
{world_rules}

KNOWN FACTS:
- Characters: {characters}
- Setting: {setting}
- Timeline: {timeline}

Check for:
1. LORE VIOLATIONS: Does anything contradict established world rules?
2. CHARACTER INCONSISTENCY: Do characters act against their established nature?
3. TIMELINE ERRORS: Does the sequence of events make sense?
4. ANACHRONISMS: Any modern words, concepts, or items that don't belong?

List all violations found. If none, say "NO VIOLATIONS".

Respond in JSON:
{{
    "violations": [
        {{"type": "lore|character|timeline|anachronism", "description": "...", "severity": "minor|major"}},
        ...
    ],
    "lore_score": <0.0-1.0, where 1.0 = no violations>
}}'''

    REVISION_PROMPT = '''Revise this passage to fix the following issues:

ORIGINAL:
"""
{passage}
"""

ISSUES TO FIX:
{issues}

Rewrite the passage fixing these issues while maintaining Tolkien's style.
Keep the same general content and length, just fix the problems.'''

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.llm = LLMClient()
        self.judge = NarrativeJudge()
        self.driver = get_driver()
        self.world_bible: Optional[WorldBible] = None
    
    def load_world_bible(self, path: str) -> None:
        """Load world bible for constraint checking."""
        self.world_bible = WorldBible.load(path)
    
    def get_context_from_neo4j(
        self,
        characters: list[str],
        place: str,
        limit: int = 10,
    ) -> dict:
        """Query Neo4j for relevant context."""
        context = {
            "characters": [],
            "place": None,
            "objects": [],
            "recent_events": [],
            "relationships": [],
        }
        
        if not self.driver:
            return context
        
        with self.driver.session() as session:
            # Get character info
            for char_name in characters:
                result = session.run("""
                    MATCH (c:Character)
                    WHERE toLower(c.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (c)-[r]-(related)
                    RETURN c.name as name, c.type as type, c.description as desc,
                           collect(DISTINCT {rel: type(r), target: related.name})[..5] as relations
                    LIMIT 1
                """, name=char_name)
                record = result.single()
                if record:
                    context["characters"].append({
                        "name": record["name"],
                        "type": record["type"],
                        "description": record["desc"],
                        "relations": record["relations"],
                    })
            
            # Get place info
            if place:
                result = session.run("""
                    MATCH (p:Place)
                    WHERE toLower(p.name) CONTAINS toLower($name)
                    RETURN p.name as name, p.description as desc, p.region as region
                    LIMIT 1
                """, name=place)
                record = result.single()
                if record:
                    context["place"] = {
                        "name": record["name"],
                        "description": record["desc"],
                        "region": record["region"],
                    }
            
            # Get recent events involving these characters
            if characters:
                result = session.run("""
                    MATCH (e:Event)
                    WHERE any(c IN $characters WHERE toLower(e.agent) CONTAINS toLower(c))
                    RETURN e.description as desc, e.era as era, e.year as year
                    ORDER BY e.year DESC
                    LIMIT $limit
                """, characters=characters, limit=limit)
                context["recent_events"] = [
                    {"description": r["desc"], "era": r["era"], "year": r["year"]}
                    for r in result
                ]
        
        return context
    
    def get_world_rules(self, categories: list[str] = None) -> str:
        """Get relevant world bible rules as text."""
        if not self.world_bible:
            return "No world bible loaded."
        
        rules = []
        for rule in self.world_bible.rules:
            if categories is None or rule.category.value in categories:
                rules.append(f"- {rule.text}")
        
        return "\n".join(rules[:20])  # Limit to avoid context overflow
    
    def generate_scene(
        self,
        scene_goal: str,
        characters: list[str],
        place: str,
        previous_context: str = "",
        objects: list[str] = None,
    ) -> Scene:
        """Generate a scene with full pipeline."""
        
        # 1. Get context from Neo4j
        neo4j_context = self.get_context_from_neo4j(characters, place)
        
        # Format context for prompt
        char_descriptions = []
        for c in neo4j_context["characters"]:
            desc = f"{c['name']}"
            if c.get("type"):
                desc += f" ({c['type']})"
            if c.get("description"):
                desc += f": {c['description'][:100]}"
            char_descriptions.append(desc)
        
        place_desc = ""
        if neo4j_context["place"]:
            p = neo4j_context["place"]
            place_desc = f"{p['name']}"
            if p.get("description"):
                place_desc += f" - {p['description'][:150]}"
        
        events_text = ""
        if neo4j_context["recent_events"]:
            events_text = "\n".join(
                f"- {e['description']}" 
                for e in neo4j_context["recent_events"][:5]
            )
        
        # 2. Generate initial scene
        prompt = self.GENERATION_PROMPT.format(
            setting=place_desc or place,
            characters="\n".join(char_descriptions) or ", ".join(characters),
            objects=", ".join(objects or []) or "None specified",
            previous_context=previous_context or events_text or "Beginning of story",
            scene_goal=scene_goal,
            world_rules=self.get_world_rules(),
        )
        
        scene_text = self.llm.generate(prompt, temperature=self.config.temperature)
        
        # 3. Create scene object
        scene = Scene(
            id=str(uuid.uuid4())[:8],
            number=0,  # Set by chapter
            text=scene_text,
            summary=scene_goal,
            characters=characters,
            places=[place] if place else [],
            objects=objects or [],
            model_used=self.config.model,
            generation_prompt=prompt,
        )
        
        # 4. Constitutional critique loop
        for iteration in range(self.config.max_critique_iterations):
            violations = self._critique_scene(scene, neo4j_context)
            
            if not violations:
                break
            
            scene.critique_notes.extend([v["description"] for v in violations])
            scene.revision_count += 1
            
            # Revise
            scene.text = self._revise_scene(scene.text, violations)
        
        # 5. Score the scene
        scene.scores = self._score_scene(scene, previous_context)
        
        # 6. Flag if below threshold
        if scene.scores.overall < self.config.min_quality_score:
            scene.status = GenerationStatus.FLAGGED
        
        return scene
    
    def _critique_scene(self, scene: Scene, context: dict) -> list[dict]:
        """Run constitutional critique on scene."""
        prompt = self.CRITIQUE_PROMPT.format(
            passage=scene.text,
            world_rules=self.get_world_rules(),
            characters=", ".join(scene.characters),
            setting=", ".join(scene.places),
            timeline="Third Age" if not context.get("recent_events") else 
                     context["recent_events"][0].get("era", "Unknown"),
        )
        
        response = self.llm.generate(prompt, temperature=0.2)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                violations = data.get("violations", [])
                # Filter to major violations only for revision
                return [v for v in violations if v.get("severity") == "major"]
        except (json.JSONDecodeError, KeyError):
            pass
        
        return []
    
    def _revise_scene(self, text: str, violations: list[dict]) -> str:
        """Revise scene to fix violations."""
        issues = "\n".join(
            f"- [{v['type'].upper()}] {v['description']}"
            for v in violations
        )
        
        prompt = self.REVISION_PROMPT.format(
            passage=text,
            issues=issues,
        )
        
        return self.llm.generate(prompt, temperature=0.7)
    
    def _score_scene(self, scene: Scene, context: str) -> SceneScores:
        """Score scene on all dimensions."""
        # Get narrative + style scores from judge
        scores, critique, weaknesses = self.judge.full_evaluation(scene.text, context)
        
        # Get lore score from critique
        violations = self._critique_scene(scene, {})
        if not violations:
            scores.lore_score = 1.0
        else:
            # Deduct based on violation count and severity
            deduction = sum(0.2 if v.get("severity") == "major" else 0.1 for v in violations)
            scores.lore_score = max(0.0, 1.0 - deduction)
        
        # Consistency score (placeholder - could check voice profiles)
        scores.consistency_score = 0.8  # TODO: Implement voice profile matching
        
        # Compute overall
        scores.compute_overall(self.config)
        
        # Add critique to notes
        if critique:
            scene.critique_notes.append(f"Judge: {critique}")
        scene.critique_notes.extend([f"Weakness: {w}" for w in weaknesses])
        
        return scores
