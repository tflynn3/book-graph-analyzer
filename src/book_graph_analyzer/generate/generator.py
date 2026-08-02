"""Scene generator with Neo4j context and constitutional critique."""

import json
import re
import uuid
from typing import Optional

from ..graph.connection import get_driver
from ..llm import LLMClient
from ..voice.dialogue import extract_dialogue
from ..worldbible import WorldBible
from .context import AssembledContext, ContextAssembler
from .judge import NarrativeJudge
from .models import GenerationConfig, GenerationStatus, Scene, SceneScores
from .pipeline import StagedPipeline
from .style_injector import StyleInjector
from .voice_patcher import VoicePatcher


class SceneGenerator:
    """Generates scenes grounded in Neo4j knowledge graph."""

    FOG_OF_WAR_PROMPT = '''Write an original high-fantasy scene in a Tolkien-inspired register.
Do not quote, paraphrase, or continue any source passage. Use only the canon facts and
scene constraints below as grounding.

SETTING: {setting}
CHARACTERS PRESENT: {characters}
OBJECTS OF NOTE: {objects}

WHAT THE CHARACTER KNOWS:
{character_knowledge}

SCENE GOAL: {scene_goal}

WORLD RULES TO RESPECT:
{world_rules}

Write this scene from a position of UNCERTAINTY. The character does not know the history of
this place or the nature of what they encounter. Let physical evidence and architecture hint
at past greatness or horror, and let silence carry weight. Do not explain what the viewpoint
character cannot know.

Register guidance for scenes of mystery:
- The landscape itself feels conscious and watching
- Details are physical and sensory — cold stone, old smell, soundlessness
- The character notices without understanding
- Formal, measured prose; no modern psychological interiority
- Dread or wonder conveyed through what is NOT said

Write 400-800 words. Begin the scene directly, no preamble.'''

    GENERATION_PROMPT = '''Write an original high-fantasy scene in a Tolkien-inspired register.
Do not quote, paraphrase, or continue any source passage. Keep the prose original while
respecting the canon facts, timeline, and scene constraints below.

SETTING: {setting}
CHARACTERS PRESENT: {characters}
OBJECTS OF NOTE: {objects}

PREVIOUS EVENTS:
{previous_context}

SCENE GOAL: {scene_goal}

WORLD RULES TO RESPECT:
{world_rules}

{style_constraints}

Write 400-800 words. Begin the scene directly, no preamble. Avoid modern idiom,
placeholder names, and characters who do not belong to this story-time.'''

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

KNOWN CURRENT STATE AND EVIDENCE:
{known_state}

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

Rewrite the passage fixing these issues while maintaining its established register and narrative intent.
Keep the same general content and length, just fix the problems.'''

    def __init__(self, config: Optional[GenerationConfig] = None, shadow_graph=None):
        self.config = config or GenerationConfig()
        self.llm = LLMClient()
        self.judge = NarrativeJudge()
        self.driver = get_driver()
        self.shadow_graph = shadow_graph
        self.context_assembler = ContextAssembler(
            shadow_graph=shadow_graph,
            neo4j_driver=self.driver,
        )
        self.world_bible: Optional[WorldBible] = None
        self.voice_patcher = VoicePatcher(llm_client=self.llm)
        self.style_injector = StyleInjector(driver=self.driver)
        self.pipeline = StagedPipeline(
            scene_generator=self,
            voice_patcher=self.voice_patcher,
            config=self.config,
        )
    
    def load_world_bible(self, path: str) -> None:
        """Load world bible for constraint checking."""
        self.world_bible = WorldBible.load(path)
    
    def get_context_from_neo4j(
        self,
        characters: list[str],
        place: str,
        limit: int = 10,
        fog_of_war: bool = False,
        story_era: Optional[str] = None,
        story_year: Optional[int] = None,
    ) -> dict:
        """Query Neo4j for relevant context.

        Args:
            fog_of_war: When True, only fetch the physical description of the place.
                Character histories and event records are withheld — the LLM writes
                from the character's limited, ground-level perspective.
        """
        context = {
            "characters": [],
            "place": None,
            "objects": [],
            "recent_events": [],
            "relationships": [],
            "timeline": {"era": story_era, "year": story_year},
        }
        
        if not self.driver:
            return context
        
        with self.driver.session() as session:
            # In fog_of_war mode: omit character history — just their names/races
            for char_name in characters:
                result = session.run("""
                    MATCH (c:Character)
                    WHERE toLower(coalesce(c.name, c.canonical_name, '')) CONTAINS toLower($name)
                    OPTIONAL MATCH (c)-[r]-(related)
                    RETURN coalesce(c.name, c.canonical_name) as name,
                           c.type as type,
                           c.description as desc,
                           collect(DISTINCT {rel: type(r), target: coalesce(related.name, related.canonical_name)})[..5] as relations
                    LIMIT 1
                """, name=char_name)
                record = result.single()
                if record:
                    if fog_of_war:
                        # Strip relational/historical context
                        context["characters"].append({
                            "name": record["name"],
                            "type": record["type"],
                            "description": None,
                            "relations": [],
                        })
                    else:
                        context["characters"].append({
                            "name": record["name"],
                            "type": record["type"],
                            "description": record["desc"],
                            "relations": record["relations"],
                        })
                        context["relationships"].extend(record["relations"] or [])
            
            # Place: always fetch physical description
            if place:
                result = session.run("""
                    MATCH (p:Place)
                    WHERE toLower(coalesce(p.name, p.canonical_name, '')) CONTAINS toLower($name)
                    RETURN coalesce(p.name, p.canonical_name) as name, p.description as desc, p.region as region
                    LIMIT 1
                """, name=place)
                record = result.single()
                if record:
                    context["place"] = {
                        "name": record["name"],
                        "description": record["desc"],
                        "region": record["region"],
                    }
            
            # Events: withheld entirely in fog_of_war mode
            if characters and not fog_of_war:
                result = session.run("""
                    MATCH (e:Event)
                    WHERE any(c IN $characters WHERE toLower(e.agent) CONTAINS toLower(c))
                      AND (
                        $story_era IS NULL
                        OR replace(toLower(coalesce(e.era, '')), '_', ' ')
                           = replace(toLower($story_era), '_', ' ')
                      )
                      AND ($story_year IS NULL OR e.year IS NULL OR e.year <= $story_year)
                    RETURN e.id as id, e.description as desc, e.era as era, e.year as year,
                           e.source_book as source_book, e.source_location as source_location
                    ORDER BY e.year DESC
                    LIMIT $limit
                """, characters=characters, story_era=story_era, story_year=story_year, limit=limit)
                context["recent_events"] = [
                    {
                        "id": r["id"],
                        "description": r["desc"],
                        "era": r["era"],
                        "year": r["year"],
                        "source_book": r["source_book"],
                        "source_location": r["source_location"],
                    }
                    for r in result
                ]
        
        return context

    def _build_character_knowledge(
        self,
        characters: list[str],
        previous_context: str,
    ) -> str:
        """
        For Fog of War mode: what does the character actually know?
        Only their personal history from the scene summaries — nothing omniscient.
        """
        lines = []
        if previous_context:
            lines.append("What has happened so far (from the character's perspective):")
            lines.append(previous_context.strip())
        else:
            lines.append(f"{', '.join(characters)} approach this place knowing only what they have witnessed on their journey.")
            lines.append("They know nothing of this location's history.")
        return "\n".join(lines)
    
    def get_world_rules(self, categories: list[str] = None) -> str:
        """Get relevant world bible rules as text."""
        if not self.world_bible:
            return "No world bible loaded."
        
        rules = []
        for category, rule_list in self.world_bible.rules.items():
            for rule in rule_list:
                if categories is None or category.value in categories:
                    rules.append(f"- [{category.value}] {rule.title}: {rule.description}")
        
        return "\n".join(rules[:20])  # Limit to avoid context overflow
    
    def generate_scene(
        self,
        scene_goal: str,
        characters: list[str],
        place: str,
        previous_context: str = "",
        objects: list[str] = None,
        fog_of_war: bool = False,
        assembled_context: Optional[AssembledContext] = None,
        story_id: Optional[str] = None,
        chapter_num: int = 0,
        scene_num: int = 0,
        story_era: Optional[str] = None,
        story_year: Optional[int] = None,
        voice_profiles: Optional[dict] = None,
    ) -> Scene:
        """Generate a scene with full pipeline.

        Args:
            fog_of_war: When True, restricts context to physical place description only.
                The characters know nothing of the location's history. The LLM writes
                from a position of uncertainty — producing naturally ominous, mysterious prose.
                Use for: entering Moria, approaching a ruined city, Fog-filled valleys.
        """
        
        # 1. Get context from Neo4j (restricted in fog_of_war mode)
        neo4j_context = self.get_context_from_neo4j(
            characters,
            place,
            fog_of_war=fog_of_war,
            story_era=story_era,
            story_year=story_year,
        )
        
        # Format context for prompt
        char_descriptions = []
        for c in neo4j_context["characters"]:
            desc = f"{c['name']}"
            if c.get("type"):
                desc += f" ({c['type']})"
            if c.get("description"):
                desc += f": {c['description'][:100]}"
            relations = c.get("relations") or []
            relation_text = ", ".join(
                f"{row.get('rel')} {row.get('target')}"
                for row in relations[:3]
                if isinstance(row, dict) and row.get("rel") and row.get("target")
            )
            if relation_text:
                desc += f"; relations: {relation_text}"
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
        
        # 2. Assemble structured context if available
        if not assembled_context and self.shadow_graph:
            resolved_story_id = story_id or getattr(self.shadow_graph, "story_id", "")
            if resolved_story_id:
                assembled_context = self.context_assembler.assemble(
                    story_id=resolved_story_id,
                    characters=characters,
                    place=place,
                    chapter_num=chapter_num,
                    scene_num=scene_num,
                )

        previous_context_text = str(previous_context or "").strip()
        context_parts = [previous_context_text] if previous_context_text else []
        if assembled_context:
            context_parts.append(assembled_context.to_prompt_block())
        if events_text:
            context_parts.append("CANON EVENT EVIDENCE:\n" + events_text)
        context_text = "\n\n".join(part for part in context_parts if part)

        scene_type: Optional[str] = None
        style_constraints_obj = None

        # 3. Generate initial scene
        if fog_of_war:
            # Fog of War: character only knows their own situation and the physical place.
            # No history, no events, no omniscient context — pure sensory grounding.
            character_knowledge = self._build_character_knowledge(
                characters, context_text
            )
            prompt = self.FOG_OF_WAR_PROMPT.format(
                setting=place_desc or place,
                characters="\n".join(char_descriptions) or ", ".join(characters),
                objects=", ".join(objects or []) or "None specified",
                character_knowledge=character_knowledge,
                scene_goal=scene_goal,
                world_rules=self.get_world_rules(),
            )
        else:
            self.style_injector.driver = self.driver
            scene_type = self.style_injector.classify_scene_type(scene_goal)
            style_constraints_obj = self.style_injector.get_style_constraints(scene_type)
            style_block = self.style_injector.build_style_block(style_constraints_obj)
            prompt = self.GENERATION_PROMPT.format(
                setting=place_desc or place,
                characters="\n".join(char_descriptions) or ", ".join(characters),
                objects=", ".join(objects or []) or "None specified",
                previous_context=context_text or events_text or "Beginning of story",
                scene_goal=scene_goal,
                world_rules=self.get_world_rules(),
                style_constraints=style_block,
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
            model_used=getattr(self.llm, "provider_label", self.config.model),
            generation_prompt=prompt,
            context_snapshot=assembled_context,
            scene_type=scene_type,
            style_constraints_used=(
                style_constraints_obj.to_dict()
                if style_constraints_obj is not None
                else None
            ),
        )
        
        # 4. Staged pipeline (lore enforce + optional voice patch)
        scene, lore_violations = self.pipeline.run(
            scene=scene,
            neo4j_context=neo4j_context,
            voice_profiles=voice_profiles or {},
        )

        # 6. Score the scene
        scene.scores = self._score_scene(
            scene,
            context_text,
            lore_violations=lore_violations,
            voice_profiles=voice_profiles or {},
        )
        
        # 7. Flag if below threshold
        if scene.scores.overall < self.config.min_quality_score:
            scene.status = GenerationStatus.FLAGGED
        
        return scene

    def _run_lore_enforcement(
        self,
        scene: Scene,
        context: dict,
    ) -> tuple[Scene, list[dict], bool]:
        """Run constitutional critique loop and return only the final check."""
        final_violations: list[dict] = []
        revisions = 0
        max_revisions = max(0, int(self.config.max_critique_iterations))

        while True:
            critique_result = self._critique_scene(scene, context)
            # Preserve compatibility with callers/tests that replace the critic
            # with the former list-only contract.
            if isinstance(critique_result, tuple) and len(critique_result) == 2:
                violations, verified = critique_result
            else:
                violations, verified = critique_result, True

            if not verified:
                scene.critique_notes.append(
                    "Lore verification failed: the critic response could not be parsed."
                )
                scene.word_count = len(scene.text.split())
                return scene, [], False

            final_violations = list(violations or [])
            major_violations = [
                violation
                for violation in final_violations
                if str(violation.get("severity") or "").lower() == "major"
            ]
            if not major_violations or revisions >= max_revisions:
                break

            scene.revision_count += 1
            scene.text = self._revise_scene(scene.text, major_violations)
            revisions += 1
            # Loop once more even after the last permitted revision: scoring
            # must describe the revised text, not the stale pre-revision check.

        scene.critique_notes.extend(
            str(violation.get("description") or "Lore issue detected.")
            for violation in final_violations
        )
        scene.word_count = len(scene.text.split())
        return scene, final_violations, True
    
    def _critique_scene(self, scene: Scene, context: dict) -> tuple[list[dict], bool]:
        """Run constitutional critique and report whether its output was verified."""
        prompt = self.CRITIQUE_PROMPT.format(
            passage=scene.text,
            world_rules=self.get_world_rules(),
            characters=", ".join(scene.characters),
            setting=", ".join(scene.places),
            timeline=(
                f"{context.get('timeline', {}).get('era') or 'Unknown'}"
                f" {context.get('timeline', {}).get('year') or ''}"
            ).strip(),
            known_state=(
                scene.context_snapshot.to_prompt_block()
                if scene.context_snapshot is not None
                else "No structured scene state supplied."
            ),
        )
        
        response = self.llm.generate(prompt, temperature=0.2)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return [], False
            data = json.loads(json_match.group())
            if not isinstance(data, dict):
                return [], False
            violations = data.get("violations", [])
            if not isinstance(violations, list) or any(
                not isinstance(violation, dict) for violation in violations
            ):
                return [], False
            return violations, True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        return [], False
    
    def _revise_scene(self, text: str, violations: list[dict]) -> str:
        """Revise scene to fix violations."""
        issues = "\n".join(
            f"- [{str(v.get('type') or 'lore').upper()}] "
            f"{str(v.get('description') or 'Lore issue detected.')}"
            for v in violations
        )
        
        prompt = self.REVISION_PROMPT.format(
            passage=text,
            issues=issues,
        )
        
        return self.llm.generate(prompt, temperature=0.7)
    
    def _score_scene(
        self,
        scene: Scene,
        context: str,
        lore_violations: Optional[list[dict]] = None,
        voice_profiles: Optional[dict] = None,
    ) -> SceneScores:
        """Score scene on all dimensions."""
        # Get narrative + style scores from judge
        scores, critique, weaknesses = self.judge.full_evaluation(scene.text, context)
        
        # Get lore score from staged-lore pass
        violations = lore_violations if lore_violations is not None else []
        lore_was_checked = "lore_enforce" in scene.pipeline_stages_run
        if not lore_was_checked:
            scores.lore_score = 0.0
            scene.critique_notes.append("Lore score is unverified: no evidence-backed lore pass ran.")
        elif not violations:
            scores.lore_score = 1.0
        else:
            # Deduct based on violation count and severity
            deduction = sum(0.2 if v.get("severity") == "major" else 0.1 for v in violations)
            scores.lore_score = max(0.0, 1.0 - deduction)
        
        supplied_voice_profiles = voice_profiles or {}
        matched_speakers: set[str] = set()
        if supplied_voice_profiles:
            dialogue = extract_dialogue(scene.text, passage_id=scene.id)
            matched_speakers = {
                line.speaker
                for line in dialogue.dialogue_lines
                if line.speaker and line.speaker in supplied_voice_profiles
            }
        if matched_speakers:
            matched_profiles = {
                speaker: supplied_voice_profiles[speaker]
                for speaker in matched_speakers
            }
            deviation = self.voice_patcher.estimate_max_deviation(scene, matched_profiles)
            scores.consistency_score = max(0.0, 1.0 - deviation)
        else:
            scores.consistency_score = 0.0
            scene.critique_notes.append(
                "Consistency score is unverified: no attributed dialogue matched "
                "a supplied voice profile."
            )
        
        # Compute overall
        scores.compute_overall(self.config)
        
        # Add critique to notes
        if critique:
            scene.critique_notes.append(f"Judge: {critique}")
        scene.critique_notes.extend([f"Weakness: {w}" for w in weaknesses])
        
        return scores
