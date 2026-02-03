"""Write generated stories to Neo4j."""

from typing import Optional
from datetime import datetime

from ..graph.connection import get_driver
from .models import Story, Chapter, Scene


class GenerationWriter:
    """Writes generated content to Neo4j."""
    
    def __init__(self):
        self.driver = get_driver()
    
    def ensure_schema(self) -> None:
        """Create constraints and indexes for generation nodes."""
        if not self.driver:
            print("Warning: No Neo4j connection")
            return
        
        with self.driver.session() as session:
            # Story constraint
            session.run("""
                CREATE CONSTRAINT story_id IF NOT EXISTS
                FOR (s:Story) REQUIRE s.id IS UNIQUE
            """)
            
            # Chapter constraint
            session.run("""
                CREATE CONSTRAINT chapter_id IF NOT EXISTS
                FOR (c:Chapter) REQUIRE c.id IS UNIQUE
            """)
            
            # Scene constraint
            session.run("""
                CREATE CONSTRAINT scene_id IF NOT EXISTS
                FOR (s:Scene) REQUIRE s.id IS UNIQUE
            """)
            
            # Indexes for queries
            session.run("""
                CREATE INDEX scene_quality IF NOT EXISTS
                FOR (s:Scene) ON (s.quality_score)
            """)
            
            session.run("""
                CREATE INDEX scene_status IF NOT EXISTS
                FOR (s:Scene) ON (s.status)
            """)
        
        print("Generation schema ready")
    
    def write_story(self, story: Story) -> dict:
        """Write a complete story to Neo4j."""
        if not self.driver:
            return {"error": "No Neo4j connection"}
        
        stats = {"stories": 0, "chapters": 0, "scenes": 0, "links": 0}
        
        with self.driver.session() as session:
            # Create story node
            session.run("""
                MERGE (s:Story {id: $id})
                SET s.title = $title,
                    s.corpus_name = $corpus,
                    s.premise = $premise,
                    s.outline = $outline,
                    s.created_at = datetime($created),
                    s.updated_at = datetime($updated)
            """, 
                id=story.id,
                title=story.title,
                corpus=story.corpus_name,
                premise=story.premise,
                outline=story.outline,
                created=story.created_at.isoformat(),
                updated=story.updated_at.isoformat(),
            )
            stats["stories"] = 1
            
            # Create chapters and scenes
            for chapter in story.chapters:
                self._write_chapter(session, story.id, chapter, stats)
        
        return stats
    
    def _write_chapter(self, session, story_id: str, chapter: Chapter, stats: dict) -> None:
        """Write a chapter and its scenes."""
        # Create chapter
        session.run("""
            MERGE (c:Chapter {id: $id})
            SET c.number = $number,
                c.title = $title,
                c.summary = $summary,
                c.outline = $outline
            WITH c
            MATCH (s:Story {id: $story_id})
            MERGE (s)-[:CONTAINS {position: $number}]->(c)
        """,
            id=chapter.id,
            number=chapter.number,
            title=chapter.title,
            summary=chapter.summary,
            outline=chapter.outline,
            story_id=story_id,
        )
        stats["chapters"] += 1
        
        # Create scenes
        for scene in chapter.scenes:
            self._write_scene(session, chapter.id, scene, stats)
    
    def _write_scene(self, session, chapter_id: str, scene: Scene, stats: dict) -> None:
        """Write a scene and link to entities."""
        # Create scene
        session.run("""
            MERGE (s:Scene {id: $id})
            SET s.number = $number,
                s.text = $text,
                s.summary = $summary,
                s.word_count = $word_count,
                s.quality_score = $quality,
                s.style_score = $style,
                s.lore_score = $lore,
                s.narrative_score = $narrative,
                s.status = $status,
                s.model_used = $model,
                s.generated_at = datetime($generated)
            WITH s
            MATCH (c:Chapter {id: $chapter_id})
            MERGE (c)-[:CONTAINS {position: $number}]->(s)
        """,
            id=scene.id,
            number=scene.number,
            text=scene.text,
            summary=scene.summary,
            word_count=scene.word_count,
            quality=scene.scores.overall,
            style=scene.scores.style_score,
            lore=scene.scores.lore_score,
            narrative=scene.scores.narrative_score,
            status=scene.status.value,
            model=scene.model_used,
            generated=scene.generated_at.isoformat(),
            chapter_id=chapter_id,
        )
        stats["scenes"] += 1
        
        # Link to characters
        for char_name in scene.characters:
            result = session.run("""
                MATCH (s:Scene {id: $scene_id})
                MATCH (c:Character)
                WHERE toLower(c.name) CONTAINS toLower($char_name)
                MERGE (s)-[:FEATURES]->(c)
                RETURN count(*) as linked
            """, scene_id=scene.id, char_name=char_name)
            stats["links"] += result.single()["linked"]
        
        # Link to places
        for place_name in scene.places:
            result = session.run("""
                MATCH (s:Scene {id: $scene_id})
                MATCH (p:Place)
                WHERE toLower(p.name) CONTAINS toLower($place_name)
                MERGE (s)-[:SET_IN]->(p)
                RETURN count(*) as linked
            """, scene_id=scene.id, place_name=place_name)
            stats["links"] += result.single()["linked"]
        
        # Link to objects
        for obj_name in scene.objects:
            result = session.run("""
                MATCH (s:Scene {id: $scene_id})
                MATCH (o:Object)
                WHERE toLower(o.name) CONTAINS toLower($obj_name)
                MERGE (s)-[:USES]->(o)
                RETURN count(*) as linked
            """, scene_id=scene.id, obj_name=obj_name)
            stats["links"] += result.single()["linked"]
        
        # Link to events depicted
        for event_desc in scene.events_depicted:
            result = session.run("""
                MATCH (s:Scene {id: $scene_id})
                MATCH (e:Event)
                WHERE toLower(e.description) CONTAINS toLower($event_desc)
                MERGE (s)-[:DEPICTS]->(e)
                RETURN count(*) as linked
            """, scene_id=scene.id, event_desc=event_desc)
            stats["links"] += result.single()["linked"]
    
    def write_scene(self, scene: Scene, chapter_id: str) -> dict:
        """Write a single scene to an existing chapter."""
        if not self.driver:
            return {"error": "No Neo4j connection"}
        
        stats = {"scenes": 0, "links": 0}
        
        with self.driver.session() as session:
            self._write_scene(session, chapter_id, scene, stats)
        
        return stats
    
    def get_flagged_scenes(self, limit: int = 10) -> list[dict]:
        """Get scenes flagged for review."""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Scene {status: 'flagged'})
                OPTIONAL MATCH (s)-[:FEATURES]->(c:Character)
                RETURN s.id as id, s.summary as summary, s.quality_score as score,
                       s.text as text, collect(c.name) as characters
                ORDER BY s.quality_score ASC
                LIMIT $limit
            """, limit=limit)
            
            return [dict(r) for r in result]
    
    def update_scene_status(self, scene_id: str, status: str, notes: str = "") -> bool:
        """Update a scene's status after review."""
        if not self.driver:
            return False
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Scene {id: $id})
                SET s.status = $status,
                    s.review_notes = $notes,
                    s.reviewed_at = datetime()
                RETURN s.id
            """, id=scene_id, status=status, notes=notes)
            
            return result.single() is not None
    
    def get_scenes_by_character(self, character_name: str, min_quality: float = 0.0) -> list[dict]:
        """Get all scenes featuring a character."""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Scene)-[:FEATURES]->(c:Character)
                WHERE toLower(c.name) CONTAINS toLower($name)
                  AND s.quality_score >= $min_quality
                RETURN s.id as id, s.summary as summary, s.text as text,
                       s.quality_score as score, c.name as character
                ORDER BY s.quality_score DESC
            """, name=character_name, min_quality=min_quality)
            
            return [dict(r) for r in result]
