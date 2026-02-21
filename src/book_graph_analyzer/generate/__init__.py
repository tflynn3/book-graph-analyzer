"""Story generation module.

Generates lore-consistent narrative content using:
- Neo4j knowledge graph for context
- Constitutional critique for constraint enforcement
- LLM-as-judge for narrative quality scoring
"""

from .models import Story, Chapter, Scene, GenerationConfig
from .generator import SceneGenerator
from .judge import NarrativeJudge
from .writer import GenerationWriter
from .context import ContextAssembler, AssembledContext
from .outliner import OutlinerEngine, StoryOutline, ChapterOutline, CanonicalEvent
from .driver import NovelDriver
from .shadow.graph import ShadowGraph
from .pipeline import StagedPipeline
from .voice_patcher import VoicePatcher

__all__ = [
    "Story",
    "Chapter", 
    "Scene",
    "GenerationConfig",
    "SceneGenerator",
    "NarrativeJudge",
    "GenerationWriter",
    "ContextAssembler",
    "AssembledContext",
    "OutlinerEngine",
    "StoryOutline",
    "ChapterOutline",
    "CanonicalEvent",
    "NovelDriver",
    "ShadowGraph",
    "StagedPipeline",
    "VoicePatcher",
]
