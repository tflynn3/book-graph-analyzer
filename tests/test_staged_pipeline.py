import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from book_graph_analyzer.generate.models import GenerationConfig, Scene
from book_graph_analyzer.generate.pipeline import StagedPipeline
from book_graph_analyzer.generate.voice_patcher import VoicePatcher
from book_graph_analyzer.voice.profile import CharacterVoiceProfile


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate(self, _prompt: str, temperature: float = 0.4):
        _ = temperature
        self.calls += 1
        return self.response


class _FakeSceneGenerator:
    def __init__(self):
        self.lore_calls = 0
        self.world_bible = None

    def _run_lore_enforcement(self, scene, _context):
        self.lore_calls += 1
        scene.revision_count += 1
        return scene, [{"severity": "major", "description": "anachronism"}]


def test_scene_round_trip_preserves_pipeline_stages():
    scene = Scene(id="s1", number=1, text="hello", pipeline_stages_run=["drafter", "voice_patch"])
    restored = Scene.from_dict(scene.to_dict())
    assert restored.pipeline_stages_run == ["drafter", "voice_patch"]


def test_voice_patcher_triggers_on_modern_contractions_for_melkor():
    llm = _FakeLLM("1. I care not.\n2. We shall prevail.")
    patcher = VoicePatcher(llm_client=llm)

    scene = Scene(
        id="melkor_1",
        number=1,
        text='"I don\'t care," said Melkor. "We\'re gonna win," said Melkor.',
    )
    profile = CharacterVoiceProfile(
        character_name="Melkor",
        contraction_ratio=0.0,
        formality_score=0.95,
        avg_utterance_length=4.0,
    )

    deviation = patcher.estimate_max_deviation(scene, {"Melkor": profile})
    assert deviation >= 0.25

    patched = patcher.patch(scene, {"Melkor": profile}, threshold=0.25)
    assert "don't" not in patched.text.lower()
    assert "we're" not in patched.text.lower()
    assert llm.calls == 1


def test_voice_patcher_skips_galadriel_when_already_formal():
    llm = _FakeLLM("unused")
    patcher = VoicePatcher(llm_client=llm)

    scene = Scene(
        id="galadriel_1",
        number=1,
        text='"I shall depart ere dawn," said Galadriel.',
    )
    profile = CharacterVoiceProfile(
        character_name="Galadriel",
        contraction_ratio=0.0,
        formality_score=0.9,
        avg_utterance_length=5.0,
    )

    deviation = patcher.estimate_max_deviation(scene, {"Galadriel": profile})
    assert deviation < 0.25

    patched = patcher.patch(scene, {"Galadriel": profile}, threshold=0.25)
    assert patched.text == scene.text
    assert llm.calls == 0


def test_pipeline_skips_lore_stage_for_low_risk_scene():
    fake_generator = _FakeSceneGenerator()
    pipeline = StagedPipeline(
        scene_generator=fake_generator,
        voice_patcher=VoicePatcher(llm_client=None),
        config=GenerationConfig(lore_enforce_only_major=True, enable_voice_patch=False),
    )

    scene = Scene(id="safe", number=1, text="The wind moved in the pines.")
    scene, violations = pipeline.run(scene, neo4j_context={}, voice_profiles={})

    assert fake_generator.lore_calls == 0
    assert violations == []
    assert scene.pipeline_stages_run == ["drafter"]


def test_pipeline_runs_and_records_lore_stage_when_world_bible_is_loaded():
    fake_generator = _FakeSceneGenerator()
    fake_generator.world_bible = SimpleNamespace(rules={"history": [object()]})
    pipeline = StagedPipeline(
        scene_generator=fake_generator,
        voice_patcher=VoicePatcher(llm_client=None),
        config=GenerationConfig(lore_enforce_only_major=True, enable_voice_patch=False),
    )

    scene = Scene(id="grounded", number=1, text="The wind moved in the pines.")
    scene, violations = pipeline.run(scene, neo4j_context={}, voice_profiles={})

    assert fake_generator.lore_calls == 1
    assert violations
    assert scene.pipeline_stages_run == ["drafter", "lore_enforce"]


def test_pipeline_does_not_treat_placeholder_bible_or_blank_state_as_lore_evidence():
    from book_graph_analyzer.generate.context import AssembledContext
    from book_graph_analyzer.generate.shadow.models import CharacterState
    from book_graph_analyzer.worldbible.models import WorldBible

    fake_generator = _FakeSceneGenerator()
    fake_generator.world_bible = WorldBible.from_markdown(
        "# Story Bible\n\n## World Rules\n- (add non-negotiable rules)\n"
    )
    pipeline = StagedPipeline(
        scene_generator=fake_generator,
        voice_patcher=VoicePatcher(llm_client=None),
        config=GenerationConfig(lore_enforce_only_major=True, enable_voice_patch=False),
    )
    scene = Scene(
        id="placeholder-bible",
        number=1,
        text="The wind moved in the pines.",
        context_snapshot=AssembledContext(
            character_states=[CharacterState(name="Beren", story_id="story-1")],
            place_facts={"name": "Beleriand"},
        ),
    )

    scene, violations = pipeline.run(scene, neo4j_context={}, voice_profiles={})

    assert fake_generator.lore_calls == 0
    assert violations == []
    assert scene.pipeline_stages_run == ["drafter"]


def _generator_with_real_lore_pipeline(critic_response: str):
    from book_graph_analyzer.generate.generator import SceneGenerator
    from book_graph_analyzer.generate.models import SceneScores
    from book_graph_analyzer.worldbible.models import WorldBible

    generator = SceneGenerator(
        config=GenerationConfig(
            lore_enforce_only_major=True,
            enable_voice_patch=False,
            max_critique_iterations=1,
        )
    )
    generator.world_bible = WorldBible.from_markdown(
        "# Middle-earth\n\n## World Rules\n- Orcs do not use pistols.\n"
    )
    generator.llm = _FakeLLM(critic_response)
    generator.judge = MagicMock()
    generator.judge.full_evaluation.return_value = (
        SceneScores(style_score=0.8, narrative_score=0.8),
        "",
        [],
    )
    return generator


def test_lore_parse_failure_is_unverified_and_scores_zero():
    generator = _generator_with_real_lore_pipeline("This is not JSON.")
    scene = Scene(id="bad-critic", number=1, text="An orc waited in the pass.")

    scene, violations = generator.pipeline.run(scene, neo4j_context={}, voice_profiles={})
    scores = generator._score_scene(
        scene,
        context="",
        lore_violations=violations,
        voice_profiles={},
    )

    assert violations == []
    assert scene.pipeline_stages_run == ["drafter", "lore_enforce_unverified"]
    assert scores.lore_score == 0.0
    assert any("could not be parsed" in note.lower() for note in scene.critique_notes)


def test_minor_lore_violation_does_not_earn_perfect_score():
    generator = _generator_with_real_lore_pipeline(
        json.dumps(
            {
                "violations": [
                    {
                        "type": "language",
                        "description": "The phrasing is too modern.",
                        "severity": "minor",
                    }
                ],
                "lore_score": 0.9,
            }
        )
    )
    scene = Scene(id="minor", number=1, text="An orc waited in the pass.")

    scene, violations = generator.pipeline.run(scene, neo4j_context={}, voice_profiles={})
    scores = generator._score_scene(
        scene,
        context="",
        lore_violations=violations,
        voice_profiles={},
    )

    assert violations[0]["severity"] == "minor"
    assert scene.pipeline_stages_run == ["drafter", "lore_enforce"]
    assert scores.lore_score == 0.9


def test_revised_scene_is_rechecked_and_resolved_violation_is_not_scored():
    generator = _generator_with_real_lore_pipeline("unused")
    major = {
        "type": "anachronism",
        "description": "A pistol appears in Middle-earth.",
        "severity": "major",
    }
    generator._critique_scene = MagicMock(
        side_effect=[([major], True), ([], True)]
    )
    generator._revise_scene = MagicMock(return_value="An orc waited in the pass.")
    scene = Scene(id="rechecked", number=1, text="An orc raised a pistol.")

    scene, violations = generator.pipeline.run(scene, neo4j_context={}, voice_profiles={})
    scores = generator._score_scene(
        scene,
        context="",
        lore_violations=violations,
        voice_profiles={},
    )

    assert generator._critique_scene.call_count == 2
    assert generator._revise_scene.call_count == 1
    assert scene.revision_count == 1
    assert violations == []
    assert scores.lore_score == 1.0
