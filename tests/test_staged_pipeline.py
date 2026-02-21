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
