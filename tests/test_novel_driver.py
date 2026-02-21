import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.generate.driver import NovelDriver
from book_graph_analyzer.generate.models import Story
from book_graph_analyzer.generate.outliner import CanonicalEvent, ChapterOutline, StoryOutline


class _FakeAssembler:
    def assemble(self, **_kwargs):
        from book_graph_analyzer.generate.context import AssembledContext

        return AssembledContext(
            recent_summaries=["Earlier events"],
            place_facts={"name": "Nevrast"},
        )


class _FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate_scene(self, **kwargs):
        self.calls += 1
        from book_graph_analyzer.generate.models import Scene, SceneScores

        return Scene(
            id=f"scene_{self.calls}",
            number=kwargs.get("scene_num", 0),
            text=f"Generated scene text {self.calls}",
            summary=kwargs["scene_goal"],
            characters=kwargs["characters"],
            places=[kwargs["place"]],
            scores=SceneScores(lore_score=0.9, style_score=0.8, narrative_score=0.85, consistency_score=0.9, overall=0.86),
        )


class _FakeShadow:
    story_id = "outline_1"

    def extract_delta_from_scene(self, **_kwargs):
        from book_graph_analyzer.generate.shadow.models import StateDelta

        return StateDelta(story_id=self.story_id, scene_id="delta_scene", scene_summary="delta")

    def commit_state_delta(self, _delta):
        return None


class _FakeSceneGeneratorForCli:
    def __init__(self, shadow_graph=None):
        self.driver = None

    def load_world_bible(self, _path):
        return None


class _FakeNovelDriverForCli:
    def __init__(self, **_kwargs):
        pass

    def generate_novel(self, outline, resume=True):
        _ = resume
        return Story(id=outline.id, title="Stub Story")


def test_novel_driver_resume_skips_completed_scene(tmp_path):
    outline = StoryOutline(
        id="outline_1",
        character="Tuor",
        anchor_a=CanonicalEvent(id="a", description="A"),
        anchor_b=CanonicalEvent(id="b", description="B"),
        chapters=[
            ChapterOutline(
                number=1,
                title="Chapter 1",
                beat=json.dumps(
                    {
                        "scenes": [
                            {"scene": 1, "goal": "first", "setting": "Nevrast", "characters": ["Tuor"]},
                            {"scene": 2, "goal": "second", "setting": "Nevrast", "characters": ["Tuor"]},
                        ]
                    }
                ),
                characters=["Tuor"],
                setting="Nevrast",
            )
        ],
    )

    driver = NovelDriver(
        scene_generator=_FakeGenerator(),
        context_assembler=_FakeAssembler(),
        shadow_graph=_FakeShadow(),
        checkpoint_dir=str(tmp_path),
    )

    story = driver.generate_novel(outline, resume=False)
    assert len(story.chapters[0].scenes) == 2

    # Resume should skip both existing scenes
    fake_generator = _FakeGenerator()
    driver2 = NovelDriver(
        scene_generator=fake_generator,
        context_assembler=_FakeAssembler(),
        shadow_graph=_FakeShadow(),
        checkpoint_dir=str(tmp_path),
    )
    resumed = driver2.generate_novel(outline, resume=True)
    assert len(resumed.chapters[0].scenes) == 2
    assert fake_generator.calls == 0


def test_generate_novel_cli_writes_output(monkeypatch, tmp_path):
    import book_graph_analyzer.generate as gen_pkg

    monkeypatch.setattr(gen_pkg, "SceneGenerator", _FakeSceneGeneratorForCli)
    monkeypatch.setattr(gen_pkg, "NovelDriver", _FakeNovelDriverForCli)

    outline_path = tmp_path / "outline.json"
    outline_path.write_text(
        json.dumps(
            {
                "id": "outline_cli",
                "character": "Tuor",
                "anchor_a": {"id": "a", "description": "A"},
                "anchor_b": {"id": "b", "description": "B"},
                "chapters": [{"number": 1, "title": "Ch", "beat": "Beat"}],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "story.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate",
            "novel",
            "--outline",
            str(outline_path),
            "--checkpoint",
            str(tmp_path / "cp"),
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["id"] == "outline_cli"
