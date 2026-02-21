import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.generate.models import Chapter
from book_graph_analyzer.generate.outliner import CanonicalEvent, OutlinerEngine


class _FakeResult:
    def __init__(self, single=None, rows=None):
        self._single = single
        self._rows = rows or []

    def single(self):
        return self._single

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self):
        self._anchor_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **kwargs):
        if "ORDER BY score DESC" in query:
            self._anchor_calls += 1
            if self._anchor_calls == 1:
                return _FakeResult(single={
                    "id": "ev_a",
                    "description": "Tuor arrives in Nevrast",
                    "era": "First Age",
                    "year": 495,
                    "agent": "Tuor",
                    "source_book": "Unfinished Tales",
                    "score": 4,
                })
            return _FakeResult(single={
                "id": "ev_b",
                "description": "Tuor reaches Gondolin",
                "era": "First Age",
                "year": 496,
                "agent": "Tuor",
                "source_book": "Unfinished Tales",
                "score": 4,
            })

        if "KNOWN CANONICAL EVENTS" in query:
            return _FakeResult(rows=[])

        # gap-events query
        return _FakeResult(rows=[])


class _FakeDriver:
    def session(self):
        return _FakeSession()


class _FakeLLM:
    def generate(self, *_args, **_kwargs):
        return json.dumps(
            {
                "chapters": [
                    {
                        "number": 1,
                        "title": "The Empty Shore",
                        "beat": "Tuor finds Vinyamar and arms himself.",
                        "characters": ["Tuor"],
                        "setting": "Nevrast",
                        "canonical_constraint": "Tuor must find Turgon's arms.",
                        "plot_thread_opens": "Who watches from afar?",
                        "plot_thread_closes": None,
                    }
                ]
            }
        )

    def extract_json(self, response):
        return json.loads(response)


class _FakeOutlinerEngine:
    def load_world_bible(self, _path):
        return None

    def find_anchor_points(self, character, _a, _b):
        return (
            CanonicalEvent(id="a", description=f"{character} arrives", era="First Age", year=1),
            CanonicalEvent(id="b", description=f"{character} departs", era="First Age", year=2),
        )

    def generate_story_outline(self, anchor_a, anchor_b, num_chapters, character):
        class _Outline:
            id = "outline_test"

            def __init__(self):
                self.chapters = [{"number": 1}] * num_chapters

            def to_dict(self):
                return {
                    "id": self.id,
                    "character": character,
                    "anchor_a": anchor_a.__dict__,
                    "anchor_b": anchor_b.__dict__,
                    "chapters": [{"number": 1}],
                }

        return _Outline()


def test_chapter_to_dict_includes_outliner_fields():
    chapter = Chapter(
        id="ch1",
        number=1,
        canonical_constraint="Hard canon constraint",
        plot_thread_opens="Mystery starts",
        plot_thread_closes="Mystery closes",
    )

    payload = chapter.to_dict()
    assert payload["canonical_constraint"] == "Hard canon constraint"
    assert payload["plot_thread_opens"] == "Mystery starts"
    assert payload["plot_thread_closes"] == "Mystery closes"


def test_outliner_generate_story_outline_structured_output():
    engine = OutlinerEngine(llm=_FakeLLM(), driver=_FakeDriver())
    a, b = engine.find_anchor_points("Tuor", "arrives in Nevrast", "reaches Gondolin")

    outline = engine.generate_story_outline(a, b, num_chapters=1, character="Tuor")

    assert outline.anchor_a.id == "ev_a"
    assert outline.anchor_b.id == "ev_b"
    assert len(outline.chapters) == 1
    assert outline.chapters[0].canonical_constraint


def test_generate_outline_cli_writes_json(monkeypatch, tmp_path):
    import book_graph_analyzer.generate as gen_pkg

    monkeypatch.setattr(gen_pkg, "OutlinerEngine", _FakeOutlinerEngine)

    out = tmp_path / "outline.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate",
            "outline",
            "--character",
            "Tuor",
            "--from",
            "arrives in Nevrast",
            "--to",
            "reaches Gondolin",
            "--chapters",
            "3",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["character"] == "Tuor"
