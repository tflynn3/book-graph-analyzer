import json

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.lore.events import Event, EventGraph


class _ExtractorBase:
    instances = []

    def __init__(self, use_llm=True, progress_callback=None):
        self.use_llm = use_llm
        self.progress_callback = progress_callback
        self.finalized = []
        type(self).instances.append(self)

    def get_resilient_summary(self, _checkpoint):
        return {"ok": 1, "retried": 0, "fallback_success": 0, "failed": 0}

    def _load_checkpoint(self, checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def graph_from_checkpoint_payload(self, events_payload, relations_payload):
        g = EventGraph()
        for e in events_payload:
            g.add_event(Event(**e))
        return g

    def finalize_checkpoint(self, checkpoint_file):
        self.finalized.append(checkpoint_file)


class _ExtractorWithData(_ExtractorBase):
    def extract_from_book(self, *_args, **_kwargs):
        g = EventGraph()
        g.add_event(Event(id="e1", description="Bilbo found ring"))
        return g


class _ExtractorEmpty(_ExtractorBase):
    def extract_from_book(self, *_args, **_kwargs):
        return EventGraph()


def _patch_common(monkeypatch, text):
    monkeypatch.setattr("book_graph_analyzer.ingest.loader.load_book", lambda *_args, **_kwargs: text)


def test_lore_events_checkpoint_not_finalized_when_neo4j_write_fails(tmp_path, monkeypatch):
    _ExtractorWithData.instances = []
    input_file = tmp_path / "hobbit.txt"
    input_file.write_text("x", encoding="utf-8")
    checkpoint = tmp_path / "hobbit.checkpoint.json"

    _patch_common(monkeypatch, "A" * 9000)
    monkeypatch.setattr("book_graph_analyzer.lore.EventExtractor", _ExtractorWithData)
    monkeypatch.setattr("book_graph_analyzer.graph.connection.check_neo4j_connection", lambda: True)

    class _FailingWriter:
        def write_event_graph(self, *args, **kwargs):
            raise RuntimeError("neo4j write failed")

        def close(self):
            pass

    monkeypatch.setattr("book_graph_analyzer.graph.writer.GraphWriter", _FailingWriter)
    monkeypatch.setattr("book_graph_analyzer.llm.LLMClient", lambda: type("L", (), {"provider": "openai", "model": "gpt-4o-mini"})())

    result = CliRunner().invoke(
        main,
        [
            "lore",
            "events",
            str(input_file),
            "-o",
            str(tmp_path / "out.json"),
            "--resilient",
            "-c",
            str(checkpoint),
            "--neo4j",
            "--chunk-size",
            "1000",
        ],
    )

    assert result.exit_code != 0
    assert _ExtractorWithData.instances[-1].finalized == []


def test_lore_events_recovers_nonempty_checkpoint_before_finalization(tmp_path, monkeypatch):
    _ExtractorEmpty.instances = []
    input_file = tmp_path / "hobbit.txt"
    input_file.write_text("x", encoding="utf-8")
    output = tmp_path / "out.json"
    checkpoint = tmp_path / "hobbit.checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "events": [{"id": "cp1", "description": "Recovered event"}],
                "relations": [],
            }
        ),
        encoding="utf-8",
    )

    _patch_common(monkeypatch, "A" * 9000)
    monkeypatch.setattr("book_graph_analyzer.lore.EventExtractor", _ExtractorEmpty)
    monkeypatch.setattr("book_graph_analyzer.llm.LLMClient", lambda: type("L", (), {"provider": "openai", "model": "gpt-4o-mini"})())

    result = CliRunner().invoke(
        main,
        [
            "lore",
            "events",
            str(input_file),
            "-o",
            str(output),
            "--resilient",
            "-c",
            str(checkpoint),
            "--chunk-size",
            "1000",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["events"]) == 1
    assert data["events"]["cp1"]["id"] == "cp1"
    assert _ExtractorEmpty.instances[-1].finalized == [str(checkpoint)]
