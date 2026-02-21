from unittest.mock import MagicMock

from click.testing import CliRunner


def test_lore_depth_models_importable():
    from book_graph_analyzer.models import LoreArtifact, BrokenReference, LoreArtifactType

    a = LoreArtifact(id="a1", name="Lay of Leithian", artifact_type=LoreArtifactType.SONG)
    b = BrokenReference(id="u1", mention_text="[[lost blade]]")

    assert a.artifact_type.value == "song"
    assert b.mention_text == "[[lost blade]]"


def test_extract_lore_depth_finds_artifacts_and_broken_refs():
    from book_graph_analyzer.lore.depth import extract_lore_depth

    text = "They sang song of Beren and carried artifact Crown of Iron. [[missing name]] and unknown relic remained."
    result = extract_lore_depth(text, source_book="Silmarillion", passage_id="p1")

    assert len(result.artifacts) >= 1
    assert any(a.name for a in result.artifacts)
    assert len(result.broken_references) >= 1


def test_graph_writer_lore_depth_methods_exist_and_write():
    from book_graph_analyzer.graph.writer import GraphWriter
    from book_graph_analyzer.models.lore_depth import LoreArtifact, LoreArtifactType, BrokenReference

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    writer = GraphWriter(driver=mock_driver)

    a_count = writer.write_lore_artifacts_batch([
        LoreArtifact(id="a1", name="Lay of Leithian", artifact_type=LoreArtifactType.SONG)
    ])
    b_count = writer.write_broken_references_batch([
        BrokenReference(id="u1", mention_text="[[missing]]")
    ])

    assert a_count == 1
    assert b_count == 1
    assert mock_session.run.called


def test_cli_registers_new_commands():
    from book_graph_analyzer.cli import main

    runner = CliRunner()
    lore_help = runner.invoke(main, ["lore", "--help"])
    world_help = runner.invoke(main, ["worldbible", "--help"])

    assert "unresolved-refs" in lore_help.output
    assert "artifacts" in world_help.output
