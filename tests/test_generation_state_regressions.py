"""Regression tests for generation context and mutable shadow state."""

import json
from unittest.mock import MagicMock


def _mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    return driver, session


def test_assembled_context_character_states_round_trip():
    from book_graph_analyzer.generate.context import AssembledContext
    from book_graph_analyzer.generate.shadow.models import CharacterState

    original = AssembledContext(
        character_states=[
            CharacterState(
                name="Aragorn",
                story_id="hunt-for-gollum",
                location="The Angle",
                possessions=["Anduril"],
                conditions=["weary", "watchful"],
                last_scene="chapter-02-scene-03",
            )
        ],
        recent_summaries=["Aragorn crossed the Hoarwell."],
        place_facts={"name": "The Angle"},
        active_plot_threads=["Find Gollum before the Enemy does."],
    )

    restored = AssembledContext.from_dict(original.to_dict())

    assert restored == original
    assert restored.character_states[0] is not original.character_states[0]


def test_context_assembler_passes_current_scene_boundary_to_shadow_graph():
    from book_graph_analyzer.generate.context import ContextAssembler
    from book_graph_analyzer.generate.shadow.models import SceneState

    shadow_graph = MagicMock()
    shadow_graph.story_id = "story-1"
    shadow_graph.get_scene_state.return_value = SceneState()

    assembler = ContextAssembler(shadow_graph=shadow_graph)
    assembler.assemble(
        story_id="story-1",
        characters=["Aragorn"],
        place="The Angle",
        chapter_num=4,
        scene_num=2,
    )

    shadow_graph.get_scene_state.assert_called_once_with(
        characters=["Aragorn"],
        place="The Angle",
        chapter_num=4,
        scene_num=2,
    )


def test_context_assembler_leaves_scene_boundary_open_when_numbers_are_unknown():
    from book_graph_analyzer.generate.context import ContextAssembler
    from book_graph_analyzer.generate.shadow.models import SceneState

    shadow_graph = MagicMock()
    shadow_graph.story_id = "story-1"
    shadow_graph.get_scene_state.return_value = SceneState()

    assembler = ContextAssembler(shadow_graph=shadow_graph)
    assembler.assemble(
        story_id="story-1",
        characters=["Aragorn"],
        place="The Angle",
        chapter_num=0,
        scene_num=0,
    )

    shadow_graph.get_scene_state.assert_called_once_with(
        characters=["Aragorn"],
        place="The Angle",
        chapter_num=None,
        scene_num=None,
    )


def test_location_change_removes_every_previous_location_relation():
    from book_graph_analyzer.generate.shadow.graph import ShadowGraph
    from book_graph_analyzer.generate.shadow.models import StateDelta

    driver, session = _mock_driver()
    graph = ShadowGraph(story_id="story-1", driver=driver)

    graph.commit_state_delta(
        StateDelta(
            story_id="story-1",
            scene_id="chapter-01-scene-02",
            character_updates={"Aragorn": {"location_change": "The Angle"}},
        )
    )

    location_queries = [
        args.args[0]
        for args in session.run.call_args_list
        if "LOCATED_AT" in args.args[0] and "Shadow_Character" in args.args[0]
    ]
    assert len(location_queries) == 1
    assert "OPTIONAL MATCH" in location_queries[0]
    assert "DELETE relation" in location_queries[0]


def test_recent_summaries_select_latest_before_scene_then_return_chronologically():
    from book_graph_analyzer.generate.shadow.graph import ShadowGraph

    driver, session = _mock_driver()
    result = MagicMock()
    result.__iter__.return_value = iter(
        [
            {"summary": "Scene 3"},
            {"summary": "Scene 2"},
            {"summary": "Scene 1"},
        ]
    )
    session.run.return_value = result
    graph = ShadowGraph(story_id="story-1", driver=driver)

    summaries = graph._get_recent_summaries(
        limit=3,
        chapter_num=2,
        scene_num=4,
    )

    query = session.run.call_args.args[0]
    assert "ORDER BY coalesce(s.chapter_num, 0) DESC" in query
    assert "coalesce(s.scene_num, 0) < $scene_num" in query
    assert session.run.call_args.kwargs == {
        "story_id": "story-1",
        "chapter_num": 2,
        "scene_num": 4,
        "limit": 3,
    }
    assert summaries == ["Scene 1", "Scene 2", "Scene 3"]


def test_delta_extraction_uses_complete_scene_by_default():
    from book_graph_analyzer.generate.shadow.graph import ShadowGraph

    driver, _ = _mock_driver()
    graph = ShadowGraph(story_id="story-1", driver=driver)
    graph._llm = MagicMock()
    graph._llm.generate.return_value = json.dumps(
        {"characters": {}, "new_entities": [], "scene_summary": "Complete."}
    )
    scene_text = "A" * 3_200 + "FINAL_STATE_CHANGE"

    graph.extract_delta_from_scene(
        scene_text=scene_text,
        characters=["Aragorn"],
        scene_id="chapter-01-scene-01",
    )

    prompt = graph._llm.generate.call_args.args[0]
    assert "FINAL_STATE_CHANGE" in prompt


def test_delta_extraction_supports_an_explicit_character_cap():
    from book_graph_analyzer.generate.shadow.graph import ShadowGraph

    driver, _ = _mock_driver()
    graph = ShadowGraph(story_id="story-1", driver=driver, delta_max_chars=12)
    graph._llm = MagicMock()
    graph._llm.generate.return_value = json.dumps(
        {"characters": {}, "new_entities": [], "scene_summary": "Capped."}
    )

    graph.extract_delta_from_scene(
        scene_text="beginning---FINAL_STATE_CHANGE",
        characters=["Aragorn"],
        scene_id="chapter-01-scene-01",
    )

    prompt = graph._llm.generate.call_args.args[0]
    assert "beginning---" in prompt
    assert "FINAL_STATE_CHANGE" not in prompt
