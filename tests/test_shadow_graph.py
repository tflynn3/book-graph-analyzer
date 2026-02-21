"""Tests for Shadow Graph story state tracking."""

import json
import pytest
from unittest.mock import MagicMock, patch, call


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_mock_driver(query_results: dict = None):
    """Build a mock Neo4j driver with configurable query results."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    # Default: all queries return nothing
    result = MagicMock()
    result.single.return_value = None
    result.__iter__ = MagicMock(return_value=iter([]))
    session.run.return_value = result

    return driver, session


# ─── ShadowGraph: commit_state_delta ─────────────────────────────────────────

class TestShadowGraphCommitDelta:
    def test_commit_basic_location_change(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph
        from book_graph_analyzer.generate.shadow.models import StateDelta

        driver, session = make_mock_driver()
        sg = ShadowGraph(story_id="test-story", driver=driver)

        delta = StateDelta(
            story_id="test-story",
            scene_id="scene-001",
            character_updates={
                "Tuor": {
                    "location_change": "Nevrast Shore",
                    "possessions_gained": ["Sword of Turgon"],
                    "possessions_lost": [],
                    "conditions_added": ["weary"],
                    "conditions_removed": [],
                }
            },
            scene_summary="Tuor arrived at the shore and found the arms.",
            chapter_num=1,
            scene_num=1,
        )

        sg.commit_state_delta(delta)

        # Should have called session.run at least for: MERGE_CHARACTER, SET_LOCATION,
        # ADD_POSSESSION, ADD_CONDITION, RECORD_SCENE, LINK_CHARACTER_SCENE
        assert session.run.call_count >= 5

    def test_commit_handles_driver_none_gracefully(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph
        from book_graph_analyzer.generate.shadow.models import StateDelta

        # No driver — should not raise
        sg = ShadowGraph(story_id="test-story", driver=None)
        delta = StateDelta(story_id="test-story", scene_id="s1")

        sg.commit_state_delta(delta)  # Should not raise

    def test_commit_handles_neo4j_exception_gracefully(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph
        from book_graph_analyzer.generate.shadow.models import StateDelta

        driver, session = make_mock_driver()
        session.run.side_effect = Exception("Neo4j connection lost")

        sg = ShadowGraph(story_id="test-story", driver=driver)
        delta = StateDelta(
            story_id="test-story",
            scene_id="s1",
            character_updates={"Tuor": {"location_change": "Gondolin"}},
        )

        # Should NOT raise — non-blocking
        sg.commit_state_delta(delta)


# ─── ShadowGraph: get_character_state ────────────────────────────────────────

class TestShadowGraphGetCharacterState:
    def test_returns_none_when_no_driver(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph

        sg = ShadowGraph(story_id="test-story", driver=None)
        assert sg.get_character_state("Tuor") is None

    def test_returns_none_when_character_not_found(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph

        driver, session = make_mock_driver()
        session.run.return_value.single.return_value = None

        sg = ShadowGraph(story_id="test-story", driver=driver)
        assert sg.get_character_state("Tuor") is None

    def test_returns_character_state_from_graph(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph

        driver, session = make_mock_driver()
        mock_record = {
            "name": "Tuor",
            "location": "Gondolin",
            "possessions": ["Sword of Turgon", "Grey Cloak"],
            "conditions": ["weary"],
        }
        session.run.return_value.single.return_value = mock_record

        sg = ShadowGraph(story_id="test-story", driver=driver)
        state = sg.get_character_state("Tuor")

        assert state is not None
        assert state.name == "Tuor"
        assert state.location == "Gondolin"
        assert "Sword of Turgon" in state.possessions
        assert "weary" in state.conditions

    def test_story_id_isolation(self):
        """Two stories don't share character state."""
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph

        driver, session = make_mock_driver()
        session.run.return_value.single.return_value = None

        sg1 = ShadowGraph(story_id="story-A", driver=driver)
        sg2 = ShadowGraph(story_id="story-B", driver=driver)

        sg1.get_character_state("Tuor")
        sg2.get_character_state("Tuor")

        # Both calls should pass their own story_id
        calls = session.run.call_args_list
        story_ids_used = [c.kwargs.get("story_id") or (c.args[1] if len(c.args) > 1 else None)
                          for c in calls]
        # At minimum, the two story IDs should both appear
        assert "story-A" in str(calls)
        assert "story-B" in str(calls)


# ─── ShadowGraph: extract_delta_from_scene ───────────────────────────────────

class TestShadowGraphExtractDelta:
    def test_extract_returns_empty_delta_on_parse_failure(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph

        driver, _ = make_mock_driver()
        sg = ShadowGraph(story_id="test-story", driver=driver)

        # Mock LLM returning garbage
        sg._llm = MagicMock()
        sg._llm.generate.return_value = "This is not JSON at all."

        delta = sg.extract_delta_from_scene(
            scene_text="Tuor walked into the rain.",
            characters=["Tuor"],
            scene_id="s-001",
        )

        # Should return empty delta, not raise
        assert delta.story_id == "test-story"
        assert delta.scene_id == "s-001"
        assert delta.character_updates == {}

    def test_extract_parses_valid_json_response(self):
        from book_graph_analyzer.generate.shadow.graph import ShadowGraph

        driver, _ = make_mock_driver()
        sg = ShadowGraph(story_id="test-story", driver=driver)

        mock_response = json.dumps({
            "characters": {
                "Tuor": {
                    "location_change": "Nevrast",
                    "possessions_gained": ["Turgon's Helm"],
                    "possessions_lost": [],
                    "conditions_added": ["determined"],
                    "conditions_removed": [],
                }
            },
            "new_entities": [],
            "scene_summary": "Tuor found the arms of Turgon on the shore.",
        })

        sg._llm = MagicMock()
        sg._llm.generate.return_value = mock_response

        delta = sg.extract_delta_from_scene(
            scene_text="He found the helm gleaming on the shore...",
            characters=["Tuor"],
            scene_id="s-002",
        )

        assert "Tuor" in delta.character_updates
        assert delta.character_updates["Tuor"]["location_change"] == "Nevrast"
        assert "Turgon's Helm" in delta.character_updates["Tuor"]["possessions_gained"]
        assert delta.scene_summary == "Tuor found the arms of Turgon on the shore."


# ─── SceneState: to_prompt_block ─────────────────────────────────────────────

class TestSceneState:
    def test_to_prompt_block_under_400_tokens(self):
        from book_graph_analyzer.generate.shadow.models import (
            SceneState, CharacterState, InventedEntity
        )

        state = SceneState(
            characters=[
                CharacterState(
                    name="Tuor",
                    story_id="s1",
                    location="Echoriath foothills",
                    possessions=["Sword of Turgon", "Grey Cloak"],
                    conditions=["weary", "determined"],
                ),
                CharacterState(
                    name="Voronwë",
                    story_id="s1",
                    location="Echoriath foothills",
                    possessions=[],
                    conditions=["resolute"],
                ),
            ],
            recent_summaries=[
                "Tuor and Voronwë fled an Orc ambush near the river.",
                "They discovered a hidden path marked by Ulmo's sign.",
            ],
            invented_entities=[
                InventedEntity(
                    type="RUINED_LOCATION",
                    name="Barad-wath",
                    description="A ruined watchtower of the First Age, half-buried in ice.",
                    story_id="s1",
                    scene_id="pre-draft",
                ),
            ],
        )

        block = state.to_prompt_block()
        # Rough token estimate: ~0.75 tokens per character
        estimated_tokens = len(block) * 0.75
        assert estimated_tokens < 400, f"Prompt block too long: ~{estimated_tokens:.0f} tokens"
        assert "Tuor" in block
        assert "Voronwë" in block
        assert "Barad-wath" in block

    def test_empty_state_returns_empty_string(self):
        from book_graph_analyzer.generate.shadow.models import SceneState
        state = SceneState()
        # Should not crash, should return empty/minimal
        block = state.to_prompt_block()
        assert isinstance(block, str)
