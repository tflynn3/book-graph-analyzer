"""Tests for the Lore Incubator and Trope Dictionary."""

import json
import pytest
from unittest.mock import MagicMock, patch


# ─── TropeDictionary ─────────────────────────────────────────────────────────

class TestTropeDictionary:
    def test_has_minimum_tropes(self):
        from book_graph_analyzer.generate.incubator import TropeDictionary
        td = TropeDictionary()
        assert len(td.TROPES) >= 6

    def test_select_returns_a_trope(self):
        from book_graph_analyzer.generate.incubator import TropeDictionary
        td = TropeDictionary()
        trope = td.select_trope()
        assert "name" in trope
        assert "description" in trope
        assert "required_elements" in trope

    def test_avoids_recently_used_tropes(self):
        from book_graph_analyzer.generate.incubator import TropeDictionary
        td = TropeDictionary()

        used = []
        for _ in range(len(td.TROPES)):
            trope = td.select_trope(used_tropes=used)
            # Should not repeat as long as pool isn't exhausted
            if len(used) < len(td.TROPES) - 1:
                assert trope["name"] not in used, \
                    f"Trope '{trope['name']}' was reused before pool exhausted"
            used.append(trope["name"])

    def test_resets_when_all_used(self):
        """When all tropes have been used, select_trope should still return one."""
        from book_graph_analyzer.generate.incubator import TropeDictionary
        td = TropeDictionary()
        all_names = [t["name"] for t in td.TROPES]

        # All tropes "used" — should still return something
        result = td.select_trope(used_tropes=all_names)
        assert result is not None
        assert "name" in result

    def test_journey_beat_prefers_journey_trope(self):
        from book_graph_analyzer.generate.incubator import TropeDictionary
        td = TropeDictionary()

        # Run multiple times and check that journey-hinted tropes appear
        journey_trope_names = [
            t["name"] for t in td.TROPES
            if t.get("scene_type_hint") == "journey"
        ]
        if not journey_trope_names:
            pytest.skip("No journey tropes defined")

        results = [td.select_trope(chapter_beat="Tuor travels through the wilderness on a long road") for _ in range(20)]
        selected_names = [r["name"] for r in results]
        assert any(name in selected_names for name in journey_trope_names), \
            "Journey tropes should appear when beat mentions travel"

    def test_get_all_returns_complete_list(self):
        from book_graph_analyzer.generate.incubator import TropeDictionary
        td = TropeDictionary()
        assert td.get_all() == td.TROPES


# ─── LoreIncubator ───────────────────────────────────────────────────────────

def make_mock_shadow_graph():
    sg = MagicMock()
    sg.get_invented_entities.return_value = []
    return sg


def make_valid_invention_response() -> str:
    return json.dumps({
        "invented_entities": [
            {
                "type": "MINOR_CHARACTER",
                "name": "Cabed Norn",
                "race": "Elf",
                "description": "A lone Sindarin scout, last survivor of a ruined garrison.",
                "tragic_history": "Watched his kin fall to Orcs at the ford of Ringlin.",
                "role_in_story": "Reluctant guide through the mountain pass.",
                "trope_connection": "The Reluctant Guide.",
            },
            {
                "type": "RUINED_LOCATION",
                "name": "Barad-wath",
                "region": "Western Echoriath",
                "description": "A shattered watchtower, its stones split by ancient cold.",
                "former_purpose": "Guarded the hidden western approach to Gondolin.",
                "how_it_fell": "Destroyed by a Balrog in the Year of Lamentation.",
                "what_remains": "A broken arch and a descent into darkness.",
            },
            {
                "type": "ARTIFACT",
                "name": "Mornbrand",
                "material": "Dark iron with a silver edge",
                "description": "A short blade of Noldorin make, cold to the touch.",
                "age_of_origin": "First Age, before the Nirnaeth",
                "tragic_history": "Carried by a slain captain; abandoned in the dark.",
                "power_or_property": "Glows faintly cold near hidden passages.",
                "trope_connection": "Relic in the Dark — found in the ruin's descent.",
            },
        ],
        "narrative_seeds": [
            "Cabed Norn emerges from shadow at the ford, blocking Tuor's path.",
            "The shattered arch of Barad-wath is the only landmark visible through the snow.",
            "In the darkness beneath the tower, Tuor's foot strikes something cold and hard.",
        ],
    })


class TestLoreIncubator:
    def test_incubate_returns_result_with_three_entities(self):
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.return_value = make_valid_invention_response()

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)
        result = incubator.incubate(
            journey_context="Tuor traveling from Nevrast to Gondolin",
            whitespace_description="400-mile gap of unnamed wilderness",
        )

        assert len(result.invented_entities) == 3
        types = {e["type"] for e in result.invented_entities}
        assert "MINOR_CHARACTER" in types
        assert "RUINED_LOCATION" in types
        assert "ARTIFACT" in types

    def test_incubate_populates_narrative_seeds(self):
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.return_value = make_valid_invention_response()

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)
        result = incubator.incubate("context", "whitespace")

        assert len(result.narrative_seeds) == 3
        assert all(isinstance(s, str) for s in result.narrative_seeds)

    def test_incubate_returns_empty_result_on_llm_failure(self):
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.side_effect = Exception("LLM unavailable")

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)
        result = incubator.incubate("context", "whitespace")

        # Should return graceful empty result, not raise
        assert result is not None
        assert result.invented_entities == []

    def test_incubate_returns_empty_result_on_bad_json(self):
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.return_value = "Sorry, I cannot help with that."

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)
        result = incubator.incubate("context", "whitespace")

        assert result.invented_entities == []

    def test_commit_to_shadow_calls_graph_for_each_entity(self):
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.return_value = make_valid_invention_response()

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)
        result = incubator.incubate("context", "whitespace")
        incubator.commit_to_shadow(result, scene_id="chapter-1-pre")

        assert sg.commit_invented_entity.call_count == 3

    def test_incubate_and_commit_convenience_method(self):
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.return_value = make_valid_invention_response()

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)
        result = incubator.incubate_and_commit(
            "Tuor's journey",
            "The wilderness gap",
            chapter_id="ch-1",
        )

        assert len(result.invented_entities) == 3
        assert sg.commit_invented_entity.call_count == 3

    def test_trope_recorded_in_result(self):
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.return_value = make_valid_invention_response()

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)
        result = incubator.incubate("context", "whitespace")

        assert result.trope_used is not None
        assert "name" in result.trope_used

    def test_used_tropes_tracked_across_calls(self):
        """Each call to incubate records the trope used."""
        from book_graph_analyzer.generate.incubator import LoreIncubator

        sg = make_mock_shadow_graph()
        llm = MagicMock()
        llm.generate.return_value = make_valid_invention_response()

        incubator = LoreIncubator(shadow_graph=sg, llm_client=llm)

        incubator.incubate("context", "whitespace 1")
        incubator.incubate("context", "whitespace 2")

        assert len(incubator._used_tropes) == 2


# ─── IncubationResult: summary ───────────────────────────────────────────────

class TestIncubationResult:
    def test_summary_includes_entity_names(self):
        from book_graph_analyzer.generate.incubator import IncubationResult

        result = IncubationResult(
            invented_entities=[
                {"type": "MINOR_CHARACTER", "name": "Cabed Norn", "description": "A scout."},
                {"type": "RUINED_LOCATION", "name": "Barad-wath", "description": "A tower."},
            ],
            narrative_seeds=["He appeared at the ford."],
            trope_used={"name": "The Reluctant Guide"},
        )

        summary = result.summary()
        assert "Cabed Norn" in summary
        assert "Barad-wath" in summary
        assert "The Reluctant Guide" in summary
