from __future__ import annotations

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.lore.unresolved_resolution import (
    InventoryEntity,
    ResolutionSuggestion,
    candidate_shortlist,
    canonicalize_entity_name_for_display,
    group_materializable_new_entity_suggestions,
    is_safe_existing_auto_apply,
)


def test_candidate_shortlist_boosts_known_aliases():
    row = {
        "mention_text": "Strider",
        "context_text": "Frodo trusted Strider in Bree after Gandalf's warning.",
        "candidates": [],
    }
    inventory = [
        InventoryEntity(
            entity_id="aragorn",
            canonical_name="Aragorn",
            aliases=["Strider", "Mr. Strider"],
        ),
        InventoryEntity(entity_id="gandalf", canonical_name="Gandalf", aliases=["Mithrandir"]),
    ]

    shortlist = candidate_shortlist(row, inventory, unique_token_index={}, limit=3)

    assert shortlist
    assert shortlist[0] == "Aragorn"


def test_character_like_filter_accepts_canon_candidate_rows():
    from book_graph_analyzer.lore.unresolved_resolution import is_character_like_reference

    assert is_character_like_reference(
        {
            "reference_class": "canon_candidate",
            "expected_type": "character",
            "mention_text": "Strider",
            "context_text": "Frodo trusted Strider in Bree.",
        }
    )


def test_character_like_filter_rejects_section_noise():
    from book_graph_analyzer.lore.unresolved_resolution import is_character_like_reference

    assert not is_character_like_reference(
        {
            "reference_class": "canon_candidate",
            "expected_type": "character",
            "mention_text": "Chapter 3",
            "context_text": "Chapter 3 The Ring Goes South",
        }
    )


def test_character_like_filter_rejects_plural_groups():
    from book_graph_analyzer.lore.unresolved_resolution import is_character_like_reference

    assert not is_character_like_reference(
        {
            "reference_class": "canon_candidate",
            "expected_type": "character",
            "mention_text": "the Hobbits",
            "context_text": "the Hobbits lived in the Shire",
        }
    )


def test_safe_existing_auto_apply_requires_strong_name_match():
    assert is_safe_existing_auto_apply(
        "Mr. Bilbo",
        InventoryEntity(entity_id="bilbo_baggins", canonical_name="Bilbo Baggins"),
    )
    assert not is_safe_existing_auto_apply(
        "Miss Melilot Brandybuck",
        InventoryEntity(entity_id="merry", canonical_name="Meriadoc Brandybuck"),
    )


def test_group_materializable_new_entity_suggestions_requires_support():
    rows = [
        {
            "id": "u1",
            "mention_text": "Glorfindel",
            "source_book": "The Fellowship of the Ring",
            "llm_resolution_entity_name": "glorfindel",
            "llm_resolution_score": 0.6,
        },
        {
            "id": "u2",
            "mention_text": "Glorfindel",
            "source_book": "The Fellowship of the Ring",
            "llm_resolution_entity_name": "Glorfindel",
            "llm_resolution_score": 0.62,
        },
        {
            "id": "u3",
            "mention_text": "Goldberry",
            "source_book": "The Fellowship of the Ring",
            "llm_resolution_entity_name": "goldberry",
            "llm_resolution_score": 0.61,
        },
        {
            "id": "u4",
            "mention_text": "none",
            "source_book": "The Fellowship of the Ring",
            "llm_resolution_entity_name": "none",
            "llm_resolution_score": 0.7,
        },
    ]

    candidates = group_materializable_new_entity_suggestions(
        rows,
        min_support=2,
        min_score=0.6,
    )

    assert len(candidates) == 1
    assert candidates[0]["canonical_name"] == "Glorfindel"
    assert candidates[0]["entity_id"] == "glorfindel"
    assert candidates[0]["support"] == 2


def test_canonicalize_entity_name_for_display_titlecases_llm_names():
    assert canonicalize_entity_name_for_display("gildor inglorion") == "Gildor Inglorion"
    assert canonicalize_entity_name_for_display("tinúviel") == "Tinúviel"
    assert canonicalize_entity_name_for_display("none") == ""


def test_candidate_shortlist_uses_existing_candidate_scores():
    row = {
        "mention_text": "the Shadow",
        "context_text": "They feared the Shadow rising again in Mordor.",
        "candidates": [
            {"canonical_id": "sauron", "surface": "the Shadow", "confidence": 0.92},
            {"canonical_id": "saruman", "surface": "the wizard", "confidence": 0.35},
        ],
    }
    inventory = [
        InventoryEntity(entity_id="sauron", canonical_name="Sauron", aliases=["the Shadow"]),
        InventoryEntity(entity_id="saruman", canonical_name="Saruman", aliases=["Sharkey"]),
    ]

    shortlist = candidate_shortlist(row, inventory, unique_token_index={}, limit=2)

    assert shortlist == ["Sauron", "Saruman"]


def test_cli_resolve_unresolved_applies_existing_matches(monkeypatch, tmp_path):
    import book_graph_analyzer.graph.writer as writer_module
    import book_graph_analyzer.lore.unresolved_resolution as resolution_module

    class _FakeWriter:
        last_instance: _FakeWriter | None = None

        def __init__(self) -> None:
            self.written: list[dict] = []
            type(self).last_instance = self

        def query_character_inventory(self) -> list[dict]:
            return [
                {"entity_id": "aragorn", "canonical_name": "Aragorn", "aliases": ["Strider"]},
            ]

        def query_unresolved_reference_queue(
            self,
            source_book=None,
            limit: int = 100,
        ) -> list[dict]:
            return [
                {
                    "id": "u1",
                    "mention_text": "Strider",
                    "reference_class": "canon_candidate",
                    "expected_type": "character",
                    "source_book": "The Fellowship of the Ring",
                    "context_text": "Frodo looked at Strider in the inn.",
                    "candidates": [],
                },
                {
                    "id": "u2",
                    "mention_text": "the Hill",
                    "reference_class": "canon_candidate",
                    "expected_type": "place",
                    "source_book": "The Fellowship of the Ring",
                    "context_text": "They climbed the Hill above Hobbiton.",
                    "candidates": [],
                },
                {
                    "id": "u3",
                    "mention_text": "the Shadow",
                    "reference_class": "canon_candidate",
                    "expected_type": "character",
                    "source_book": "The Fellowship of the Ring",
                    "context_text": "They feared the Shadow rising again in Mordor.",
                    "llm_resolution_action": "reject",
                    "candidates": [],
                },
            ]

        def write_unresolved_resolution_suggestions(self, suggestions: list[dict]) -> int:
            self.written = list(suggestions)
            return len(self.written)

        def close(self) -> None:
            return None

    class _FakeResolver:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def resolve_batch(self, refs, inventory, *, apply_existing: bool = True):
            assert len(refs) == 1
            assert refs[0]["mention_text"] == "Strider"
            assert inventory[0].canonical_name == "Aragorn"
            return [
                ResolutionSuggestion(
                    ref_id="u1",
                    mention_text="Strider",
                    source_book="The Fellowship of the Ring",
                    reference_class="canon_candidate",
                    stage1_verdict="character",
                    action="existing",
                    entity_id="aragorn",
                    entity_name="Aragorn",
                    shortlist=["Aragorn"],
                    notes=["existing_from_shortlist"],
                    model="Qwen/Qwen2.5-72B-Instruct",
                    provider="auto",
                    applied=apply_existing,
                    score=0.85,
                )
            ]

    monkeypatch.setattr(writer_module, "GraphWriter", _FakeWriter)
    monkeypatch.setattr(resolution_module, "StagedHFUnresolvedResolver", _FakeResolver)

    report_path = tmp_path / "resolution-report.json"
    result = CliRunner().invoke(
        main,
        [
            "lore",
            "resolve-unresolved",
            "--limit",
            "5",
            "--json-out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Applied existing matches" in result.output
    assert _FakeWriter.last_instance is not None
    assert _FakeWriter.last_instance.written[0]["id"] == "u1"
    assert _FakeWriter.last_instance.written[0]["entity_id"] == "aragorn"
    assert _FakeWriter.last_instance.written[0]["applied"] is True
    assert report_path.exists()


def test_cli_materialize_unresolved_groups_repeated_new_entities(monkeypatch, tmp_path):
    import book_graph_analyzer.graph.writer as writer_module

    class _FakeWriter:
        last_instance: _FakeWriter | None = None

        def __init__(self) -> None:
            self.materialized = []
            type(self).last_instance = self

        def query_llm_new_entity_suggestions(self, limit: int = 500) -> list[dict]:
            return [
                {
                    "id": "u1",
                    "mention_text": "Glorfindel",
                    "source_book": "The Fellowship of the Ring",
                    "llm_resolution_entity_name": "glorfindel",
                    "llm_resolution_score": 0.6,
                },
                {
                    "id": "u2",
                    "mention_text": "Glorfindel",
                    "source_book": "The Fellowship of the Ring",
                    "llm_resolution_entity_name": "Glorfindel",
                    "llm_resolution_score": 0.62,
                },
                {
                    "id": "u3",
                    "mention_text": "Goldberry",
                    "source_book": "The Fellowship of the Ring",
                    "llm_resolution_entity_name": "goldberry",
                    "llm_resolution_score": 0.61,
                },
            ]

        def materialize_llm_character_suggestions(self, candidates: list[dict]) -> int:
            self.materialized = list(candidates)
            return len(candidates)

        def close(self) -> None:
            return None

    monkeypatch.setattr(writer_module, "GraphWriter", _FakeWriter)

    report_path = tmp_path / "materialized-report.json"
    result = CliRunner().invoke(
        main,
        [
            "lore",
            "materialize-unresolved",
            "--min-support",
            "2",
            "--json-out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Glorfindel" in result.output
    assert _FakeWriter.last_instance is not None
    assert _FakeWriter.last_instance.materialized[0]["canonical_name"] == "Glorfindel"
    assert report_path.exists()
