from book_graph_analyzer.worldbible.models import WorldBible, WorldBibleCategory


def test_markdown_world_bible_loads_asserted_rules_and_ignores_placeholders(tmp_path):
    path = tmp_path / "story_bible.md"
    path.write_text(
        """# Hunt Story Bible

## Premise
An unwritten journey.

## Core Characters
- Aragorn

## World Rules
- News cannot cross Eriador without travel time.
- (add non-negotiable rules)

## Geography
- The Hoarwell must be crossed before the High Pass.

## Open Questions
- Who saw the traveller?
""",
        encoding="utf-8",
    )

    bible = WorldBible.load(path)

    assert bible.name == "Hunt Story Bible"
    assert [rule.description for rule in bible.rules[WorldBibleCategory.THEMES]] == [
        "News cannot cross Eriador without travel time."
    ]
    assert [rule.description for rule in bible.rules[WorldBibleCategory.GEOGRAPHY]] == [
        "The Hoarwell must be crossed before the High Pass."
    ]
    assert sum(len(rules) for rules in bible.rules.values()) == 2
