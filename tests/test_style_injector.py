from unittest.mock import MagicMock

from book_graph_analyzer.generate.style_injector import StyleInjector, StyleConstraints


def _mock_driver(rows):
    driver = MagicMock()
    session = MagicMock()
    session.run.return_value = rows
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    return driver


def test_classify_scene_type_examples():
    injector = StyleInjector(driver=None)
    assert injector.classify_scene_type("battle outside Angband") == "battle"
    assert injector.classify_scene_type("Manwe speaks to the Valar") == "council"


def test_build_style_block_uses_constraints():
    injector = StyleInjector(driver=None)
    constraints = StyleConstraints(
        scene_type="battle",
        sample_size=8,
        avg_sentence_length_words=14.0,
        dialogue_ratio=0.12,
        passive_ratio=0.09,
        archaic_word_density=0.05,
        characteristic_vocab=["smote", "hewed"],
    )
    block = injector.build_style_block(constraints)
    assert "STYLE CONSTRAINTS (battle scene" in block
    assert "Target sentence length" in block


def test_get_style_constraints_returns_none_when_sparse():
    rows = [
        {
            "avg_sentence_length": 12.0,
            "dialogue_density": 0.1,
            "passive_ratio": 0.05,
            "archaic_word_count": 1,
            "sentence_count": 2,
            "text": "he smote the foe",
        }
    ]
    injector = StyleInjector(driver=_mock_driver(rows), min_samples=5)
    assert injector.get_style_constraints("battle") is None


def test_get_style_constraints_aggregates_passage_data():
    rows = [
        {
            "avg_sentence_length": 12.0,
            "dialogue_density": 0.1,
            "passive_ratio": 0.05,
            "archaic_word_count": 2,
            "sentence_count": 2,
            "text": "He smote the armored captain in darkness",
        },
        {
            "avg_sentence_length": 16.0,
            "dialogue_density": 0.05,
            "passive_ratio": 0.1,
            "archaic_word_count": 1,
            "sentence_count": 2,
            "text": "The warhost gathered with iron banners",
        },
        {
            "avg_sentence_length": 14.0,
            "dialogue_density": 0.08,
            "passive_ratio": 0.03,
            "archaic_word_count": 1,
            "sentence_count": 2,
            "text": "They hewed through shadow and thunder",
        },
        {
            "avg_sentence_length": 15.0,
            "dialogue_density": 0.06,
            "passive_ratio": 0.04,
            "archaic_word_count": 1,
            "sentence_count": 2,
            "text": "Shields splintered as captains rallied",
        },
        {
            "avg_sentence_length": 13.0,
            "dialogue_density": 0.07,
            "passive_ratio": 0.06,
            "archaic_word_count": 1,
            "sentence_count": 2,
            "text": "The press of battle rolled onward",
        },
    ]
    injector = StyleInjector(driver=_mock_driver(rows), min_samples=5)
    constraints = injector.get_style_constraints("battle")

    assert constraints is not None
    assert constraints.scene_type == "battle"
    assert constraints.sample_size == 5
    assert 13.0 <= constraints.avg_sentence_length_words <= 15.0
    assert constraints.dialogue_ratio > 0
