from unittest.mock import MagicMock

from click.testing import CliRunner


from book_graph_analyzer.lore.sociolinguistic_registers import (
    RegisterOutputThresholds,
    SociolinguisticRegister,
    SociolinguisticRegisterClassifier,
    detect_register_drift,
    ground_character_entity_id,
    profile_corpus_registers,
)


def test_classifier_detects_ritual_register():
    classifier = SociolinguisticRegisterClassifier()
    text = "By sacred oath and hallowed rite, we swear before the stars."
    profile = classifier.classify(text)
    assert profile.dominant_register in {
        SociolinguisticRegister.RITUAL.value,
        SociolinguisticRegister.PROPHETIC.value,
    }
    assert profile.confidence > 0


def test_drift_detection_reports_shift():
    classifier = SociolinguisticRegisterClassifier()
    baseline = classifier.classify("We shared bread, ale, and stories by the hearth.")
    current = classifier.classify("By sacred oath, we march beneath the banner.")
    drift = detect_register_drift(baseline, current)
    assert drift.baseline_register != ""
    assert drift.current_register != ""
    assert drift.severity in {"low", "medium", "high"}


def test_model_assist_fallback_is_safe():
    classifier = SociolinguisticRegisterClassifier()
    base = classifier.classify("By sacred oath we march.")

    def bad_assist(_text, _base):
        raise RuntimeError("boom")

    assisted = classifier.classify("By sacred oath we march.", model_assist=bad_assist)
    assert assisted.dominant_register == base.dominant_register


def test_profile_corpus_registers_builds_distribution_and_drifts():
    report = profile_corpus_registers([
        {"entity_id": "char_frodo", "order": 1, "text": "We shared bread and ale by the fire."},
        {"entity_id": "char_frodo", "order": 2, "text": "By sacred oath we march beneath the banner."},
        {"entity_id": "char_sam", "order": 1, "text": "Home and garden are worth the road."},
    ])
    assert report.total_samples == 3
    assert sum(report.dominant_distribution.values()) == 3
    assert "char_frodo" in report.per_entity_latest
    assert isinstance(report.strongest_drifts, list)


def test_ground_character_entity_id_enforces_character_only():
    assert ground_character_entity_id("char_Aragorn") == "char_aragorn"
    assert ground_character_entity_id("Aragorn") == "char_aragorn"
    assert ground_character_entity_id("place_bree") is None
    assert ground_character_entity_id("narration") is None
    assert ground_character_entity_id("char_register") is None


def test_profile_corpus_registers_filters_non_characters_and_sets_quality_gate():
    report = profile_corpus_registers(
        [
            {"entity_id": "char_frodo", "order": 1, "text": "bread and ale"},
            {"entity_id": "Frodo", "order": 2, "text": "sacred oath and banner"},
            {"entity_id": "place_bree", "order": 1, "text": "the road to Bree"},
        ],
        thresholds=RegisterOutputThresholds(min_character_samples=2, min_register_families=1, min_drift_events=1),
    )
    assert report.total_samples == 2
    assert "char_frodo" in report.per_entity_latest
    assert "place_bree" not in report.per_entity_latest
    assert report.quality_gate["passed"] is True


class TestGraphWriterSocioreg:
    def _make_writer(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        writer = GraphWriter(driver=mock_driver)
        return writer, mock_session

    def test_write_profile_and_observation(self):
        writer, session = self._make_writer()
        profile = SociolinguisticRegisterClassifier().classify("The captain gave command.")

        writer.write_register_profile("char_aragorn", profile, source_passage_id="p1")
        writer.write_register_observation("char_aragorn", profile, observed_at="TA 3019", source_passage_id="p1")

        # Resolver path adds lookup queries before writes.
        assert session.run.call_count >= 2

    def test_query_drift_returns_list(self):
        writer, session = self._make_writer()
        session.run.return_value = [
            {
                "observed_at": "TA 3018",
                "dominant_register": "folk",
                "formality_score": 0.2,
                "archaism_rate": 0.01,
                "confidence": 0.6,
            },
            {
                "observed_at": "TA 3019",
                "dominant_register": "martial",
                "formality_score": 0.5,
                "archaism_rate": 0.08,
                "confidence": 0.7,
            },
        ]
        rows = writer.query_register_drift("char_aragorn", min_delta=0.1)
        assert len(rows) == 1
        assert rows[0]["magnitude"] >= 0.1

    def test_query_drift_summary(self):
        writer, session = self._make_writer()
        session.run.return_value = [
            {
                "observed_at": "TA 3018",
                "dominant_register": "folk",
                "formality_score": 0.1,
                "archaism_rate": 0.01,
                "confidence": 0.6,
            },
            {
                "observed_at": "TA 3019",
                "dominant_register": "ritual",
                "formality_score": 0.7,
                "archaism_rate": 0.12,
                "confidence": 0.8,
            },
        ]
        summary = writer.query_register_drift_summary("char_aragorn", min_delta=0.1)
        assert summary["entity_id"] == "char_aragorn"
        assert summary["drift_count"] == 1
        assert summary["strongest"] is not None

    def test_write_profile_rejects_non_character_entity(self):
        writer, _session = self._make_writer()
        profile = SociolinguisticRegisterClassifier().classify("The captain gave command.")

        import pytest

        with pytest.raises(ValueError):
            writer.write_register_profile("place_bree", profile, source_passage_id="p1")


def test_cli_commands_registered_and_run():
    from book_graph_analyzer.cli import main

    runner = CliRunner()

    result_profile = runner.invoke(main, ["lore", "socioreg-profile", "--text", "By oath and rite we stand"])
    assert result_profile.exit_code == 0
    assert "Sociolinguistic Register Profile" in result_profile.output

    result_drift = runner.invoke(
        main,
        [
            "lore",
            "socioreg-drift",
            "--baseline",
            "We shared bread by the fire",
            "--current",
            "By sacred oath we march",
        ],
    )
    assert result_drift.exit_code == 0
    assert "Sociolinguistic Register Drift" in result_drift.output

    result_profile_json = runner.invoke(main, ["lore", "socioreg-profile", "--text", "By oath and rite", "--json"])
    assert result_profile_json.exit_code == 0
    assert '"dominant_register"' in result_profile_json.output

    with runner.isolated_filesystem():
        import json
        with open("samples.json", "w", encoding="utf-8") as f:
            json.dump([
                {"entity_id": "char_frodo", "order": 1, "text": "bread and ale"},
                {"entity_id": "char_frodo", "order": 2, "text": "sacred oath and banner"},
            ], f)
        result_corpus = runner.invoke(main, ["lore", "socioreg-corpus", "--input", "samples.json"])
        assert result_corpus.exit_code == 0
        assert "Corpus Socioreg Profile" in result_corpus.output
