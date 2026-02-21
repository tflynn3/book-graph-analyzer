from unittest.mock import MagicMock

from click.testing import CliRunner


from book_graph_analyzer.lore.sociolinguistic_registers import (
    SociolinguisticRegister,
    SociolinguisticRegisterClassifier,
    detect_register_drift,
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

        assert session.run.call_count == 2

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
