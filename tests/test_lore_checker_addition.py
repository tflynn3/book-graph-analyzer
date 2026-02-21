"""Tests for LoreChecker.check_addition() — permissive RAG path."""

import pytest
from unittest.mock import MagicMock, patch


class TestLoreCheckerAddition:
    """
    check_addition() is the permissive RAG path.
    UNKNOWN + no contradictions = PLAUSIBLE (creative addition accepted).
    INVALID = still INVALID (hard canon violation).
    """

    def _make_checker(self):
        from book_graph_analyzer.lore.checker import LoreChecker
        checker = LoreChecker(use_llm=False)
        return checker

    def test_unknown_with_no_contradictions_becomes_plausible(self):
        from book_graph_analyzer.lore.checker import LoreChecker, ValidationStatus

        checker = self._make_checker()

        # An entirely invented entity — not in any loaded canon
        result = checker.check_addition("Tuor found an ancient Elven dagger named Aeglos-minor")

        assert result.status == ValidationStatus.PLAUSIBLE
        assert "creative addition accepted" in result.explanation.lower()

    def test_plausible_confidence_is_at_least_0_6(self):
        from book_graph_analyzer.lore.checker import LoreChecker, ValidationStatus

        checker = self._make_checker()
        result = checker.check_addition("A ruined tower called Barad-wath stood in the mountains")

        assert result.status == ValidationStatus.PLAUSIBLE
        assert result.confidence >= 0.6

    def test_invalid_claim_stays_invalid(self):
        """A hard canon violation must not be promoted to PLAUSIBLE."""
        from book_graph_analyzer.lore.checker import LoreChecker, ValidationStatus

        checker = self._make_checker()

        # Inject a contradicting evidence manually by mocking check()
        with patch.object(checker, 'check') as mock_check:
            from book_graph_analyzer.lore.checker import ValidationResult, Evidence
            from book_graph_analyzer.lore.parser import ParsedClaim, ClaimType

            fake_claim = MagicMock(spec=ParsedClaim)
            fake_claim.original_text = "Tuor met Elrond in Nevrast"

            mock_result = ValidationResult(
                claim=fake_claim,
                status=ValidationStatus.INVALID,
                confidence=0.9,
                contradicting=[
                    Evidence(
                        text="Elrond was not yet born during Tuor's time in Nevrast",
                        source="Timeline",
                        supports=False,
                    )
                ],
                explanation="Elrond is not yet born in this era.",
            )
            mock_check.return_value = mock_result

            result = checker.check_addition("Tuor met Elrond in Nevrast")

        # check() returned INVALID — check_addition should NOT override that
        assert result.status == ValidationStatus.INVALID

    def test_plausible_icon_is_plus(self):
        """PLAUSIBLE status should show [+] not [~] in summary."""
        from book_graph_analyzer.lore.checker import LoreChecker, ValidationStatus

        checker = self._make_checker()
        result = checker.check_addition("A small village of Men named Aldburg stood near the river")

        assert result.status == ValidationStatus.PLAUSIBLE
        summary = result.summary()
        assert "[+]" in summary, f"Expected [+] in summary, got: {summary[:50]}"
        assert "[~]" not in summary

    def test_partial_with_no_contradictions_becomes_plausible(self):
        """PARTIAL + no contradictions should also be promoted."""
        from book_graph_analyzer.lore.checker import LoreChecker, ValidationStatus, ValidationResult, Evidence
        from book_graph_analyzer.lore.parser import ParsedClaim

        checker = self._make_checker()

        with patch.object(checker, 'check') as mock_check:
            fake_claim = MagicMock(spec=ParsedClaim)
            fake_claim.original_text = "Elves can craft named swords"

            mock_result = ValidationResult(
                claim=fake_claim,
                status=ValidationStatus.PARTIAL,
                confidence=0.5,
                supporting=[
                    Evidence(text="Elves are skilled smiths", source="World Bible", supports=True)
                ],
                contradicting=[],  # No contradictions
                explanation="Some aspects valid",
            )
            mock_check.return_value = mock_result

            result = checker.check_addition("Elves can craft named swords")

        assert result.status == ValidationStatus.PLAUSIBLE
