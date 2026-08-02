"""Tests for temporal validity on graph relationships (issue #3)."""

from book_graph_analyzer.graph.temporal import (
    TemporalValidity,
    canonicalize_era,
    era_to_order,
    era_before_or_equal,
    era_after_or_equal,
    point_in_time_cypher_where,
)


# ---------------------------------------------------------------------------
# Era ordering
# ---------------------------------------------------------------------------

class TestEraOrdering:
    def test_before_time_is_earliest(self):
        assert era_to_order("Before Time") < era_to_order("First Age")

    def test_third_age_after_second_age(self):
        assert era_to_order("Third Age") > era_to_order("Second Age")

    def test_fourth_age_is_latest_known(self):
        assert era_to_order("Fourth Age") > era_to_order("Third Age")

    def test_unknown_era_sorts_last(self):
        assert era_to_order("Unknown") > era_to_order("Fourth Age")
        assert era_to_order(None) > era_to_order("Fourth Age")

    def test_full_ordering(self):
        eras = ["Before Time", "Years of the Lamps", "Years of the Trees",
                "First Age", "Second Age", "Third Age", "Fourth Age"]
        orders = [era_to_order(e) for e in eras]
        assert orders == sorted(orders), "Eras should be in strictly ascending order"

    def test_era_before_or_equal(self):
        assert era_before_or_equal("First Age", "Third Age")
        assert era_before_or_equal("Third Age", "Third Age")
        assert not era_before_or_equal("Third Age", "Second Age")

    def test_era_after_or_equal(self):
        assert era_after_or_equal("Third Age", "First Age")
        assert era_after_or_equal("Second Age", "Second Age")
        assert not era_after_or_equal("First Age", "Third Age")


class TestEraAliases:
    def test_sa_canonicalises_to_second_age(self):
        assert canonicalize_era("SA") == "Second Age"

    def test_ta_canonicalises_to_third_age(self):
        assert canonicalize_era("TA") == "Third Age"

    def test_fa_canonicalises_to_first_age(self):
        assert canonicalize_era("FA") == "First Age"

    def test_unknown_alias_returns_itself(self):
        assert canonicalize_era("Madeup Age") == "Madeup Age"

    def test_none_returns_none(self):
        assert canonicalize_era(None) is None


# ---------------------------------------------------------------------------
# TemporalValidity
# ---------------------------------------------------------------------------

class TestTemporalValidity:

    def test_always_valid_for_any_era(self):
        tv = TemporalValidity.always()
        assert tv.is_valid_at("First Age")
        assert tv.is_valid_at("Third Age")
        assert tv.is_valid_at("Second Age", year=1600)

    def test_era_start_filters_earlier_eras(self):
        tv = TemporalValidity(era_start="Third Age")
        assert not tv.is_valid_at("Second Age")
        assert not tv.is_valid_at("First Age")
        assert tv.is_valid_at("Third Age")
        assert tv.is_valid_at("Fourth Age")

    def test_era_end_filters_later_eras(self):
        tv = TemporalValidity(era_end="Second Age")
        assert tv.is_valid_at("First Age")
        assert tv.is_valid_at("Second Age")
        assert not tv.is_valid_at("Third Age")

    def test_era_range(self):
        tv = TemporalValidity(era_start="Second Age", era_end="Third Age")
        assert not tv.is_valid_at("First Age")
        assert tv.is_valid_at("Second Age")
        assert tv.is_valid_at("Third Age")
        assert not tv.is_valid_at("Fourth Age")

    def test_year_start_within_same_era(self):
        tv = TemporalValidity(era_start="Third Age", year_start=3001)
        assert tv.is_valid_at("Third Age", year=3001)
        assert tv.is_valid_at("Third Age", year=3018)
        assert not tv.is_valid_at("Third Age", year=2999)

    def test_year_end_within_same_era(self):
        tv = TemporalValidity(era_end="Third Age", year_end=3021)
        assert tv.is_valid_at("Third Age", year=3000)
        assert tv.is_valid_at("Third Age", year=3021)
        assert not tv.is_valid_at("Third Age", year=3022)

    def test_year_filter_only_applies_in_matching_era(self):
        tv = TemporalValidity(era_start="Third Age", year_start=3001)
        # Fourth Age: year filtering doesn't apply (different era)
        assert tv.is_valid_at("Fourth Age", year=1)

    def test_to_dict_omits_none_values(self):
        tv = TemporalValidity(era_start="Third Age")
        d = tv.to_dict()
        assert "era_start" in d
        assert "era_end" not in d
        assert "year_start" not in d

    def test_to_dict_includes_all_set_values(self):
        tv = TemporalValidity(
            era_start="Second Age", era_end="Third Age",
            year_start=1600, year_end=3021,
            source_passage_id="p_001",
        )
        d = tv.to_dict()
        assert d["era_start"] == "Second Age"
        assert d["era_end"] == "Third Age"
        assert d["year_start"] == 1600
        assert d["year_end"] == 3021
        assert d["source_passage_id"] == "p_001"

    def test_from_dict_roundtrip(self):
        tv = TemporalValidity(
            era_start="Third Age", era_end="Fourth Age",
            year_start=3018, confidence=0.9,
        )
        d = tv.to_dict()
        tv2 = TemporalValidity.from_dict(d)
        assert tv2.era_start == "Third Age"
        assert tv2.era_end == "Fourth Age"
        assert tv2.year_start == 3018
        assert abs(tv2.confidence - 0.9) < 0.01

    def test_from_era_factory(self):
        tv = TemporalValidity.from_era("Second Age", year=1600, passage_id="p_001")
        assert tv.era_start == "Second Age"
        assert tv.year_start == 1600
        assert tv.era_end is None
        assert tv.source_passage_id == "p_001"

    def test_alias_canonicalised_in_to_dict(self):
        tv = TemporalValidity(era_start="TA", era_end="FA4")
        d = tv.to_dict()
        # Aliases should be resolved to canonical names in the dict
        assert d["era_start"] == "Third Age"
        assert d["era_end"] == "Fourth Age"


# ---------------------------------------------------------------------------
# Cypher WHERE fragment
# ---------------------------------------------------------------------------

class TestPointInTimeCypher:
    def test_returns_non_empty_string(self):
        fragment = point_in_time_cypher_where()
        assert isinstance(fragment, str)
        assert len(fragment) > 50

    def test_contains_era_references(self):
        fragment = point_in_time_cypher_where("r", "$era", "$year")
        assert "era_start" in fragment
        assert "era_end" in fragment
        assert "$era" in fragment

    def test_custom_aliases(self):
        fragment = point_in_time_cypher_where("rel", "$myEra", "$myYear")
        assert "rel.era_start" in fragment
        assert "$myEra" in fragment
        assert "$myYear" in fragment
