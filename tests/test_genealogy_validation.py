from book_graph_analyzer.worldbible.genealogy_validation import evaluate_genealogy_threshold


def test_genealogy_thresholds_are_book_specific():
    two_towers = evaluate_genealogy_threshold("The Two Towers", observed=3)
    return_king = evaluate_genealogy_threshold("The Return of the King", observed=5)

    assert two_towers.threshold == 4
    assert two_towers.passed is False
    assert return_king.threshold == 4
    assert return_king.passed is True
