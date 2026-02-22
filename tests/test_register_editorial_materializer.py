from book_graph_analyzer.ingest.register_editorial_materializer import _extract_character_names, _iter_events


def test_iter_events_accepts_dict_and_list():
    assert len(_iter_events({"events": {"1": {"id": "1"}}})) == 1
    assert len(_iter_events({"events": [{"id": "1"}, {"id": "2"}]})) == 2


def test_extract_character_names_filters_generic_groups():
    names = _extract_character_names("the group", None)
    assert names == []


def test_extract_character_names_splits_compounds():
    names = _extract_character_names("Aragorn, Legolas and Gimli", None)
    assert "Aragorn" in names
    assert "Legolas" in names
    assert "Gimli" in names
