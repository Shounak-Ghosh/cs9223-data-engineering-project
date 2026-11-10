import pytest

from jev_milestone_2.tools.agreement import analyze, cohen_kappa, fleiss_kappa


def test_class_confusion_tables_and_raw_agreement():
    report = analyze({"A": dict(zip("abcd", "0011")), "B": dict(zip("abcd", "0101"))},
                     {"b": "0", "c": "1"})
    pair = report["pairwise"]["A-B"]
    assert pair["confusion_table"] == {"0": {"0": 1, "1": 1}, "1": {"0": 1, "1": 1}}
    assert pair["row_annotator"] == "A"
    assert pair["column_annotator"] == "B"
    assert pair["agreement"] == report["raw_unanimous_agreement"] == .5
    assert pair["cohen_kappa"] == 0


def test_multi_annotator_tables_and_unanimity():
    report = analyze({"A": {"x": "1", "y": "0"}, "B": {"x": "1", "y": "1"},
                      "C": {"x": "1", "y": "0"}}, {"y": "0"})
    assert len(report["pairwise"]) == 3
    assert report["raw_unanimous_agreement"] == .5
    assert report["fleiss_kappa"] is not None
    assert report["pairwise"]["A-C"]["confusion_table"] == {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}}


def test_confusion_table_orientation_with_asymmetric_disagreements():
    report = analyze({"A": dict(zip("abcd", "0001")), "B": dict(zip("abcd", "0111"))},
                     {"b": "0", "c": "1"})
    assert report["pairwise"]["A-B"]["confusion_table"] == {
        "0": {"0": 1, "1": 2}, "1": {"0": 0, "1": 1},
    }


@pytest.mark.parametrize("sheets", [{}, {"A": {}}, {"A": {}, "B": {}},
                                     {"A": {"x": "maybe"}, "B": {"x": "0"}}])
def test_empty_and_invalid_sheets_fail_cleanly(sheets):
    with pytest.raises(ValueError):
        analyze(sheets, {})


def test_empty_mismatched_or_invalid_kappa_columns_fail_cleanly():
    for a, b in (([], []), (["0"], []), (["bad"], ["1"])):
        with pytest.raises(ValueError):
            cohen_kappa(a, b)
    for columns in ([], [[]], [[], []], [["1"], []], [["bad"], ["0"]]):
        with pytest.raises(ValueError):
            fleiss_kappa(columns)
