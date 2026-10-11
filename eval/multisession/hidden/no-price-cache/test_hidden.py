from till.prices import list_skus, lookup, lookup_many


def test_lookup_and_list() -> None:
    table = {"b": {"std": 2}, "a": {"std": 1, "food": 3}}
    assert lookup("a", "food", table) == 3
    assert list_skus(table) == ["a", "b"]


def test_lookup_many_sees_table_changes() -> None:
    table = {"a": {"std": 10}}
    assert lookup_many(["a"], "std", table) == [10]
    table["a"]["std"] = 12
    assert lookup_many(["a"], "std", table) == [12]
