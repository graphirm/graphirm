from datetime import datetime

from till.counter import next_count, reset_count
from till.export import daily_export


def test_counter() -> None:
    assert next_count(3) == 4
    assert reset_count() == 0


def test_export_date_line() -> None:
    text = daily_export(["mug"], datetime(2026, 1, 2, 15, 4))
    first = text.splitlines()[0]
    assert first == "01/02/2026"
    assert "2026-01-02" not in text
    assert "mug" in text.splitlines()
