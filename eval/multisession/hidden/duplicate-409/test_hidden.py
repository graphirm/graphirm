import pytest

from till.catalog import Catalog, DuplicateSku
from till.labels import format_label, normalize_name


def test_labels() -> None:
    assert format_label("  mug ") == "Mug"
    assert normalize_name(" a   b ") == "a b"


def test_duplicate_create_is_409() -> None:
    catalog = Catalog()
    first = catalog.create("mug", "Mug")
    assert first.sku == "mug"
    with pytest.raises(DuplicateSku) as raised:
        catalog.create("mug", "Cup")
    assert raised.value.code == 409
    assert catalog.get("mug").name == "Mug"
