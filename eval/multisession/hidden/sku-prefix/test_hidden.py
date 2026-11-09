from till.catalog import Catalog
from till.sku import is_normalized, normalize_sku


def test_normalize_and_import() -> None:
    assert normalize_sku("ab12") == "T-AB12"
    assert normalize_sku("t-ab12") == "T-AB12"
    assert is_normalized("T-AB12")
    catalog = Catalog()
    item = catalog.import_row("ab12", "Mug")
    assert item.sku == "T-AB12"
    assert item.name == "Mug"
    assert catalog.get("T-AB12").name == "Mug"
