from till.catalog import Catalog, Item
from till.drawer import Drawer
from till.legacy.discounts import half_off
from till.money import parse_cents


def test_parse_cents() -> None:
    assert parse_cents("1.25") == 125
    assert parse_cents("-0.05") == -5


def test_catalog_add_get() -> None:
    catalog = Catalog()
    catalog.add(Item("abc", "Mug"))
    assert catalog.get("abc").name == "Mug"
    assert catalog.skus() == ["abc"]


def test_drawer_deposit() -> None:
    drawer = Drawer()
    drawer.deposit(40)
    assert drawer.cents == 40


def test_legacy_half_off() -> None:
    assert half_off(10) == 5
