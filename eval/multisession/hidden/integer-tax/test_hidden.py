import pytest

from till.drawer import Drawer
from till.names import initials
from till.tax import tax_due


def test_initials_and_withdraw() -> None:
    assert initials("Ada Lovelace") == "AL"
    drawer = Drawer()
    drawer.deposit(10)
    drawer.withdraw(4)
    assert drawer.cents == 6
    with pytest.raises(ValueError):
        drawer.withdraw(100)


def test_tax_half_up_integers() -> None:
    assert tax_due(1, 5000) == 1
    assert tax_due(1, 4999) == 0
    assert tax_due(199, 825) == 16
    assert isinstance(tax_due(199, 825), int)
