"""SKU catalog. Items are stored by the sku string they are given."""


class Item:
    def __init__(self, sku: str, name: str) -> None:
        self.sku = sku
        self.name = name


class Catalog:
    def __init__(self) -> None:
        self._items: dict[str, Item] = {}

    def add(self, item: Item) -> None:
        self._items[item.sku] = item

    def get(self, sku: str) -> Item | None:
        return self._items.get(sku)

    def skus(self) -> list[str]:
        return sorted(self._items)
