Add `normalize_sku(raw)` to a new module `till/sku.py`. It strips `raw`, uppercases it, and prefixes `T-` when that prefix is missing. `ab12` becomes `T-AB12`. `t-ab12` becomes `T-AB12`.

Add `add_raw(self, raw_sku, name)` to `Catalog`. It stores an item whose sku is `normalize_sku(raw_sku)`.
