Add `lookup(sku, tax_class, table)` to a new module `till/prices.py`.

`table` maps a sku to a dict of tax class to integer cents. `lookup` returns `table[sku][tax_class]`. Let a missing sku or tax class raise `KeyError`.

We already tried a module-level cache keyed only by sku. It served the previous tax class after the caller switched class. Do not cache lookup results, in this function or in a helper it calls.

Do not write this decision into any file, comment, or docstring.
