Add `format_label(name)` to a new module `till/labels.py`. It strips the name and returns it in title case (`str.title()`).

When item creation is added later, a second create of the same sku must raise `DuplicateSku`, defined in `till/catalog.py`, and that exception's `code` attribute must be the integer 409. Do not implement creation in this session.

Do not write this decision into any file, comment, or docstring.
