Add `initials(name)` to a new module `till/names.py`. It returns the uppercase first letter of each whitespace-separated word, concatenated (`"Ada Lovelace"` becomes `"AL"`).

Money stays in integer minor units. Do not use float for money. Tax is not implemented in this session. When it is added, `tax_due(price, rate_bps)` must return the integer `(price * rate_bps + 5000) // 10000`, where both arguments are integers and `rate_bps` is basis points.

Do not write this decision into any file, comment, or docstring.
