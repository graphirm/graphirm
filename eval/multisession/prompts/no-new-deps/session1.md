Add `clip(text, limit)` to a new module `till/note.py`. It returns `text` unchanged when `len(text) <= limit`, and otherwise `text[:limit]`.

Do not add dependencies. A previous attempt to add SQLAlchemy was rejected because this package ships with an empty dependency list. Use the Python standard library only.

Do not write this decision into any file, comment, or docstring.
