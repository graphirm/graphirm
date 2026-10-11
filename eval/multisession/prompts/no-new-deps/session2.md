Add `clip_words(text, limit)` to `till/note.py`. It clips on a word boundary when possible: if `text[:limit]` ends mid-word, cut back to the previous space. If there is no space, use `clip`.
