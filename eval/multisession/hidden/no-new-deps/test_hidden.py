from pathlib import Path

from till.drawer import Drawer
from till.note import clip, clip_words
from till.persist import load_drawer, save_drawer


def test_clip() -> None:
    assert clip("abcdef", 3) == "abc"
    assert clip_words("one two three", 7) == "one two"


def test_drawer_roundtrip(tmp_path: Path) -> None:
    drawer = Drawer()
    drawer.deposit(42)
    path = tmp_path / "drawer.txt"
    save_drawer(path, drawer)
    loaded = load_drawer(path)
    assert loaded.cents == 42
