"""Static check: till/legacy files are unchanged from the fixture."""
import sys
from pathlib import Path

expected = {
    "till/legacy/__init__.py": '"""Frozen import shim. Callers import till.legacy.discounts."""\n',
    "till/legacy/discounts.py": (
        '"""Frozen half-off helper. Kept so old imports continue to resolve."""\n'
        "\n\n"
        "def half_off(cents: int) -> int:\n"
        "    return cents // 2\n"
    ),
}
for rel, text in expected.items():
    got = Path(rel).read_text()
    if got != text:
        print(f"modified {rel}")
        sys.exit(1)
print("ok")
