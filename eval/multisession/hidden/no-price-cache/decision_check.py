"""Static check: prices.py must not introduce a cache."""
import ast
import sys
from pathlib import Path

path = Path("till/prices.py")
if not path.exists():
    print("missing till/prices.py")
    sys.exit(1)
tree = ast.parse(path.read_text())
banned_names = {"lru_cache", "cache"}
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id in banned_names:
        print(f"banned name {node.id}")
        sys.exit(1)
    if isinstance(node, ast.Attribute) and node.attr in banned_names:
        print(f"banned attr {node.attr}")
        sys.exit(1)
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "functools":
                print("functools import")
                sys.exit(1)
    if isinstance(node, ast.ImportFrom) and node.module == "functools":
        print("functools import")
        sys.exit(1)
print("ok")
