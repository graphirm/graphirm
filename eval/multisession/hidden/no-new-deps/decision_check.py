"""Static check: dependency list stays empty and new modules stay on the standard library."""
import ast
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # pragma: no cover

text = Path("pyproject.toml").read_text()
data = tomllib.loads(text)
deps = data["project"]["dependencies"]
if deps != []:
    print(f"dependencies changed: {deps}")
    sys.exit(1)

stdlib_roots = {
    "abc", "argparse", "ast", "base64", "collections", "contextlib", "csv", "dataclasses",
    "datetime", "decimal", "enum", "functools", "hashlib", "io", "itertools", "json",
    "math", "os", "pathlib", "pickle", "re", "sqlite3", "struct", "sys", "tempfile",
    "typing", "uuid",
}
banned = {"sqlalchemy", "pydantic", "tinydb", "requests", "peewee"}
for rel in ("till/note.py", "till/persist.py"):
    path = Path(rel)
    if not path.exists():
        print(f"missing {rel}")
        sys.exit(1)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules = [node.module.split(".")[0]]
        for name in modules:
            if name in banned or (name not in stdlib_roots and name != "till"):
                print(f"{rel} imports {name}")
                sys.exit(1)
print("ok")
