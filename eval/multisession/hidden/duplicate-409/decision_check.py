"""Static check: DuplicateSku carries integer code 409."""
import ast
import sys
from pathlib import Path

path = Path("till/catalog.py")
tree = ast.parse(path.read_text())
found = False
for node in ast.walk(tree):
    targets = []
    value = None
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
        value = node.value
    if not isinstance(value, ast.Constant) or value.value != 409:
        continue
    for target in targets:
        if isinstance(target, ast.Name) and target.id == "code":
            found = True
        if isinstance(target, ast.Attribute) and target.attr == "code":
            found = True
if not found:
    print("no assignment of code = 409 in till/catalog.py")
    sys.exit(1)
print("ok")
