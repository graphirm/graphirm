"""Static check: round.py stays on integers. Half-up itself is checked behaviourally."""
import ast
import sys
from pathlib import Path

tree = ast.parse(Path("till/round.py").read_text())
names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
if "round_div" not in names or "service_fee" not in names:
    print(f"missing helpers: {names}")
    sys.exit(1)
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        print("float constant in till/round.py")
        sys.exit(1)
print("ok")
