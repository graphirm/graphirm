"""Static check: till/tax.py does not use floats."""
import ast
import sys
from pathlib import Path

path = Path("till/tax.py")
if not path.exists():
    print("missing till/tax.py")
    sys.exit(1)
tree = ast.parse(path.read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        print(f"float constant {node.value}")
        sys.exit(1)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
        print("float() call")
        sys.exit(1)
print("ok")
