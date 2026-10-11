"""Static check: the T- prefix still lives in the sku helper from session 1."""
import sys
from pathlib import Path

text = Path("till/sku.py").read_text()
if "T-" not in text:
    print("till/sku.py lost the T- prefix")
    sys.exit(1)
print("ok")
