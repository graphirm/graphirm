"""Static check: the format string MM/DD/YYYY is not required to appear, but ISO format must not be the date line.

The behavioural test covers the format. This check rejects an ISO strftime in till/export.py.
"""
import sys
from pathlib import Path

path = Path("till/export.py")
if not path.exists():
    print("missing till/export.py")
    sys.exit(1)
text = path.read_text()
if "%Y-%m-%d" in text or "isoformat" in text:
    print("export uses an ISO date")
    sys.exit(1)
print("ok")
