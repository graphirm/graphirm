#!/usr/bin/env python3
"""Convert GLiNER spans JSONL to synthetic human-annotation format for validate-agreement."""
import json
import sys

for line in sys.stdin:
    if not line.strip():
        continue
    obj = json.loads(line)
    segments = [
        {"type": s["label"], "start": s["start"], "end": s["end"]}
        for s in obj.get("spans", [])
    ]
    out = {
        "session_id": obj["session_id"],
        "turn_index": obj["turn_index"],
        "segments": segments,
    }
    print(json.dumps(out))
