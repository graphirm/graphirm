"""Write the two memory-track sequences into instances.lock.json.

Reads the official SWE-Bench-CL curriculum JSON and the CL-Bench codebase
jsonl plus its default schedule. Does not download problem statements into
the lock. Instance ids, order, difficulty and image names only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parent / "instances.lock.json"
CL_JSON = Path("/tmp/swe-bench-cl.json")
CLBENCH_JSONL = Path("/tmp/clbench-codebase.jsonl")

SWE_BENCH_CL_SOURCE = (
    "https://github.com/thomasjoshi/agents-never-forget/blob/main/"
    "data/SWE-Bench-CL-Curriculum.json"
)
CLBENCH_SOURCE = (
    "https://github.com/pgasawa/continual-learning-bench/blob/main/"
    "data/codebase_adaptation/final-dataset.jsonl"
)
CLBENCH_SCHEDULE = (
    "https://github.com/pgasawa/continual-learning-bench/blob/main/"
    "src/tasks/codebase_adaptation/schedules/default.json"
)


def swe_bench_cl(calibration_ids: set[str]) -> dict:
    data = json.loads(CL_JSON.read_text())
    sequences = []
    overlap = []
    for sequence in data["sequences"]:
        tasks = []
        for index, task in enumerate(sequence["tasks"], start=1):
            meta = task["metadata"]
            row = {
                "position": index,
                "instance_id": meta["instance_id"],
                "created_at": meta["created_at"],
                "difficulty": meta["difficulty"],
                "base_commit": meta["base_commit"],
            }
            tasks.append(row)
            if meta["instance_id"] in calibration_ids:
                overlap.append(
                    {
                        "instance_id": meta["instance_id"],
                        "repo": sequence["repo"],
                        "position": index,
                    }
                )
        sequences.append(
            {
                "repo": sequence["repo"],
                "n_tasks": len(tasks),
                "tasks": tasks,
            }
        )
    return {
        "id": "swe-bench-cl",
        "paper": "https://arxiv.org/abs/2507.00014",
        "source": SWE_BENCH_CL_SOURCE,
        "huggingface_note": (
            "The paper names the GitHub JSON only. The croissant file in that "
            "repo names https://huggingface.co/datasets/thomasjoshi/swe-bench-cl."
        ),
        "n_tasks": data["metadata"]["total_tasks"],
        "order": (
            "Official curriculum order: difficulty band, then created_at "
            "within the band. Django is one band, so it is chronological."
        ),
        "instance_ids": "SWE-bench Verified instance ids",
        "harness": (
            "SWE-bench Verified images and the swebench 4.1.0 harness. "
            "The paper's 2024 harness did not support Verified images."
        ),
        "calibration_overlap": overlap,
        "overlap_rule": (
            "Keep overlapping calibration tasks in the sequence. "
            "Calibration informed configuration, not memory behaviour."
        ),
        "sequences": sequences,
    }


def clbench() -> dict:
    rows = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in CLBENCH_JSONL.read_text().splitlines()
        if line.strip()
    }
    # Official blocked order, not file order, though they match.
    order = [
        "jazzband__tablib-534",
        "jazzband__tablib-540",
        "jazzband__tablib-547",
        "jazzband__tablib-579",
        "jazzband__tablib-584",
        "jazzband__tablib-594",
        "jazzband__tablib-595",
        "jazzband__tablib-596",
        "jazzband__tablib-613",
        "jd__tenacity-597",
        "jd__tenacity-603",
        "jd__tenacity-604",
        "jd__tenacity-606",
        "jd__tenacity-609",
        "jd__tenacity-610",
        "jd__tenacity-611",
        "jd__tenacity-614",
        "jd__tenacity-615",
        "jd__tenacity-628",
    ]
    tasks = []
    for index, instance_id in enumerate(order, start=1):
        row = rows[instance_id]
        tasks.append(
            {
                "position": index,
                "instance_id": instance_id,
                "repo": row["repo"],
                "created_at": row["created_at"],
                "base_commit": row["base_commit"],
                "image_name": row["image_name"],
            }
        )
    return {
        "id": "clbench-codebase",
        "paper": "https://arxiv.org/abs/2606.05661",
        "source": CLBENCH_SOURCE,
        "schedule": CLBENCH_SCHEDULE,
        "n_tasks": len(tasks),
        "order": "default schedule: all 9 tablib tasks, then all 10 tenacity tasks",
        "repo_switch": "Memory resets between tablib and tenacity.",
        "harness": (
            "The benchmark's own harness, not SWE-bench Verified images. "
            "Images are pgasawa2/continual-learning-bench:tablib and :tenacity. "
            "Official metrics are reward and gain. max_steps is 40."
        ),
        "tasks": tasks,
    }


def main() -> int:
    if not CL_JSON.is_file() or not CLBENCH_JSONL.is_file():
        print("missing /tmp/swe-bench-cl.json or /tmp/clbench-codebase.jsonl", file=sys.stderr)
        return 1
    lock = json.loads(LOCK.read_text())
    calibration_ids = {row["instance_id"] for row in lock["calibration"]}
    lock["memory_tracks"] = {
        "swe_bench_cl": swe_bench_cl(calibration_ids),
        "clbench_codebase": clbench(),
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    overlap = lock["memory_tracks"]["swe_bench_cl"]["calibration_overlap"]
    print(f"overlap {len(overlap)}")
    print(f"cl tasks {lock['memory_tracks']['swe_bench_cl']['n_tasks']}")
    print(f"clbench tasks {lock['memory_tracks']['clbench_codebase']['n_tasks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
