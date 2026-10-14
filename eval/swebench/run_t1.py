#!/usr/bin/env python3
"""Fresh T1 rerun on the 210 valid headline tasks.

Arms are interleaved per task. Outputs stay under /home/krs/graphirm-runs/t1.
Each arm records git -C /testbed rev-parse HEAD. Grading uses the same
image tag. A commit mismatch is flagged and the run continues.

Stdout progress lines omit per-arm resolve flags.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))

from durable import append_ledger, backup_record, load_done, push_backup, runs_dir
from run_arm import load_dotenv, render_task, strip_test_hunks

REPO = Path(__file__).resolve().parents[2]
ARROW_DIR = Path(
    "/home/krs/.cache/huggingface/datasets/nebius___swe-rebench-leaderboard"
    "/default/0.0.0/34d5a58864acf91613740a09ec5d205228dcfa39"
)
LOCK = REPO / "eval/swebench/instances.lock.json"
RUNNER = REPO / "eval/swebench/adapters/graphirm_runner.sh"
BINARY = REPO / "target/x86_64-unknown-linux-musl/release/graphirm"
MINI_CONFIG = REPO / "eval/swebench/adapters/mini_config.yaml"
SWEBENCH_YAML = (
    "/home/krs/anaconda3/envs/py312/lib/python3.12/site-packages"
    "/minisweagent/config/benchmarks/swebench.yaml"
)
EXCLUDE = {
    "sigma67__ytmusicapi-909_interface",
    "huggingface__trl-5135",
    "vyperlang__vyper-4801",
    "keras-team__keras-22316",
    "tox-dev__tox-3931",
}
ARMS = ["M", "G", "G-md", "G-b16", "G-b16-recency"]
TRACK = "t1"
SPEND_CAP = 35.0
DISK_LIMIT = 78.0
ARM_WORKERS = 4

load_dotenv(REPO / ".env")
ledger_lock = threading.Lock()


def used_pct() -> float:
    usage = shutil.disk_usage("/")
    return 100.0 * usage.used / usage.total


def wait_disk() -> None:
    import time

    while used_pct() >= DISK_LIMIT:
        print(f"DISK {used_pct():.1f} waiting", flush=True)
        time.sleep(30)


def headline_ids() -> list[str]:
    lock = json.loads(LOCK.read_text())
    ids = [
        row["instance_id"]
        for row in lock["headline"]
        if row["instance_id"] not in EXCLUDE
    ]
    if len(ids) != 210:
        raise SystemExit(f"expected 210 headline ids, found {len(ids)}")
    return ids


def load_instances(ids: list[str]) -> list[dict]:
    wanted = set(ids)
    found: dict[str, dict] = {}
    for path in sorted(ARROW_DIR.glob("swe-rebench-leaderboard-2026_0[1-5].arrow")):
        with pa.memory_map(str(path), "r") as source:
            for row in pa.ipc.open_stream(source).read_all().to_pylist():
                if row["instance_id"] in wanted:
                    found[row["instance_id"]] = row
    missing = [iid for iid in ids if iid not in found]
    if missing:
        raise SystemExit(f"missing {len(missing)} instances, first {missing[:5]}")
    return [found[iid] for iid in ids]


def image_head(client: docker.DockerClient, image: str) -> str | None:
    try:
        raw = client.containers.run(
            image,
            entrypoint=["git"],
            command=["-C", "/testbed", "rev-parse", "HEAD"],
            remove=True,
            user="root",
        )
    except Exception:
        return None
    text = raw.decode().strip() if isinstance(raw, bytes) else str(raw).strip()
    return text or None


def arm_env(arm: str) -> dict[str, str]:
    env = {}
    if arm == "G-md":
        env["GRAPHIRM_REPLY"] = "md"
    if arm in {"G-b16", "G-b16-recency"}:
        env["GRAPHIRM_CONTEXT_MAX_TOKENS"] = "16000"
    if arm == "G-b16-recency":
        env["GRAPHIRM_SCORING"] = "recency"
    return env


def run_mini(instance: dict, out: Path) -> dict:
    from minisweagent.agents import get_agent
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment
    from minisweagent.utils.serialize import recursive_merge

    base = get_config_from_spec(SWEBENCH_YAML)
    mini = get_config_from_spec(str(MINI_CONFIG))
    config = recursive_merge(base, mini)
    config.setdefault("environment", {})
    config["environment"]["pull_timeout"] = 600
    config["environment"]["container_timeout"] = "35m"
    traj = out / "M.traj.json"
    agent_cfg = {
        "system_template": config["agent"]["system_template"],
        "instance_template": config["agent"]["instance_template"],
        "step_limit": 100,
        "cost_limit": 0.5,
        "wall_time_limit_seconds": 1800,
        "output_path": traj,
    }
    env = get_sb_environment(config, instance)
    record = {
        "arm": "M",
        "image": instance["image_name"],
        "start_commit": None,
        "cost_usd": 0.0,
        "error": None,
    }
    try:
        probed = env.execute({"command": "git -C /testbed rev-parse HEAD"}, timeout=60)
        record["start_commit"] = (probed.get("output") or "").strip() or None
        agent = get_agent(get_model(config=config.get("model", {})), env, agent_cfg, default_type="default")
        info = agent.run(instance["problem_statement"])
        submission = info.get("submission") or ""
        patch = strip_test_hunks(submission, instance.get("test_patch") or "")
        (out / "M.patch").write_text(patch)
        record["cost_usd"] = float(getattr(agent, "cost", 0.0) or 0.0)
        summary = {
            "arm": "M",
            "start_commit": record["start_commit"],
            "image": instance["image_name"],
            "cost_usd": record["cost_usd"],
            "exit_status": info.get("exit_status"),
        }
        (out / "M.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
        if not (out / "M.patch").exists():
            (out / "M.patch").write_text("")
    finally:
        cleanup = getattr(env, "cleanup", None)
        if cleanup:
            cleanup()
    return record


def run_graphirm(instance: dict, arm: str, out: Path) -> dict:
    iid = instance["instance_id"]
    name = f"t1-{arm}-{iid}".lower().replace("_", "-")[:60]
    client = docker.from_env(timeout=600)
    record = {
        "arm": arm,
        "image": instance["image_name"],
        "start_commit": None,
        "cost_usd": 0.0,
        "error": None,
        "http_429": 0,
    }
    container = None
    try:
        try:
            client.containers.get(name).remove(force=True)
        except docker.errors.NotFound:
            pass
        container = client.containers.run(
            instance["image_name"],
            command=["sleep", "2100"],
            detach=True,
            name=name,
            working_dir="/testbed",
            user="root",
        )
        task = out / f"{arm}.issue.md"
        task.write_text(render_task(instance["problem_statement"]))
        test_patch = out / "test.patch"
        test_patch.write_text(instance.get("test_patch") or "")
        graph_out = out / f"{arm}.graph.db"
        summary_out = out / f"{arm}.summary.json"
        retry_log = out / f"{arm}.retries.log"
        env = os.environ.copy()
        env.update(arm_env(arm))
        env.update(
            {
                "CONTAINER": container.id,
                "TASK_FILE": str(task),
                "WORKDIR": "/testbed",
                "GRAPH_OUT": str(graph_out),
                "SUMMARY_OUT": str(summary_out),
                "TEST_PATCH_FILE": str(test_patch),
                "GRAPHIRM_BIN": str(BINARY),
                "GRAPHIRM_IMAGE": instance["image_name"],
                "RETRY_LOG_OUT": str(retry_log),
                "GRAPHIRM_MAX_TURNS": "100",
                "GRAPHIRM_MAX_COST": "0.50",
            }
        )
        proc = subprocess.run(
            ["sh", str(RUNNER)],
            cwd=REPO,
            env=env,
            timeout=1900,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            record["error"] = (proc.stderr or proc.stdout or f"exit {proc.returncode}")[:400]
        if summary_out.is_file():
            summary = json.loads(summary_out.read_text())
            record["start_commit"] = summary.get("start_commit")
            record["cost_usd"] = float(summary.get("cost_usd") or 0.0)
        elif (Path(str(summary_out) + ".start_commit")).is_file():
            record["start_commit"] = Path(str(summary_out) + ".start_commit").read_text().strip()
        if retry_log.is_file():
            record["http_429"] = retry_log.read_text().count("429")
        patch_path = Path(str(graph_out) + ".diff")
        if not patch_path.is_file():
            patch_path.write_text("")
        (out / f"{arm}.patch").write_text(patch_path.read_text())
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
        if not (out / f"{arm}.patch").exists():
            (out / f"{arm}.patch").write_text("")
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
    return record


def grade_arms(instance: dict, out: Path, client) -> dict[str, bool | None]:
    from grade_fork import grade_patch, remove_task_image

    graded: dict[str, bool | None] = {}
    for arm in ARMS:
        patch = (out / f"{arm}.patch").read_text() if (out / f"{arm}.patch").is_file() else ""
        if not patch.strip():
            graded[arm] = False
            continue
        try:
            graded[arm] = grade_patch(
                instance,
                patch,
                client,
                1800,
                f"t1{arm.lower().replace('-', '')}",
                arm,
            )
        except Exception:
            graded[arm] = None
    try:
        remove_task_image(instance, client)
    except Exception as exc:
        print(f"RM_ERR {instance['instance_id']} {type(exc).__name__}: {exc}", flush=True)
    return graded


def spent(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += float(record.get("cost_usd") or 0.0)
    return total


def main() -> None:
    if not BINARY.is_file():
        raise SystemExit(f"missing binary {BINARY}")
    root = runs_dir(TRACK)
    os.chdir(root)
    arm_ledger = root / "arms.jsonl"
    task_ledger = root / "tasks.jsonl"
    done_tasks = load_done(task_ledger)
    done_arms = set()
    if arm_ledger.exists():
        for line in arm_ledger.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("complete") is True:
                done_arms.add(record["instance_id"])
    instances = load_instances(headline_ids())
    pending = [row for row in instances if row["instance_id"] not in done_tasks]
    print(
        f"START total=210 done={len(done_tasks)} pending={len(pending)} "
        f"spend={spent(arm_ledger):.2f} disk={used_pct():.1f}",
        flush=True,
    )
    finished = len(done_tasks)
    total_errors = 0
    total_429 = 0
    total_mismatch = 0
    for instance in pending:
        if spent(arm_ledger) >= SPEND_CAP:
            print(f"STOP spend={spent(arm_ledger):.2f} cap={SPEND_CAP}", flush=True)
            break
        iid = instance["instance_id"]
        wait_disk()
        out = root / iid
        out.mkdir(parents=True, exist_ok=True)
        client = docker.from_env(timeout=600)
        try:
            client.images.pull(instance["image_name"])
        except Exception as exc:
            print(f"PULL_ERR {iid} {type(exc).__name__}", flush=True)
            continue
        head = image_head(client, instance["image_name"])
        mismatches = []
        errors = 0
        http_429 = 0

        def one_arm(arm: str) -> dict:
            key = f"{iid}::{arm}"
            if key in done_arms:
                return {"instance_id": key, "complete": True, "skipped": True}
            if arm == "M":
                result = run_mini(instance, out)
            else:
                result = run_graphirm(instance, arm, out)
            result["instance_id"] = key
            result["task"] = iid
            result["image_head"] = head
            result["commit_mismatch"] = bool(
                result.get("start_commit") and head and result["start_commit"] != head
            )
            result["complete"] = True
            with ledger_lock:
                append_ledger(arm_ledger, result)
                backup_record(
                    TRACK,
                    result,
                    [p for p in (out / f"{arm}.patch", out / f"{arm}.summary.json") if p.is_file()],
                )
            return result

        with ThreadPoolExecutor(max_workers=ARM_WORKERS) as pool:
            futures = [pool.submit(one_arm, arm) for arm in ARMS]
            arm_results = [future.result() for future in as_completed(futures)]
        for result in arm_results:
            if result.get("skipped"):
                continue
            if result.get("error"):
                errors += 1
                total_errors += 1
            http_429 += int(result.get("http_429") or 0)
            total_429 += int(result.get("http_429") or 0)
            if result.get("commit_mismatch"):
                mismatches.append(result["arm"])
                total_mismatch += 1
        try:
            graded = grade_arms(instance, out, client)
        except Exception:
            graded = {arm: None for arm in ARMS}
            print(f"GRADE_ERR {iid}", flush=True)
            traceback.print_exc()
        task_record = {
            "instance_id": iid,
            "complete": all(graded.get(arm) is not None for arm in ARMS),
            "image": instance["image_name"],
            "image_head": head,
            "commit_mismatch": mismatches,
            "errors": errors,
            "http_429": http_429,
            "grades_present": sum(value is not None for value in graded.values()),
        }
        # Resolved flags stay in the grade file. Progress lines do not print them.
        (out / "grades.json").write_text(json.dumps(graded, indent=2) + "\n")
        append_ledger(task_ledger, task_record)
        backup_record(TRACK, task_record, [out / "grades.json"])
        finished += 1
        if finished % 10 == 0:
            try:
                push_backup()
                print(f"PUSH tasks={finished}", flush=True)
            except Exception as exc:
                print(f"PUSH_ERR {type(exc).__name__}", flush=True)
        if finished % 25 == 0:
            print(
                f"PROGRESS tasks={finished} spend={spent(arm_ledger):.2f} "
                f"disk={used_pct():.1f} errors={total_errors} http_429={total_429} "
                f"commit_mismatch={total_mismatch}",
                flush=True,
            )
    print(f"HOLD tasks={finished} spend={spent(arm_ledger):.2f}", flush=True)


if __name__ == "__main__":
    main()
