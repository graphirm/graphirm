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

from reconstruct_check import check as reconstruct_check
from reconstruct_check import sampled as reconstruction_sampled
from provenance.check import summarize as provenance_summary
from config_guard import (
    GRAPHIRM_MODEL,
    MINI_MODEL,
    PROVIDER,
    REASONING,
    arm_problem,
    graph_size_report,
    provider_error,
    toml_problem,
)
from durable import (
    append_ledger,
    backup_record,
    pending_backup_count,
    push_backup,
    remember_failed_backup,
    retry_failed_backups,
    runs_dir,
)
from run_arm import load_dotenv, render_task, strip_test_hunks
from failure_class import CONFIG, INFRASTRUCTURE, REQUEST, classify_arm

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
SPEND_CAP = 45.0
DISK_LIMIT = 78.0
ARM_WORKERS = 4

load_dotenv(REPO / ".env")
ledger_lock = threading.Lock()


def used_pct() -> float:
    usage = shutil.disk_usage("/")
    return 100.0 * usage.used / usage.total


_disk_alerted = False


def wait_disk() -> None:
    import time

    from disk_guard import clear_stamp_if_below, reclaim

    global _disk_alerted
    reclaimed = False
    while used_pct() >= DISK_LIMIT:
        pct = used_pct()
        if not reclaimed:
            print(f"DISK {pct:.1f} reclaim", flush=True)
            print(reclaim(), flush=True)
            reclaimed = True
            if used_pct() < DISK_LIMIT:
                break
            pct = used_pct()
        print(f"DISK {pct:.1f} waiting", flush=True)
        if pct >= 85.0 and not _disk_alerted:
            print(f"DISK_HIGH {pct:.1f}", flush=True)
            _disk_alerted = True
        time.sleep(30)
    clear_stamp_if_below(DISK_LIMIT, used_pct())


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
        record["diag"] = f"{type(exc).__name__}: {exc}"[:4000]
        record["error"] = record["diag"][:400]
        if not (out / "M.patch").exists():
            (out / "M.patch").write_text("")
    finally:
        cleanup = getattr(env, "cleanup", None)
        if cleanup:
            cleanup()
    return record


def run_graphirm(instance: dict, arm: str, out: Path, trace: Path | None = None) -> dict:
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
        if trace is not None:
            env["PACK_TRACE_OUT"] = str(trace)
        proc = subprocess.run(
            ["sh", str(RUNNER)],
            cwd=REPO,
            env=env,
            timeout=1900,
            capture_output=True,
            text=True,
        )
        diag = proc.stderr or proc.stdout or ""
        record["diag"] = diag[-4000:]
        if proc.returncode != 0:
            record["error"] = (diag or f"exit {proc.returncode}")[-400:]
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
        record["diag"] = f"{type(exc).__name__}: {exc}"[:4000]
        record["error"] = record["diag"][:400]
        if not (out / f"{arm}.patch").exists():
            (out / f"{arm}.patch").write_text("")
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
    return record


def grade_arms(
    instance: dict, out: Path, client, skip: set[str] | None = None
) -> dict[str, bool | None]:
    from grade_fork import grade_patch, remove_task_image

    graded: dict[str, bool | None] = {}
    skipped = skip or set()
    for arm in ARMS:
        if arm in skipped:
            graded[arm] = None
            continue
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


def merge_run_checks(instance_id: str, arm: str, out: Path) -> None:
    """Provenance on every Graphirm graph. Reconstruction on a stable 5%."""
    graph = out / f"{arm}.graph.db"
    summary_path = out / f"{arm}.summary.json"
    summary: dict = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
    if not graph.is_file() or graph.stat().st_size == 0:
        return
    try:
        summary["provenance"] = provenance_summary(graph)
    except Exception as exc:
        summary["provenance"] = {"error": type(exc).__name__}
    if reconstruction_sampled(instance_id, arm):
        try:
            summary["reconstruction"] = reconstruct_check(
                graph,
                out / f"{arm}.pack-trace.jsonl",
                arm,
                f"{instance_id}::{arm}",
            )
        except Exception as exc:
            summary["reconstruction"] = {"error": type(exc).__name__}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def apply_config_guard(arm: str, out: Path, result: dict) -> None:
    reason = arm_problem(arm, out, result.get("diag") or result.get("error") or "")
    result.pop("diag", None)
    if not reason:
        return
    result["invalid_config"] = True
    result["status"] = "invalid_config"
    result["error"] = f"invalid_config: {reason}"[:400]


def preflight(root: Path, image: str) -> None:
    """One short call per arm. Refuse to start when the resolved settings differ."""
    problem = toml_problem()
    if problem:
        raise SystemExit(f"PREFLIGHT {problem}")
    client = docker.from_env(timeout=600)
    print(f"PREFLIGHT pull {image}", flush=True)
    client.images.pull(image)
    out = root / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    name = "t1-preflight"
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        pass
    container = client.containers.run(
        image,
        command=["sleep", "900"],
        detach=True,
        name=name,
        working_dir="/testbed",
        user="root",
    )
    try:
        for arm in ("G", "G-md", "G-b16", "G-b16-recency"):
            task = out / f"{arm}.issue.md"
            task.write_text("Reply with ok.\n")
            env = os.environ.copy()
            env.update(arm_env(arm))
            env.update(
                {
                    "CONTAINER": container.id,
                    "TASK_FILE": str(task),
                    "WORKDIR": "/testbed",
                    "GRAPH_OUT": str(out / f"{arm}.graph.db"),
                    "SUMMARY_OUT": str(out / f"{arm}.summary.json"),
                    "GRAPHIRM_BIN": str(BINARY),
                    "GRAPHIRM_IMAGE": image,
                    "GRAPHIRM_MAX_TURNS": "1",
                    "GRAPHIRM_MAX_COST": "0.05",
                    "GRAPHIRM_MODEL": "openrouter/deepseek/deepseek-v4-flash",
                }
            )
            proc = subprocess.run(
                ["sh", str(RUNNER)],
                cwd=REPO,
                env=env,
                timeout=180,
                capture_output=True,
                text=True,
            )
            reason = arm_problem(arm, out, (proc.stderr or "")[-4000:])
            if reason or proc.returncode not in (0, 2):
                tail = (proc.stderr or proc.stdout or "")[-300:].replace("\n", " ")
                raise SystemExit(f"PREFLIGHT {arm} {reason or 'exit ' + str(proc.returncode)} {tail}")
            print(f"PREFLIGHT {arm} model={GRAPHIRM_MODEL} provider={PROVIDER} reasoning={REASONING}", flush=True)
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.utils.serialize import recursive_merge

    base = get_config_from_spec(SWEBENCH_YAML)
    mini = get_config_from_spec(str(MINI_CONFIG))
    config = recursive_merge(base, mini)
    model_cfg = config.get("model", {})
    if model_cfg.get("model_name") != MINI_MODEL:
        raise SystemExit(f"PREFLIGHT M model {model_cfg.get('model_name')}")
    model = get_model(config=model_cfg)
    try:
        response = model._query([{"role": "user", "content": "Reply with ok."}], max_tokens=16)
    except Exception as exc:
        raise SystemExit(f"PREFLIGHT M {type(exc).__name__}: {exc}") from exc
    served = str(response.get("model") or "")
    if served and "v4.1" not in served and "v4-1" not in served:
        raise SystemExit(f"PREFLIGHT M served {served}")
    reason = mini_problem_from_cfg(model_cfg)
    if reason:
        raise SystemExit(f"PREFLIGHT M {reason}")
    print(f"PREFLIGHT M model={MINI_MODEL} provider={PROVIDER} reasoning={REASONING}", flush=True)


def mini_problem_from_cfg(model_cfg: dict) -> str | None:
    from config_guard import mini_problem

    return mini_problem(model_cfg)


def maybe_graph_report(root: Path, task_ids: list[str]) -> None:
    if len(task_ids) < 25:
        return
    dest = root / "graph-size-25.json"
    if dest.is_file():
        return
    sample = task_ids[:25]
    report = graph_size_report(root, sample)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"GRAPH_SIZE tasks={report['tasks']} graphs={report['graphs']} "
        f"median_gz={report['median_gz']} total_gz={report['total_gz']} "
        f"track_gz={report['track_gz']}",
        flush=True,
    )
    # Graph tarballs stay local until every track has finished.


def upload_graphs(root: Path, task_ids: list[str]) -> None:
    repo_id = os.environ.get("GRAPHIRM_GRAPH_DATASET", "consoulhub/graphirm-eval-graphs")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("GRAPH_UPLOAD huggingface_hub is not installed", flush=True)
        return
    api = HfApi()
    try:
        api.whoami()
    except Exception as exc:
        print(f"GRAPH_UPLOAD waiting for a Hugging Face login ({type(exc).__name__})", flush=True)
        return
    bundle = root / "graph-size-25"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir()
    for iid in task_ids:
        for arm in ("G", "G-md", "G-b16", "G-b16-recency"):
            src = root / iid / f"{arm}.graph.db"
            if src.is_file() and src.stat().st_size > 0:
                dest = bundle / f"{iid}__{arm}.graph.db"
                shutil.copy2(src, dest)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(bundle), repo_id=repo_id, repo_type="dataset")
    print(f"GRAPH_UPLOAD {repo_id}", flush=True)


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
    if (root / "stop-reason").is_file():
        raise SystemExit(f"STOP held {(root / 'stop-reason').read_text().strip()}")
    (root / "running").write_text("t1\n")
    arm_ledger = root / "arms.jsonl"
    task_ledger = root / "tasks.jsonl"
    latest_arms: dict[str, dict] = {}
    if arm_ledger.exists():
        for line in arm_ledger.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest_arms[record["instance_id"]] = record
    done_arms = {
        key
        for key, record in latest_arms.items()
        if record.get("request_caused")
        or (record.get("complete") is True and not record.get("error"))
    }
    dropped = []
    for key in list(done_arms):
        if latest_arms[key].get("request_caused"):
            continue
        iid, arm = key.split("::", 1)
        reason = arm_problem(arm, root / iid, "")
        if reason:
            done_arms.remove(key)
            dropped.append(f"{key} {reason}")
    if dropped:
        print("INVALID_CONFIG resume " + "; ".join(dropped[:8]), flush=True)
    instances = load_instances(headline_ids())
    done_tasks = {
        row["instance_id"]
        for row in instances
        if all(f"{row['instance_id']}::{arm}" in done_arms for arm in ARMS)
    }
    pending = [row for row in instances if row["instance_id"] not in done_tasks]
    print(
        f"START total=210 done={len(done_tasks)} pending={len(pending)} "
        f"spend={spent(arm_ledger):.2f} disk={used_pct():.1f}",
        flush=True,
    )
    if pending:
        preflight(root, pending[0]["image_name"])
    finished_ids = [row["instance_id"] for row in instances if row["instance_id"] in done_tasks]
    maybe_graph_report(root, finished_ids)
    finished = len(done_tasks)
    invalid_total = 0
    streak = 0
    request_streak = 0
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
                trace = (
                    out / f"{arm}.pack-trace.jsonl"
                    if reconstruction_sampled(iid, arm)
                    else None
                )
                result = run_graphirm(instance, arm, out, trace)
                merge_run_checks(iid, arm, out)
            result["instance_id"] = key
            result["task"] = iid
            result["image_head"] = head
            result["commit_mismatch"] = bool(
                result.get("start_commit") and head and result["start_commit"] != head
            )
            apply_config_guard(arm, out, result)
            result["complete"] = not bool(result.get("error"))
            files = [p for p in (out / f"{arm}.patch", out / f"{arm}.summary.json") if p.is_file()]
            with ledger_lock:
                append_ledger(arm_ledger, result)
            if not backup_record(TRACK, result, files):
                remember_failed_backup(TRACK, result, files)
            return result

        recover_wait = 60
        unresolved_arms: set[str] = set()
        while True:
            with ThreadPoolExecutor(max_workers=ARM_WORKERS) as pool:
                futures = [pool.submit(one_arm, arm) for arm in ARMS]
                arm_results = [future.result() for future in as_completed(futures)]
            invalid_arms = set()
            by_arm = {result.get("arm"): result for result in arm_results if not result.get("skipped")}
            ordered = [by_arm[arm] for arm in ARMS if arm in by_arm]
            request_hits = []
            config_hits = []
            infra_hits = []
            for result in ordered:
                kind = classify_arm(out, result.get("arm") or "", result.get("error") or "")
                if result.get("error"):
                    result["failure_class"] = kind
                if kind == REQUEST and result.get("error"):
                    request_hits.append(result)
                    request_streak += 1
                elif kind == CONFIG and result.get("error"):
                    config_hits.append(result)
                elif kind == INFRASTRUCTURE and result.get("error"):
                    infra_hits.append(result)
                    request_streak = 0
                elif result.get("invalid_config") or provider_error(result.get("error") or ""):
                    request_streak = 0
                    if result.get("invalid_config"):
                        invalid_arms.add(result.get("arm"))
                        invalid_total += 1
                        print(
                            f"INVALID_CONFIG {iid} {result.get('arm')} {result.get('error')}",
                            flush=True,
                        )
                    streak += 1
                else:
                    request_streak = 0
                    streak = 0
                if result.get("error"):
                    errors += 1
                    total_errors += 1
                http_429 += int(result.get("http_429") or 0)
                total_429 += int(result.get("http_429") or 0)
                if result.get("commit_mismatch"):
                    mismatches.append(result["arm"])
                    total_mismatch += 1
            if config_hits:
                (root / "stop-reason").write_text("config\n")
                detail = config_hits[0]
                raise SystemExit(
                    f"STOP config {iid} {detail.get('arm')} {detail.get('error')}"
                )
            for result in request_hits:
                result["request_caused"] = True
                result["complete"] = True
                result["unresolved"] = True
                append_ledger(arm_ledger, result)
                done_arms.add(str(result.get("instance_id")))
                unresolved_arms.add(str(result.get("arm")))
                print(
                    f"UNRESOLVED {iid} {result.get('arm')} {result.get('error')}",
                    flush=True,
                )
            if request_streak > 3:
                (root / "stop-reason").write_text("request\n")
                raise SystemExit(
                    f"STOP request streak={request_streak} tasks={finished} spend={spent(arm_ledger):.2f}"
                )
            if infra_hits:
                names = ",".join(str(result.get("arm")) for result in infra_hits)
                print(f"RECOVER {iid} arms={names} wait={recover_wait}", flush=True)
                import time

                time.sleep(recover_wait)
                recover_wait = min(recover_wait * 2, 1800)
                preflight(root, instance["image_name"])
                continue
            if streak > 3:
                (root / "stop-reason").write_text("config\n")
                print(
                    f"PAUSE invalid_config streak={streak} tasks={finished} spend={spent(arm_ledger):.2f}",
                    flush=True,
                )
                raise SystemExit(
                    f"PAUSE invalid_config streak={streak} tasks={finished} spend={spent(arm_ledger):.2f}"
                )
            break
        try:
            graded = grade_arms(instance, out, client, invalid_arms | unresolved_arms)
        except Exception:
            graded = {arm: None for arm in ARMS}
            print(f"GRADE_ERR {iid}", flush=True)
            traceback.print_exc()
        for arm in unresolved_arms:
            graded[arm] = False
        task_record = {
            "instance_id": iid,
            "complete": all(graded.get(arm) is not None for arm in ARMS),
            "image": instance["image_name"],
            "image_head": head,
            "commit_mismatch": mismatches,
            "errors": errors,
            "http_429": http_429,
            "invalid_config": sorted(invalid_arms),
            "grades_present": sum(value is not None for value in graded.values()),
        }
        # Resolved flags stay in the grade file. Progress lines do not print them.
        (out / "grades.json").write_text(json.dumps(graded, indent=2) + "\n")
        append_ledger(task_ledger, task_record)
        grade_file = out / "grades.json"
        if not backup_record(TRACK, task_record, [grade_file] if grade_file.is_file() else []):
            remember_failed_backup(
                TRACK, task_record, [grade_file] if grade_file.is_file() else []
            )
        if not task_record["complete"]:
            continue
        finished += 1
        finished_ids.append(iid)
        maybe_graph_report(root, finished_ids)
        if finished % 10 == 0:
            still = retry_failed_backups(TRACK)
            if push_backup():
                print(f"PUSH tasks={finished} backup_failed={still}", flush=True)
            else:
                print(f"PUSH_WARN tasks={finished} backup_failed={still}", flush=True)
        if finished % 25 == 0:
            spend_now = spent(arm_ledger)
            projected = spend_now / finished * 210
            over = " projected_over_33" if projected > 33 else ""
            print(
                f"PROGRESS tasks={finished} spend={spend_now:.2f} "
                f"projected_210={projected:.2f}{over} "
                f"disk={used_pct():.1f} errors={total_errors} http_429={total_429} "
                f"commit_mismatch={total_mismatch} invalid_config={invalid_total} "
                f"backup_failed={pending_backup_count(TRACK)}",
                flush=True,
            )
    if finished >= 210:
        from closing import write_report
        from reconstruct_check import sequential_sample_rate

        equal, checked = sequential_sample_rate(root)
        print(f"RECONSTRUCTION sequential={equal}/{checked}", flush=True)
        write_report(root)
        from integrity import finish_track

        problems = finish_track(TRACK, root)
        if problems:
            print(f"INTEGRITY_FAIL {len(problems)}", flush=True)
            for line in problems[:30]:
                print(f"INTEGRITY {line}", flush=True)
        else:
            (root / "running").unlink(missing_ok=True)
            print("INTEGRITY_OK t1", flush=True)
    print(f"HOLD tasks={finished} spend={spent(arm_ledger):.2f}", flush=True)


if __name__ == "__main__":
    main()
