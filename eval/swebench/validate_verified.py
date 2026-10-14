#!/usr/bin/env python3
"""Gold and empty checks for the 273 SWE-Bench-CL tasks.

Uses site-packages swebench 4.1.0 and the published Verified images
(namespace swebench). Do not put the SWE-rebench fork on PYTHONPATH
when running this script.

An empty patch is graded by running the eval script with no model patch.
swebench 4.1.0's run_instance tries to apply the patch, and an empty
patch does not apply, so that path never runs the tests.

Outputs go under /home/krs/graphirm-runs/t2-validity/.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docker
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_build import (
    build_container,
    close_logger,
    setup_logger,
)
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    remove_image,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec

from durable import append_ledger, backup_record, load_done, runs_dir

CURRICULUM = Path("/home/krs/graphirm-runs/sources/SWE-Bench-CL-Curriculum.json")
TRACK = "t2-validity"
WORKERS = 2
DISK_LIMIT = 82.0
NAMESPACE = "swebench"


def used_pct() -> float:
    usage = shutil.disk_usage("/")
    return 100.0 * usage.used / usage.total


def wait_disk() -> None:
    import time

    while used_pct() >= DISK_LIMIT:
        print(f"DISK {used_pct():.1f} waiting", flush=True)
        time.sleep(30)


def curriculum_ids() -> list[str]:
    data = json.loads(CURRICULUM.read_text())
    ids = []
    for sequence in data["sequences"]:
        for task in sequence["tasks"]:
            ids.append(task["metadata"]["instance_id"])
    if len(ids) != 273 or len(set(ids)) != 273:
        raise SystemExit(f"expected 273 unique ids, found {len(ids)} unique {len(set(ids))}")
    return ids


def load_instances(ids: list[str]) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    found = {row["instance_id"]: dict(row) for row in dataset}
    missing = [iid for iid in ids if iid not in found]
    if missing:
        raise SystemExit(f"missing {len(missing)} verified instances, first {missing[:5]}")
    return [found[iid] for iid in ids]


def prepare_spec(instance: dict):
    spec = make_test_spec(instance, namespace=NAMESPACE)
    if not spec.docker_specs:
        spec.docker_specs = {}
    return spec


def _drop(client, name: str) -> None:
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        return
    except Exception:
        return


def grade_empty(spec, client, timeout: int) -> bool:
    pred = {
        KEY_INSTANCE_ID: spec.instance_id,
        KEY_PREDICTION: "",
        KEY_MODEL: "empty",
    }
    model_name = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / "t2-valid-empty" / model_name / spec.instance_id
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return bool(report[spec.instance_id]["resolved"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(spec.instance_id + "-empty", log_dir / "run_instance.log")
    container = None
    try:
        _drop(client, spec.get_instance_container_name("t2-valid-empty"))
        container = build_container(spec, client, "t2-valid-empty", logger, False, False)
        container.start()
        eval_file = log_dir / "eval.sh"
        eval_file.write_text(spec.eval_script)
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))
        test_output, timed_out, _runtime = exec_run_with_timeout(
            container, "/bin/bash /eval.sh", timeout
        )
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(test_output or "")
        if timed_out:
            raise TimeoutError(f"empty timed out after {timeout}s")
        report = get_eval_report(
            test_spec=spec,
            prediction=pred,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        report_path.write_text(json.dumps(report))
        return bool(report[spec.instance_id]["resolved"])
    finally:
        cleanup_container(client, container, logger)
        close_logger(logger)


def one(instance: dict) -> dict:
    iid = instance["instance_id"]
    wait_disk()
    spec = prepare_spec(instance)
    timeout = 1800
    client = docker.from_env(timeout=600)
    record = {
        "instance_id": iid,
        "base_commit": instance.get("base_commit"),
        "gold_resolved": None,
        "empty_resolved": None,
        "gold_completed": None,
        "gold_error": None,
        "empty_error": None,
        "complete": False,
    }
    try:
        _drop(client, spec.get_instance_container_name("t2-valid-gold"))
        pred = {
            KEY_INSTANCE_ID: iid,
            KEY_PREDICTION: instance["patch"],
            KEY_MODEL: "gold",
        }
        result = run_instance(
            spec,
            pred,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id="t2-valid-gold",
            timeout=timeout,
        )
        record["gold_completed"] = bool(result.get("completed"))
        if result.get("completed"):
            record["gold_resolved"] = bool(result.get("resolved"))
        else:
            record["gold_error"] = "gold did not finish"
        if record["gold_error"] is None:
            record["empty_resolved"] = grade_empty(spec, client, timeout)
        record["complete"] = (
            record["gold_resolved"] is not None and record["empty_resolved"] is not None
        )
        return record
    finally:
        try:
            remove_image(client, spec.instance_image_key, "quiet")
        except Exception as exc:
            print(f"RM_ERR {iid} {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    root = runs_dir(TRACK)
    os.chdir(root)
    ledger = root / "ledger.jsonl"
    done = load_done(ledger)
    instances = load_instances(curriculum_ids())
    pending = [row for row in instances if row["instance_id"] not in done]
    print(
        f"START total={len(instances)} done={len(done)} pending={len(pending)} disk={used_pct():.1f}",
        flush=True,
    )
    if not pending:
        print("HOLD nothing pending", flush=True)
        return

    def run_one(instance: dict) -> None:
        iid = instance["instance_id"]
        try:
            record = one(instance)
        except Exception as exc:
            record = {
                "instance_id": iid,
                "gold_resolved": None,
                "empty_resolved": None,
                "gold_error": f"{type(exc).__name__}: {exc}"[:400],
                "empty_error": None,
                "complete": False,
                "trace": traceback.format_exc()[-600:],
            }
        append_ledger(ledger, record)
        backup_record(TRACK, record)
        print(
            f"VALID {iid} gold={record.get('gold_resolved')} empty={record.get('empty_resolved')} "
            f"complete={record.get('complete')} gold_error={record.get('gold_error')} "
            f"empty_error={record.get('empty_error')}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, row) for row in pending]
        for future in as_completed(futures):
            future.result()
    print("HOLD validation finished", flush=True)


if __name__ == "__main__":
    main()
