#!/usr/bin/env python3
"""Grade one SWE-rebench patch with the SWE-rebench fork.

The fork checkout is /home/krs/graphirm-runs/SWE-bench-fork. This script
puts that tree first on sys.path so it is not mixed with site-packages
swebench 4.1.0.

An empty patch is graded by running the official eval script with no model
patch. The fork's CLI skips empty patches and never runs those tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

FORK = Path("/home/krs/graphirm-runs/SWE-bench-fork")
sys.path.insert(0, str(FORK))

import docker  # noqa: E402

from swebench.harness.constants import (  # noqa: E402
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
)
from swebench.harness.docker_build import (  # noqa: E402
    build_container,
    close_logger,
    setup_logger,
)
from swebench.harness.docker_utils import (  # noqa: E402
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    remove_image,
)
from swebench.harness.eval import get_log_dir, run_instance  # noqa: E402
from swebench.harness.grading import get_eval_report  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402


def prepare_spec(instance: dict):
    spec = make_test_spec(instance)
    if not spec.docker_specs:
        spec.docker_specs = {}
    return spec


def grade_gold(instance: dict, client, timeout: int, run_id: str = "t1-gold") -> bool:
    spec = prepare_spec(instance)
    pred = {
        KEY_INSTANCE_ID: spec.instance_id,
        KEY_PREDICTION: instance["patch"],
        KEY_MODEL: "gold",
    }
    _drop(client, spec.get_instance_container_name(run_id))
    _iid, report = run_instance(
        spec,
        pred,
        rm_image=False,
        force_rebuild=False,
        client=client,
        run_id=run_id,
        timeout=timeout,
    )
    return bool(report[spec.instance_id]["resolved"])


def grade_empty(instance: dict, client, timeout: int, run_id: str = "t1-empty") -> bool:
    """Run conda plus test_patch with no model patch."""
    spec = prepare_spec(instance)
    pred = {
        KEY_INSTANCE_ID: spec.instance_id,
        KEY_PREDICTION: "",
        KEY_MODEL: "empty",
    }
    log_dir = get_log_dir(pred, run_id, spec.instance_id)
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return bool(report[spec.instance_id]["resolved"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(spec.instance_id, log_dir / "run_instance.log")
    container = None
    try:
        _drop(client, spec.get_instance_container_name(run_id))
        container = build_container(spec, client, run_id, logger, False, False, None)
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


def remove_task_image(instance: dict, client) -> None:
    spec = prepare_spec(instance)
    remove_image(client, spec.instance_image_key, "quiet")


def _drop(client, name: str) -> None:
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        return
    except Exception:
        return
