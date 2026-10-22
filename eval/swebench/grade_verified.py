#!/usr/bin/env python3
"""Grade one SWE-bench Verified patch with site-packages swebench 4.1.0.

T2 (SWE-Bench-CL) uses the published Verified images under the swebench
namespace. Do not import grade_fork (the SWE-rebench fork) in the same
process; that module puts the fork first on sys.path.

Unlike run_instance, this grader does not reuse an existing report.json,
distinguishes a patch that fails to apply (settled, unresolved) from a
harness failure (retry once), and writes the patch without newline
translation so CR bytes reach git apply unchanged.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import docker
from swebench.harness.constants import (
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_TEST_OUTPUT,
    UTF8,
)
from swebench.harness.docker_build import build_container, close_logger, setup_logger
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    remove_image,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.test_spec.test_spec import make_test_spec

NAMESPACE = "swebench"
SETTLED = {"completed", "apply_failed", "tests_timed_out"}


@dataclass
class Grade:
    status: str
    resolved: bool | None
    error: str = ""


def prepare_spec(instance: dict):
    spec = make_test_spec(instance, namespace=NAMESPACE)
    if not spec.docker_specs:
        spec.docker_specs = {}
    return spec


def image_name(instance: dict) -> str:
    return prepare_spec(instance).instance_image_key


def _drop(client, name: str) -> None:
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        return
    except Exception:
        return


def grade_once(instance: dict, patch: str, client, timeout: int, log_dir: Path, run_id: str) -> Grade:
    spec = prepare_spec(instance)
    pred = {KEY_INSTANCE_ID: spec.instance_id, KEY_PREDICTION: patch, KEY_MODEL: run_id}
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"{spec.instance_id}-{run_id}-{int(time.time())}", log_dir / "run_instance.log")
    container = None
    try:
        _drop(client, spec.get_instance_container_name(run_id))
        container = build_container(spec, client, run_id, logger, False, False)
        container.start()
        patch_file = log_dir / "patch.diff"
        with open(patch_file, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            handle.write(patch)
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))
        applied = False
        output = b""
        for cmd in GIT_APPLY_CMDS:
            val = container.exec_run(f"{cmd} {DOCKER_PATCH}", workdir=DOCKER_WORKDIR, user=DOCKER_USER)
            output = val.output or b""
            if val.exit_code == 0:
                applied = True
                break
        if not applied:
            (log_dir / "apply.log").write_bytes(output)
            return Grade("apply_failed", False, output.decode(UTF8, errors="replace")[-300:])
        eval_file = log_dir / "eval.sh"
        eval_file.write_text(spec.eval_script)
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))
        test_output, timed_out, _runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(test_output or "")
        if timed_out:
            return Grade("tests_timed_out", False, f"eval.sh exceeded {timeout}s")
        report = get_eval_report(
            test_spec=spec,
            prediction=pred,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        (log_dir / "report.json").write_text(json.dumps(report))
        return Grade("completed", bool(report[spec.instance_id]["resolved"]))
    except Exception as exc:
        return Grade("harness_exception", None, f"{type(exc).__name__}: {exc}"[:400])
    finally:
        cleanup_container(client, container, logger)
        close_logger(logger)


def grade_patch(instance: dict, patch: str, client, timeout: int, log_root: Path, run_id: str) -> Grade:
    """Two attempts on harness failure, then an unsettled Grade for the caller to log."""
    last = Grade("not_run", None)
    for attempt in (1, 2):
        last = grade_once(instance, patch, client, timeout, log_root / f"attempt{attempt}", run_id)
        if last.status in SETTLED:
            return last
    return last


def remove_task_image(instance: dict, client) -> None:
    try:
        remove_image(client, prepare_spec(instance).instance_image_key, "quiet")
    except Exception:
        return
