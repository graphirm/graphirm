"""Harbor agent for the mini-swe-agent arm.

Harbor already ships `mini-swe-agent`. This subclass keeps that install, run,
and trajectory collection, and pins the eval budgets: the same model string,
reasoning effort, frozen LiteLLM registry, and a 0.50 USD cost limit. The
built-in agent also exports `MSWEA_COST_TRACKING=ignore_errors`. With the
registry in place the price resolves, so the cost limit uses the frozen
table; ignore_errors only applies if registration fails.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import yaml
import importlib.util

from harbor.agents.installed.mini_swe_agent import MiniSweAgent

def _revert_shell(patch_file: str) -> str:
    path = Path(__file__).resolve().parent / "test_filter.py"
    spec = importlib.util.spec_from_file_location("t4_test_filter_mini", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.revert_shell(patch_file)


def prepare_mini_command(command: str, patch_file: str = "/logs/agent/M.patch") -> str:
    """Insert the frozen-cost patch and the test-file filter.

    The task text stays inside Harbor's shlex.quote of --task. This function
    does not re-quote it.
    """
    command = command.replace(
        "&& mini-swe-agent --help",
        '&& export PATH="/workspace/.local/bin:$PATH" && mini-swe-agent --help',
    )
    if "mini-swe-agent --yolo" in command:
        command = command.replace(
            "mini-swe-agent --yolo",
            "export PATH=\"/workspace/.local/bin:$HOME/.local/bin:$PATH\"; "
            "bin=$(command -v mini-swe-agent); "
            "line=$(head -n 1 \"$bin\"); "
            "py=${line#\\#!}; py=${py# }; "
            "case \"$py\" in *env\\ *) py=$(command -v ${py##* });; esac; "
            f"printf '%s' '{_COST_PATCH_B64}' | base64 -d | \"$py\"; "
            "mini-swe-agent --yolo",
        )
        command += "; " + _revert_shell(patch_file)
    return command


REGISTRY = Path(__file__).resolve().parents[1] / "litellm_model_registry.json"
MINI_CONFIG = Path(__file__).resolve().parents[1] / "adapters" / "mini_config.yaml"

# OpenRouter's model class returns usage.cost, which is the bill. The $0.50 cap
# has to see the frozen formula instead. The patch runs inside the trial image,
# on that image's copy of mini-swe-agent, and leaves usage.cost in the trajectory.
_COST_PATCH = r"""
import pathlib, re
import minisweagent.models.openrouter_model as model
path = pathlib.Path(model.__file__)
text = path.read_text()
replacement = '''    def _calculate_cost(self, response) -> dict[str, float]:
        usage = response.get("usage", {}) or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        output = int(usage.get("completion_tokens") or 0)
        if cached > prompt:
            cached = prompt
        cost = (prompt - cached) * 4.2e-8 + cached * 9e-9 + output * 6e-7
        return {"cost": float(cost)}
'''
updated, count = re.subn(
    r"    def _calculate_cost\(self, response\).*?(?=\n    def )",
    replacement,
    text,
    count=1,
    flags=re.S,
)
if count != 1 or "4.2e-8" not in updated:
    raise SystemExit("frozen cost patch did not match openrouter_model.py")
path.write_text(updated)
"""
_COST_PATCH_B64 = base64.b64encode(_COST_PATCH.encode()).decode()


class EvalMiniSweAgent(MiniSweAgent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("litellm_model_registry_file", str(REGISTRY))
        kwargs.setdefault("reasoning_effort", "high")
        if kwargs.get("cost_limit") in (None, "0", 0):
            kwargs["cost_limit"] = "0.50"
        if "config" not in kwargs and "config_file" not in kwargs:
            cfg = yaml.safe_load(MINI_CONFIG.read_text())
            agent_cfg = cfg.setdefault("agent", {})
            agent_cfg["cost_limit"] = 0.5
            agent_cfg["step_limit"] = 100
            agent_cfg["wall_time_limit_seconds"] = 1800
            kwargs["config"] = cfg
        super().__init__(*args, **kwargs)

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        # Harbor's installer puts the tool on $HOME/.local/bin. On these
        # images uv writes it under /workspace/.local/bin instead, and the
        # later PATH assignment drops that directory. Put it back immediately
        # before the CLI is invoked. The task string is already shlex.quoted
        # by Harbor; prepare_mini_command does not quote it again.
        command = prepare_mini_command(command)
        # Harbor forwards MSWEA_API_KEY when that name is set on the host,
        # and then does not also forward OPENROUTER_API_KEY. LiteLLM's
        # OpenRouter client reads OPENROUTER_API_KEY.
        forwarded = dict(env or {})
        if "OPENROUTER_API_KEY" not in forwarded:
            key = os.environ.get("OPENROUTER_API_KEY") or forwarded.get("MSWEA_API_KEY")
            if key:
                forwarded["OPENROUTER_API_KEY"] = key
        return await super().exec_as_agent(
            environment,
            command=command,
            env=forwarded or None,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )

    @staticmethod
    def name() -> str:
        return "eval-mini-swe-agent"
