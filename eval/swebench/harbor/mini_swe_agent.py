"""Harbor agent for the mini-swe-agent arm.

Harbor already ships `mini-swe-agent`. This subclass keeps that install, run,
and trajectory collection, and pins the eval budgets: the same model string,
reasoning effort, frozen LiteLLM registry, and a 0.50 USD cost limit. The
built-in agent also exports `MSWEA_COST_TRACKING=ignore_errors`. With the
registry in place the price resolves, so the cost limit uses the frozen
table; ignore_errors only applies if registration fails.
"""

from __future__ import annotations

import os
from pathlib import Path

from harbor.agents.installed.mini_swe_agent import MiniSweAgent

REGISTRY = Path(__file__).resolve().parents[1] / "litellm_model_registry.json"


class EvalMiniSweAgent(MiniSweAgent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("litellm_model_registry_file", str(REGISTRY))
        kwargs.setdefault("reasoning_effort", "high")
        if kwargs.get("cost_limit") in (None, "0", 0):
            kwargs["cost_limit"] = "0.50"
        super().__init__(*args, **kwargs)

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        # Harbor's installer puts the tool on $HOME/.local/bin. On these
        # images uv writes it under /workspace/.local/bin instead, and the
        # later PATH assignment drops that directory. Put it back immediately
        # before the CLI is invoked.
        command = command.replace(
            "&& mini-swe-agent --help",
            '&& export PATH="/workspace/.local/bin:$PATH" && mini-swe-agent --help',
        )
        command = command.replace(
            "mini-swe-agent --yolo",
            'export PATH="/workspace/.local/bin:$PATH"; mini-swe-agent --yolo',
        )
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
