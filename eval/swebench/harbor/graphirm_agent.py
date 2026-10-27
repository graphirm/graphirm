"""Harbor installed agent that runs the Graphirm static binary inside the task.

Install uploads the binary, the pinned eval config, and the frozen price
table. Run writes the task instruction to a file and calls `graphirm run`.
Harbor collects `/logs/agent/graph.db` and `/logs/agent/summary.json` after
the trial. The instruction string is the task text Harbor passes into `run`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import importlib.util

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

def _graphirm_shell(patch_file: str, model: str, max_turns: int, cost_limit: float) -> str:
    path = Path(__file__).resolve().parent / "test_filter.py"
    spec = importlib.util.spec_from_file_location("t4_test_filter", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.graphirm_shell(
        patch_file,
        model=model,
        max_turns=max_turns,
        cost_limit=cost_limit,
    )


REPO = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO / "target/x86_64-unknown-linux-musl/release/graphirm"
DEFAULT_CONFIG = REPO / "eval/swebench/adapters/eval-agent.toml"
DEFAULT_PRICES = REPO / "eval/swebench/prices.lock.json"


class GraphirmAgent(BaseInstalledAgent):
    def __init__(
        self,
        max_turns: int = 100,
        cost_limit: float = 0.50,
        binary: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_turns = int(max_turns)
        self._cost_limit = float(cost_limit)
        self._binary = Path(binary) if binary else DEFAULT_BINARY

    @staticmethod
    def name() -> str:
        return "graphirm"

    async def install(self, environment: BaseEnvironment) -> None:
        if not self._binary.is_file():
            raise FileNotFoundError(f"graphirm binary not found: {self._binary}")
        await environment.upload_file(self._binary, "/usr/local/bin/graphirm")
        await environment.upload_file(DEFAULT_CONFIG, "/tmp/eval-agent.toml")
        await environment.upload_file(DEFAULT_PRICES, "/tmp/prices.lock.json")
        await self.exec_as_root(environment, command="chmod 755 /usr/local/bin/graphirm")

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self._upload_config_text(
            environment,
            content=instruction,
            remote_path="/tmp/issue.md",
            filename="issue.md",
        )
        key = os.environ.get("OPENROUTER_API_KEY", "")
        env = {"OPENROUTER_API_KEY": key} if key else None
        await self.exec_as_agent(
            environment,
            command=_graphirm_shell(
                "/logs/agent/G.patch",
                model=self.model_name or "openrouter/deepseek/deepseek-v4.1-flash",
                max_turns=self._max_turns,
                cost_limit=self._cost_limit,
            ),
            env=env,
            timeout_sec=1800,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        summary_path = self.logs_dir / "summary.json"
        if not summary_path.is_file():
            return
        summary = json.loads(summary_path.read_text())
        context.n_input_tokens = summary.get("tokens_in")
        context.n_output_tokens = summary.get("tokens_out")
        context.cost_usd = summary.get("cost_usd")
