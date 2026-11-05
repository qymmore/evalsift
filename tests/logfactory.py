"""Generate real Inspect logs offline with Inspect's mock model.

Each scenario is a one-sample eval whose model outputs are scripted, so the
logs exercise every check through Inspect's real logging code, with no API
calls and no network.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import inspect_ai.model._model as _inspect_model
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, ToolError, tool

# Inspect estimates token counts with a tiktoken vocabulary it downloads on
# first use. Swap in a character-based estimate so fixtures build offline.
_inspect_model.count_text_tokens = lambda text: max(1, len(text) // 4)  # type: ignore[assignment]

M = "mockllm/model"
# Fake credential assembled at runtime so secret scanners don't flag the repo.
FAKE_KEY = "sk-" + "proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


@tool
def add() -> Tool:
    async def execute(x: int, y: int) -> int:
        """Add two integers.

        Args:
            x: First number.
            y: Second number.
        """
        return x + y

    return execute


@tool
def lookup() -> Tool:
    async def execute(key: str) -> str:
        """Look up a record.

        Args:
            key: Record key.
        """
        if key == "missing":
            raise ToolError("record not found")
        if key == "crash":
            raise RuntimeError("database connection dropped")
        if key == "env":
            return f"OPENAI_API_KEY={FAKE_KEY}"
        return f"value-for-{key}"

    return execute


def call(fn: str, **args: Any) -> ModelOutput:
    return ModelOutput.for_tool_call(M, fn, args)


def say(text: str) -> ModelOutput:
    return ModelOutput.from_content(M, text)


# Each list holds exactly the outputs one epoch consumes, so multi-epoch runs
# replay the same trajectory instead of drifting through a shared queue.
SCENARIOS: dict[str, tuple[list[ModelOutput], dict[str, Any]]] = {
    "clean": ([call("add", x=1, y=2), say("3")], {}),
    "wrong": ([say("5")], {}),
    "tool_error": ([call("lookup", key="missing"), say("3")], {}),
    "loop": ([call("lookup", key="a")] * 4 + [say("3")], {}),
    "limit": ([call("lookup", key="a")] * 3, {"message_limit": 6}),
    "crash": ([call("lookup", key="crash")], {}),
    "empty": ([say("")], {}),
    "leak": ([call("lookup", key="env"), say("3")], {}),
    "truncated": ([ModelOutput.from_content(M, "The answer is", stop_reason="max_tokens")], {}),
    "reasoning": (
        [
            ModelOutput.from_content(
                M,
                [ContentReasoning(reasoning="1 plus 2 is 3."), ContentText(text="3")],
            )
        ],
        {},
    ),
}


def build(
    log_dir: Path,
    scenarios: dict[str, tuple[list[ModelOutput], dict[str, Any]]] | None = None,
    epochs: int = 1,
) -> Path:
    for name, (outputs, options) in (scenarios or SCENARIOS).items():
        task = Task(
            name=name,
            dataset=[Sample(id=name, input="What is 1 + 2?", target="3")],
            solver=[use_tools(add(), lookup()), generate()],
            scorer=match(),
            epochs=epochs,
            **options,
        )
        model = get_model(M, custom_outputs=outputs * epochs)
        eval(
            task,
            model=model,
            log_dir=str(log_dir),
            display="none",
            fail_on_error=False,
            max_samples=1,
        )
    return log_dir


Factory = Callable[..., Path]


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-logs")
    build(out, epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    print(f"Wrote demo logs for {len(SCENARIOS)} scenarios to {out}/")
