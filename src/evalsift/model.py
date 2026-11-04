"""Normalised, source-independent representation of agent runs.

Adapters convert a log format (currently Inspect) into these types, and every
check, table and export works only on them. That keeps the analysis code free
of any one framework's schema and makes new adapters cheap to add.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """How much a finding matters.

    ERROR makes the whole run invalid: it should be excluded from behavioural
    statistics and reported separately. WARNING flags a turn or run worth
    reading but does not exclude it. INFO is diagnostic only.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2}[self.value]


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str | None
    function: str | None
    content: str
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class Turn:
    """One assistant message, plus the tool results that answered it."""

    index: int
    text: str
    reasoning: list[str] = field(default_factory=list)
    reasoning_redacted: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    stop_reason: str | None = None
    latency_s: float | None = None
    model_error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.reasoning and not self.tool_calls


@dataclass(frozen=True)
class Score:
    name: str
    value: Any
    answer: str | None = None
    explanation: str | None = None

    def as_binary(self) -> int | None:
        """Map a score to 1/0 when it is unambiguously pass/fail, else None.

        Inspect's conventions: "C" correct, "I" incorrect. Booleans and the
        integers 0/1 are also accepted. Partial ("P"), no-answer ("N") and
        continuous scores return None so they never enter a binomial interval.
        """
        v = self.value
        if isinstance(v, bool):
            return int(v)
        if v in ("C", "I"):
            return 1 if v == "C" else 0
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
        return None


@dataclass
class Run:
    """One sample-epoch: a single trajectory of the agent under evaluation."""

    run_id: str
    source: str
    task: str
    model: str
    sample_id: str
    epoch: int
    turns: list[Turn]
    scores: list[Score] = field(default_factory=list)
    input_text: str = ""
    target: str = ""
    error: str | None = None
    limit: str | None = None
    final_stop_reason: str | None = None
    total_time_s: float | None = None
    working_time_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)


@dataclass(frozen=True)
class Finding:
    run_id: str
    check: str
    severity: Severity
    message: str
    turn: int | None = None
