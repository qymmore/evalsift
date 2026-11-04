"""Validity and failure checks.

A check is a function ``(run, config) -> Iterable[Finding]`` registered under a
name with a default severity. Two kinds of problem are kept distinct:

* **Invalid runs** (ERROR): the trajectory does not measure the model's
  behaviour, e.g. an infrastructure error or a missing score. Exclude these
  from rates and report how many there were.
* **Failed turns** (WARNING): the run is usable but a turn went wrong, e.g. a
  tool error or a malformed tool call. Worth reading; not grounds to drop it.

Severities can be overridden per check, because what counts as invalid depends
on the study. Hitting a message limit is a normal outcome in some benchmarks
and a broken run in others.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from evalsift.model import Finding, Run, Severity


@dataclass
class CheckConfig:
    repeat_threshold: int = 3
    severity_overrides: dict[str, Severity] = field(default_factory=dict)
    disabled: set[str] = field(default_factory=set)


CheckFn = Callable[[Run, CheckConfig], Iterable[Finding]]


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    description: str
    fn: CheckFn


REGISTRY: dict[str, Check] = {}


def register(name: str, severity: Severity, description: str) -> Callable[[CheckFn], CheckFn]:
    def wrap(fn: CheckFn) -> CheckFn:
        if name in REGISTRY:
            raise ValueError(f"check {name!r} is already registered")
        REGISTRY[name] = Check(name, severity, description, fn)
        return fn

    return wrap


def run_checks(runs: Iterable[Run], config: CheckConfig | None = None) -> list[Finding]:
    config = config or CheckConfig()
    unknown = (set(config.severity_overrides) | config.disabled) - set(REGISTRY)
    if unknown:
        raise ValueError(f"unknown check name(s): {', '.join(sorted(unknown))}")
    findings: list[Finding] = []
    for run in runs:
        for check in REGISTRY.values():
            if check.name in config.disabled:
                continue
            severity = config.severity_overrides.get(check.name, check.severity)
            for f in check.fn(run, config):
                findings.append(Finding(f.run_id, f.check, severity, f.message, f.turn))
    return findings


def invalid_run_ids(findings: Iterable[Finding]) -> set[str]:
    return {f.run_id for f in findings if f.severity is Severity.ERROR}


def _f(run: Run, check: str, message: str, turn: int | None = None) -> Finding:
    # Severity here is a placeholder; run_checks applies the effective one.
    return Finding(run.run_id, check, Severity.INFO, message, turn)


# --------------------------------------------------------------------------
# Run-level checks
# --------------------------------------------------------------------------


@register("sample_error", Severity.ERROR, "The sample raised an error and did not finish.")
def _sample_error(run: Run, _: CheckConfig) -> Iterator[Finding]:
    if run.error:
        yield _f(run, "sample_error", _first_line(run.error))


@register("missing_score", Severity.ERROR, "No usable score was recorded.")
def _missing_score(run: Run, _: CheckConfig) -> Iterator[Finding]:
    if run.error:
        return  # already invalid; one root cause is enough
    if not run.scores:
        yield _f(run, "missing_score", "run has no scores")
        return
    for s in run.scores:
        if s.value is None:
            yield _f(run, "missing_score", f"scorer {s.name!r} returned no value")


@register("limit_hit", Severity.WARNING, "A message, token, time or cost limit ended the run.")
def _limit_hit(run: Run, _: CheckConfig) -> Iterator[Finding]:
    if run.limit:
        yield _f(run, "limit_hit", f"stopped by limit ({run.limit})")


@register("no_turns", Severity.ERROR, "The model never produced an assistant message.")
def _no_turns(run: Run, _: CheckConfig) -> Iterator[Finding]:
    if not run.turns and not run.error:
        yield _f(run, "no_turns", "no assistant messages in transcript")


@register(
    "empty_final",
    Severity.WARNING,
    "The last assistant message has no text and no tool calls.",
)
def _empty_final(run: Run, _: CheckConfig) -> Iterator[Finding]:
    if run.turns and run.turns[-1].is_empty and not run.limit:
        yield _f(run, "empty_final", "final answer is empty", run.turns[-1].index)


# --------------------------------------------------------------------------
# Turn-level checks
# --------------------------------------------------------------------------


@register(
    "truncated_output",
    Severity.WARNING,
    "Generation stopped on the output-token or context limit.",
)
def _truncated(run: Run, _: CheckConfig) -> Iterator[Finding]:
    for t in run.turns:
        if t.stop_reason in ("max_tokens", "model_length"):
            yield _f(run, "truncated_output", f"stop_reason={t.stop_reason}", t.index)


@register("content_filter", Severity.WARNING, "The provider's content filter stopped generation.")
def _content_filter(run: Run, _: CheckConfig) -> Iterator[Finding]:
    for t in run.turns:
        if t.stop_reason == "content_filter":
            yield _f(run, "content_filter", "stop_reason=content_filter", t.index)


@register("model_error", Severity.WARNING, "The model API call for this turn returned an error.")
def _model_error(run: Run, _: CheckConfig) -> Iterator[Finding]:
    for t in run.turns:
        if t.model_error:
            yield _f(run, "model_error", _first_line(t.model_error), t.index)


@register("tool_parse_error", Severity.WARNING, "A tool call's arguments could not be parsed.")
def _tool_parse(run: Run, _: CheckConfig) -> Iterator[Finding]:
    for t in run.turns:
        for c in t.tool_calls:
            if c.parse_error:
                yield _f(
                    run,
                    "tool_parse_error",
                    f"{c.function}: {_first_line(c.parse_error)}",
                    t.index,
                )


@register("tool_error", Severity.WARNING, "A tool returned an error result.")
def _tool_error(run: Run, _: CheckConfig) -> Iterator[Finding]:
    for t in run.turns:
        for r in t.tool_results:
            if r.error_type or r.error_message:
                kind = r.error_type or "error"
                msg = _first_line(r.error_message or "")
                yield _f(run, "tool_error", f"{r.function or '?'} [{kind}] {msg}".strip(), t.index)


@register(
    "unanswered_tool_call",
    Severity.WARNING,
    "A tool call never received a result in the transcript.",
)
def _unanswered(run: Run, _: CheckConfig) -> Iterator[Finding]:
    # When an error or limit ended the run, the last turn's calls going
    # unanswered is a consequence already reported, not a separate failure.
    stopped_early = bool(run.error or run.limit)
    for t in run.turns:
        if stopped_early and t is run.turns[-1]:
            continue
        answered = {r.call_id for r in t.tool_results}
        for c in t.tool_calls:
            if c.id not in answered:
                yield _f(run, "unanswered_tool_call", f"{c.function} (id={c.id})", t.index)


@register(
    "repeated_tool_call",
    Severity.WARNING,
    "The same tool call (function and arguments) was made several turns in a row.",
)
def _repeated(run: Run, config: CheckConfig) -> Iterator[Finding]:
    streak = 0
    previous: str | None = None
    for t in run.turns:
        key = (
            json.dumps(
                [(c.function, c.arguments) for c in t.tool_calls],
                sort_keys=True,
                default=str,
            )
            if t.tool_calls
            else None
        )
        if key is not None and key == previous:
            streak += 1
        else:
            streak = 1 if key is not None else 0
        previous = key
        if streak == config.repeat_threshold:
            names = ", ".join(c.function for c in t.tool_calls)
            yield _f(
                run,
                "repeated_tool_call",
                f"{names} repeated {streak} turns in a row (possible loop)",
                t.index,
            )


@register(
    "empty_turn",
    Severity.WARNING,
    "A non-final assistant message has no text, reasoning or tool calls.",
)
def _empty_turn(run: Run, _: CheckConfig) -> Iterator[Finding]:
    for t in run.turns[:-1]:
        if t.is_empty:
            yield _f(run, "empty_turn", "assistant turn is empty", t.index)


def _first_line(text: str, width: int = 200) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= width else line[: width - 1] + "…"
