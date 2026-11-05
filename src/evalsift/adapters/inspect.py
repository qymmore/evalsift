"""Adapter for Inspect (inspect_ai) evaluation logs, both .eval and .json.

Reading goes through Inspect's own ``read_eval_log`` so the adapter follows
whatever log-format versions the installed Inspect supports. Field access is
defensive (``getattr`` with defaults) because the schema gains optional
fields between Inspect releases.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from evalsift.model import Run, Score, ToolCall, ToolResult, Turn

LOG_SUFFIXES = (".eval", ".json")
# Index files Inspect may write alongside logs; not logs themselves.
_NON_LOG_NAMES = {"logs.json", "listing.json"}
# Model calls made by scorers are not agent turns.
_NON_AGENT_ROLES = {"grader", "scorer"}


class NotAnInspectLog(ValueError):
    """A file could not be read as an Inspect log."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path} is not a readable Inspect log ({reason})")
        self.path = path
        self.reason = reason


SkipHandler = Callable[[NotAnInspectLog], None]


def discover(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated list of logs."""
    return sorted(_discover_with_origin(paths))


def _discover_with_origin(paths: Iterable[str | Path]) -> dict[Path, bool]:
    """Map each candidate log to whether the user named it explicitly.

    A file named on the command line is the user's claim that it is a log, so
    failing to read it is an error. A file found by scanning a folder is only
    a guess based on its extension, so failing to read it means skip it.
    """
    found: dict[Path, bool] = {}
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for suffix in LOG_SUFFIXES:
                for f in p.rglob(f"*{suffix}"):
                    if f.name not in _NON_LOG_NAMES:
                        found.setdefault(f, False)
        elif p.is_file():
            found[p] = True  # explicit wins over found-by-scan
        else:
            raise FileNotFoundError(f"No such log file or directory: {p}")
    return found


def load_runs(paths: Iterable[str | Path], on_skip: SkipHandler | None = None) -> list[Run]:
    """Load every sample-epoch from the given logs as normalised runs.

    Files found by scanning a folder that turn out not to be Inspect logs are
    skipped and passed to ``on_skip`` (default: a Python warning). A file
    named explicitly that cannot be read raises ``NotAnInspectLog``.
    """
    runs: list[Run] = []
    for path, explicit in sorted(_discover_with_origin(paths).items()):
        try:
            runs.extend(read_log(path))
        except NotAnInspectLog as exc:
            if explicit:
                raise
            (on_skip or _warn_skip)(exc)
    return runs


def _warn_skip(exc: NotAnInspectLog) -> None:
    warnings.warn(f"skipped {exc}", stacklevel=3)


def _read_inspect_log(path: Path) -> Any:
    """Read one file with Inspect, translating any failure to NotAnInspectLog.

    Only Inspect's own read is wrapped. Errors in evalsift's conversion code
    after a successful read are bugs and must not be hidden as "skipped".
    """
    # Imported lazily so `evalsift --help` and the pure-Python modules stay fast.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from inspect_ai.log import read_eval_log

        try:
            log = read_eval_log(str(path))
        except Exception as exc:  # Inspect raises ValueError, pydantic errors, OSError
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
            raise NotAnInspectLog(path, f"{type(exc).__name__}: {first}"[:200]) from exc
    if getattr(log, "eval", None) is None:
        raise NotAnInspectLog(path, "no eval header")
    return log


def read_log(path: Path) -> Iterator[Run]:
    log = _read_inspect_log(path)

    task = str(getattr(log.eval, "task", "") or "")
    model = str(getattr(log.eval, "model", "") or "")
    eval_id = str(getattr(log.eval, "eval_id", "") or getattr(log.eval, "task_id", ""))
    if log.status == "error" and not log.samples:
        message = getattr(log.error, "message", "unknown error")
        yield Run(
            run_id=f"{eval_id}:<eval>",
            source=str(path),
            task=task,
            model=model,
            sample_id="<eval>",
            epoch=0,
            turns=[],
            error=f"eval failed before producing samples: {message}",
        )
        return

    for sample in log.samples or []:
        yield _sample_to_run(sample, source=str(path), task=task, model=model, eval_id=eval_id)


def _sample_to_run(sample: Any, *, source: str, task: str, model: str, eval_id: str) -> Run:
    turns = _turns_from_messages(getattr(sample, "messages", []) or [])
    _attach_model_events(turns, getattr(sample, "events", []) or [])

    output = getattr(sample, "output", None)
    final_stop = getattr(output, "stop_reason", None) if output is not None else None
    if turns and turns[-1].stop_reason is None:
        turns[-1].stop_reason = final_stop

    in_tok = out_tok = reason_tok = 0
    for usage in (getattr(sample, "model_usage", None) or {}).values():
        in_tok += int(getattr(usage, "input_tokens", 0) or 0)
        out_tok += int(getattr(usage, "output_tokens", 0) or 0)
        reason_tok += int(getattr(usage, "reasoning_tokens", 0) or 0)

    error = getattr(sample, "error", None)
    limit = getattr(sample, "limit", None)
    sample_id = str(getattr(sample, "id", ""))
    epoch = int(getattr(sample, "epoch", 1) or 1)

    return Run(
        run_id=f"{eval_id}:{sample_id}:{epoch}",
        source=source,
        task=task,
        model=model,
        sample_id=sample_id,
        epoch=epoch,
        turns=turns,
        scores=[
            Score(
                name=name,
                value=getattr(s, "value", None),
                answer=getattr(s, "answer", None),
                explanation=getattr(s, "explanation", None),
            )
            for name, s in (getattr(sample, "scores", None) or {}).items()
        ],
        input_text=_input_text(getattr(sample, "input", "")),
        target=_target_text(getattr(sample, "target", "")),
        error=getattr(error, "message", None) if error is not None else None,
        limit=_describe_limit(limit),
        final_stop_reason=final_stop,
        total_time_s=getattr(sample, "total_time", None),
        working_time_s=getattr(sample, "working_time", None),
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=reason_tok,
        metadata=dict(getattr(sample, "metadata", None) or {}),
    )


def _turns_from_messages(messages: list[Any]) -> list[Turn]:
    turns: list[Turn] = []
    call_owner: dict[str, Turn] = {}
    for message in messages:
        role = getattr(message, "role", None)
        if role == "assistant":
            reasoning, redacted = _reasoning(getattr(message, "content", None))
            calls = [
                ToolCall(
                    id=str(getattr(c, "id", "")),
                    function=str(getattr(c, "function", "")),
                    arguments=dict(getattr(c, "arguments", None) or {}),
                    parse_error=getattr(c, "parse_error", None),
                )
                for c in (getattr(message, "tool_calls", None) or [])
            ]
            turn = Turn(
                index=len(turns),
                text=str(getattr(message, "text", "") or ""),
                reasoning=reasoning,
                reasoning_redacted=redacted,
                tool_calls=calls,
            )
            turns.append(turn)
            for c in calls:
                call_owner[c.id] = turn
        elif role == "tool":
            err = getattr(message, "error", None)
            call_id = getattr(message, "tool_call_id", None)
            result = ToolResult(
                call_id=call_id,
                function=getattr(message, "function", None),
                content=str(getattr(message, "text", "") or ""),
                error_type=getattr(err, "type", None) if err is not None else None,
                error_message=getattr(err, "message", None) if err is not None else None,
            )
            owner = call_owner.get(call_id or "") or (turns[-1] if turns else None)
            if owner is not None:
                owner.tool_results.append(result)
    return turns


def _reasoning(content: Any) -> tuple[list[str], bool]:
    if not isinstance(content, list):
        return [], False
    blocks: list[str] = []
    redacted = False
    for part in content:
        if getattr(part, "type", None) != "reasoning":
            continue
        if getattr(part, "redacted", False):
            redacted = True
            summary = getattr(part, "summary", None)
            if summary:
                blocks.append(str(summary))
            continue
        text = getattr(part, "reasoning", "") or getattr(part, "summary", "") or ""
        if text:
            blocks.append(str(text))
    return blocks, redacted


def _attach_model_events(turns: list[Turn], events: list[Any]) -> None:
    """Copy per-call latency, stop reason and errors from model events.

    Each agent generate() call yields one model event and one assistant
    message, in order. If the counts disagree (for example, a solver that
    edits the message list) we attach nothing rather than guess.
    """
    model_events = [
        e
        for e in events
        if getattr(e, "event", None) == "model" and getattr(e, "role", None) not in _NON_AGENT_ROLES
    ]
    if len(model_events) != len(turns):
        return
    for turn, event in zip(turns, model_events, strict=True):
        turn.latency_s = getattr(event, "working_time", None)
        out = getattr(event, "output", None)
        turn.stop_reason = getattr(out, "stop_reason", None) if out is not None else None
        err = getattr(event, "error", None)
        turn.model_error = str(err) if err else None


def _describe_limit(limit: Any) -> str | None:
    if limit is None:
        return None
    kind = getattr(limit, "type", None) or "limit"
    value = getattr(limit, "limit", None)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{kind}={value}" if value is not None else str(kind)


def _input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(str(getattr(m, "text", m)) for m in value)
    return str(value)


def _target_text(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return str(value or "")
