"""Reasoning-trace export: readable Markdown per run, and machine-readable JSONL.

Markdown traces put findings inline next to the turn they refer to, so a
reviewer reading a transcript sees "tool_error here" at the right place
instead of cross-referencing a table.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evalsift.model import Finding, Run, Severity
from evalsift.redact import Redactor

_ICON = {Severity.ERROR: "❌", Severity.WARNING: "⚠️", Severity.INFO: "ℹ️"}


def trace_markdown(
    run: Run,
    findings: Sequence[Finding] = (),
    redactor: Redactor | None = None,
    max_tool_output: int = 2000,
) -> str:
    red = redactor or Redactor()
    by_turn: dict[int | None, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.run_id == run.run_id:
            by_turn[f.turn].append(f)

    lines = [
        f"# {run.task} · sample {run.sample_id} · epoch {run.epoch}",
        "",
        f"- **Model:** `{run.model}`",
        f"- **Run ID:** `{run.run_id}`",
        f"- **Turns:** {len(run.turns)} · **Tool calls:** {run.n_tool_calls}",
    ]
    if run.total_time_s is not None:
        lines.append(
            f"- **Time:** {run.total_time_s:.2f}s total, {run.working_time_s or 0:.2f}s working"
        )
    if run.scores:
        scored = ", ".join(f"{s.name}={s.value!r}" for s in run.scores)
        lines.append(f"- **Scores:** {red.redact(scored)}")
    if run.limit:
        lines.append(f"- **Limit:** {run.limit}")
    if run.error:
        lines.append(f"- **Error:** {red.redact(run.error.strip().splitlines()[0])}")

    run_level = by_turn.get(None, [])
    if run_level:
        lines += ["", "## Run-level findings", ""]
        lines += [_finding_line(f) for f in run_level]

    lines += ["", "## Input", "", _fence(red.redact(run.input_text))]
    if run.target:
        lines += ["", f"**Target:** {red.redact(run.target)}"]

    for t in run.turns:
        header = f"## Turn {t.index}"
        meta = [
            m
            for m in (
                f"stop={t.stop_reason}" if t.stop_reason else "",
                f"{t.latency_s:.2f}s" if t.latency_s is not None else "",
            )
            if m
        ]
        if meta:
            header += f" ({', '.join(meta)})"
        lines += ["", header, ""]
        lines += [_finding_line(f) for f in by_turn.get(t.index, [])]
        if t.reasoning or t.reasoning_redacted:
            lines += ["", "<details><summary>Reasoning</summary>", ""]
            if t.reasoning_redacted and not t.reasoning:
                lines.append("_Reasoning was redacted by the provider._")
            for block in t.reasoning:
                lines += [_fence(red.redact(block)), ""]
            lines.append("</details>")
        if t.text.strip():
            lines += ["", red.redact(t.text.strip())]
        results = {r.call_id: r for r in t.tool_results}
        for c in t.tool_calls:
            args = red.redact(json.dumps(c.arguments, indent=2, default=str))
            lines += ["", f"**Tool call** `{c.function}`", _fence(args, "json")]
            r = results.get(c.id)
            if r is not None:
                label = f"**Result** ({r.error_type})" if r.error_type else "**Result**"
                body = r.error_message if r.error_message and not r.content else r.content
                lines += [label, _fence(red.redact(_clip(body, max_tool_output)))]
    lines.append("")
    return "\n".join(lines)


def write_traces(
    runs: Iterable[Run],
    findings: Sequence[Finding],
    out_dir: Path,
    redactor: Redactor | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    used: set[str] = set()
    for run in runs:
        name = _safe_name(f"{run.task}__{run.model}__{run.sample_id}__e{run.epoch}")
        while name in used:
            name += "_"
        used.add(name)
        path = out_dir / f"{name}.md"
        path.write_text(trace_markdown(run, findings, redactor), encoding="utf-8")
        written.append(path)
    return written


def write_jsonl(
    runs: Iterable[Run],
    findings: Sequence[Finding],
    path: Path,
    redactor: Redactor | None = None,
) -> None:
    red = redactor or Redactor()
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in findings:
        by_run[f.run_id].append({**asdict(f), "severity": f.severity.value})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for run in runs:
            record = asdict(run)
            record["findings"] = by_run.get(run.run_id, [])
            # Redact the serialised text so every nested string is covered.
            fh.write(red.redact(json.dumps(record, default=str, ensure_ascii=False)) + "\n")


def _finding_line(f: Finding) -> str:
    return f"> {_ICON[f.severity]} **{f.check}** ({f.severity.value}): {f.message}"


def _fence(text: str, lang: str = "") -> str:
    # Use a fence longer than any backtick run inside, so content can't escape.
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters truncated]"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:150]
