"""Runs, timing, findings and summary tables.

Every table is a list of flat dicts so it can be written as CSV, rendered as
Markdown, or loaded into pandas without evalsift needing pandas itself.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from evalsift.checks import invalid_run_ids
from evalsift.model import Finding, Run, Severity
from evalsift.stats import wilson

Row = dict[str, Any]
GROUP_KEYS = ("task", "model")


def runs_table(runs: Sequence[Run], findings: Sequence[Finding]) -> list[Row]:
    by_run: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_run[f.run_id].append(f)
    invalid = invalid_run_ids(findings)

    score_names = sorted({s.name for r in runs for s in r.scores})
    rows: list[Row] = []
    for r in runs:
        fs = by_run.get(r.run_id, [])
        row: Row = {
            "run_id": r.run_id,
            "task": r.task,
            "model": r.model,
            "sample_id": r.sample_id,
            "epoch": r.epoch,
            "valid": r.run_id not in invalid,
            "turns": len(r.turns),
            "tool_calls": r.n_tool_calls,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "total_time_s": _round(r.total_time_s),
            "working_time_s": _round(r.working_time_s),
            "stop_reason": r.final_stop_reason or "",
            "limit": r.limit or "",
            "errors": sum(f.severity is Severity.ERROR for f in fs),
            "warnings": sum(f.severity is Severity.WARNING for f in fs),
            "checks": ";".join(sorted({f.check for f in fs})),
            "source": r.source,
        }
        scores = {s.name: s.value for s in r.scores}
        for name in score_names:
            row[f"score:{name}"] = scores.get(name, "")
        rows.append(row)
    return rows


def timing_table(runs: Sequence[Run]) -> list[Row]:
    """Per-run latency profile, from per-turn model-call working time."""
    rows: list[Row] = []
    for r in runs:
        lat = [t.latency_s for t in r.turns if t.latency_s is not None]
        rows.append(
            {
                "run_id": r.run_id,
                "task": r.task,
                "model": r.model,
                "turns": len(r.turns),
                "total_time_s": _round(r.total_time_s),
                "working_time_s": _round(r.working_time_s),
                "model_time_s": _round(sum(lat)) if lat else "",
                "turn_latency_mean_s": _round(mean(lat)) if lat else "",
                "turn_latency_median_s": _round(median(lat)) if lat else "",
                "turn_latency_max_s": _round(max(lat)) if lat else "",
                "slowest_turn": (
                    max(
                        (t for t in r.turns if t.latency_s is not None),
                        key=lambda t: t.latency_s or 0.0,
                    ).index
                    if lat
                    else ""
                ),
            }
        )
    return rows


def findings_table(findings: Iterable[Finding]) -> list[Row]:
    return [
        {
            "run_id": f.run_id,
            "turn": "" if f.turn is None else f.turn,
            "severity": f.severity.value,
            "check": f.check,
            "message": f.message,
        }
        for f in sorted(findings, key=lambda f: (f.run_id, -f.severity.rank, f.turn or 0))
    ]


def summary_table(
    runs: Sequence[Run],
    findings: Sequence[Finding],
    group_by: Sequence[str] = GROUP_KEYS,
    confidence: float = 0.95,
) -> list[Row]:
    """One row per group: validity, then score rates among valid runs only.

    Invalid runs are excluded from score rates but counted, so the reader can
    see how much data was dropped and judge whether that biases the rate.
    """
    invalid = invalid_run_ids(findings)
    warned = {f.run_id for f in findings if f.severity is Severity.WARNING}

    groups: dict[tuple[str, ...], list[Run]] = defaultdict(list)
    for r in runs:
        groups[tuple(_group_value(r, k) for k in group_by)].append(r)

    score_names = sorted({s.name for r in runs for s in r.scores})
    rows: list[Row] = []
    for key, members in sorted(groups.items()):
        valid = [r for r in members if r.run_id not in invalid]
        inv = wilson(len(members) - len(valid), len(members), confidence)
        row: Row = dict(zip(group_by, key, strict=True))
        row.update(
            {
                "runs": len(members),
                "valid": len(valid),
                "invalid_rate": _fmt(inv.estimate),
                "invalid_ci": _ci(inv.low, inv.high),
                "runs_with_warnings": sum(r.run_id in warned for r in members),
            }
        )
        for name in score_names:
            binary = [
                b
                for r in valid
                for s in r.scores
                if s.name == name and (b := s.as_binary()) is not None
            ]
            if binary:
                p = wilson(sum(binary), len(binary), confidence)
                row[f"{name}:rate"] = _fmt(p.estimate)
                row[f"{name}:ci"] = _ci(p.low, p.high)
                row[f"{name}:n"] = p.n
            else:
                numeric = [
                    float(s.value)
                    for r in valid
                    for s in r.scores
                    if s.name == name
                    and isinstance(s.value, (int, float))
                    and not isinstance(s.value, bool)
                ]
                row[f"{name}:rate"] = ""
                row[f"{name}:ci"] = ""
                row[f"{name}:n"] = len(numeric)
                if numeric:
                    row[f"{name}:mean"] = _round(mean(numeric))
        rows.append(row)
    return rows


def write_csv(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _columns(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_csv(rows: Sequence[Row]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_columns(rows), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def to_markdown(rows: Sequence[Row], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "_No rows._\n"
    cols = list(columns) if columns else _columns(rows)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_cell(row.get(c, "")) for c in cols) + " |" for row in rows]
    return "\n".join([head, sep, *body]) + "\n"


def _columns(rows: Sequence[Row]) -> list[str]:
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    return cols


def _group_value(run: Run, key: str) -> str:
    if key in ("task", "model", "sample_id"):
        return str(getattr(run, key))
    if key == "epoch":
        return str(run.epoch)
    if key.startswith("metadata."):
        return str(run.metadata.get(key.removeprefix("metadata."), ""))
    raise ValueError(
        f"cannot group by {key!r}; use task, model, sample_id, epoch or metadata.<field>"
    )


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _round(value: float | None, places: int = 3) -> float | str:
    return "" if value is None else round(value, places)


def _fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.3f}"


def _ci(low: float, high: float) -> str:
    return "" if math.isnan(low) else f"[{low:.3f}, {high:.3f}]"
