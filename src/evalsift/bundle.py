"""Review bundles: one redacted zip a reviewer can open without any tooling.

A bundle holds the summary, findings and runs tables, Markdown traces for a
chosen subset of runs (every invalid run plus a seeded random sample of valid
ones by default), and a manifest that pins the source logs by SHA-256 and
records exactly how the bundle was made. Every text file is redacted before it
is written, and the redacted contents are re-scanned; if anything still looks
like a secret, nothing is written and an error is raised.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalsift import __version__
from evalsift.checks import invalid_run_ids
from evalsift.export import trace_markdown
from evalsift.model import Finding, Run, Severity
from evalsift.redact import Redactor
from evalsift.tables import (
    Row,
    findings_table,
    runs_table,
    summary_table,
    timing_table,
    to_csv,
    to_markdown,
)


class SecretLeakError(RuntimeError):
    pass


@dataclass
class BundleOptions:
    include_invalid: bool = True
    include_warned: bool = True
    sample_valid: int = 5
    seed: int = 0
    run_ids: list[str] = field(default_factory=list)
    title: str = "Evaluation review bundle"
    notes: str = ""


def select_runs(runs: Sequence[Run], findings: Sequence[Finding], opts: BundleOptions) -> list[Run]:
    invalid = invalid_run_ids(findings)
    warned = {f.run_id for f in findings if f.severity is Severity.WARNING}
    chosen: dict[str, Run] = {}
    explicit = set(opts.run_ids)
    missing = explicit - {r.run_id for r in runs}
    if missing:
        raise ValueError(f"run id(s) not found: {', '.join(sorted(missing))}")
    for r in runs:
        if (
            r.run_id in explicit
            or (opts.include_invalid and r.run_id in invalid)
            or (opts.include_warned and r.run_id in warned)
        ):
            chosen[r.run_id] = r
    clean = [r for r in runs if r.run_id not in invalid and r.run_id not in chosen]
    rng = random.Random(opts.seed)
    for r in rng.sample(clean, min(opts.sample_valid, len(clean))):
        chosen[r.run_id] = r
    order = {r.run_id: i for i, r in enumerate(runs)}
    return sorted(chosen.values(), key=lambda r: order[r.run_id])


def build_bundle(
    runs: Sequence[Run],
    findings: Sequence[Finding],
    out_path: Path,
    opts: BundleOptions | None = None,
    redactor: Redactor | None = None,
    group_by: Sequence[str] = ("task", "model"),
) -> dict[str, Any]:
    opts = opts or BundleOptions()
    red = redactor or Redactor()
    selected = select_runs(runs, findings, opts)
    invalid = invalid_run_ids(findings)

    summary = summary_table(runs, findings, group_by)
    files: dict[str, str] = {
        "summary.csv": to_csv(summary),
        "runs.csv": to_csv(runs_table(runs, findings)),
        "timing.csv": to_csv(timing_table(runs)),
        "findings.csv": to_csv(findings_table(findings)),
    }
    trace_names: list[str] = []
    for i, run in enumerate(selected):
        prefix = "INVALID_" if run.run_id in invalid else ""
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{run.sample_id}_e{run.epoch}")
        name = f"traces/{i:03d}_{prefix}{stem}.md"
        files[name] = trace_markdown(run, findings, red)
        trace_names.append(name)

    sources = sorted({r.source for r in runs})
    manifest: dict[str, Any] = {
        "tool": "evalsift",
        "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "options": {
            "include_invalid": opts.include_invalid,
            "include_warned": opts.include_warned,
            "sample_valid": opts.sample_valid,
            "seed": opts.seed,
            "run_ids": opts.run_ids,
            "group_by": list(group_by),
        },
        "counts": {
            "runs": len(runs),
            "invalid": len(invalid & {r.run_id for r in runs}),
            "traces": len(selected),
        },
        "sources": [{"path": Path(s).name, "sha256": _sha256(Path(s))} for s in sources],
        "traces": [
            {"file": n, "run_id": r.run_id, "invalid": r.run_id in invalid}
            for n, r in zip(trace_names, selected, strict=True)
        ],
    }
    files["manifest.json"] = json.dumps(manifest, indent=2)
    files["README.md"] = _readme(opts, summary, manifest)

    # Redact everything once more (tables may carry error text), then verify.
    files = {name: red.redact(body) for name, body in files.items()}
    leaks = [(n, m.name, m.line) for n, body in files.items() for m in red.find(body)]
    if leaks:
        raise SecretLeakError(f"refusing to write bundle; possible secrets remain: {leaks[:5]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, body in sorted(files.items()):
            zf.writestr(name, body)
    return manifest


def _readme(opts: BundleOptions, summary: list[Row], manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    parts = [
        f"# {opts.title}",
        "",
        f"Generated by evalsift {manifest['version']} on {manifest['created_utc']}.",
        "",
    ]
    if opts.notes:
        parts += [opts.notes, ""]
    parts += [
        "## Summary",
        "",
        f"{counts['runs']} runs, of which {counts['invalid']} are invalid and excluded "
        "from score rates. Intervals are 95% Wilson score intervals.",
        "",
        to_markdown(summary),
        "## What's in this bundle",
        "",
        f"- `traces/`: {counts['traces']} transcripts. Invalid runs are prefixed `INVALID_`; "
        f"runs with warnings are included; plus up to {opts.sample_valid} randomly sampled "
        f"clean runs (seed {opts.seed}).",
        "- `summary.csv`, `runs.csv`, `timing.csv`, `findings.csv`: full tables for all runs.",
        "- `manifest.json`: source-log hashes and the options used, for reproducibility.",
        "",
        "Secrets matching known key formats were redacted as `[REDACTED:<kind>]`.",
        "",
    ]
    return "\n".join(parts)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
