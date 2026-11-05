"""Command-line interface.

    evalsift check   LOGS...            flag invalid runs and failed turns
    evalsift summary LOGS...            rates with Wilson intervals, per group
    evalsift export  LOGS... -o DIR     Markdown traces, tables and JSONL
    evalsift bundle  LOGS... -o FILE    redacted review zip for a reviewer
    evalsift scan    PATHS...           look for secrets before you share files
    evalsift checks                     list available checks

Exit codes: 0 ok, 1 findings at or above --fail-on (check) or secrets found
(scan), 2 usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from evalsift import __version__
from evalsift.adapters import NotAnInspectLog, load_runs
from evalsift.bundle import BundleOptions, SecretLeakError, build_bundle
from evalsift.checks import REGISTRY, CheckConfig, invalid_run_ids, run_checks
from evalsift.export import write_jsonl, write_traces
from evalsift.model import Finding, Run, Severity
from evalsift.redact import Redactor, scan_paths
from evalsift.tables import (
    findings_table,
    runs_table,
    summary_table,
    timing_table,
    to_csv,
    to_markdown,
    write_csv,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        code: int = args.func(args)
        return code
    except (FileNotFoundError, ValueError, SecretLeakError) as exc:
        print(f"evalsift: error: {exc}", file=sys.stderr)
        return 2


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    runs, findings = _load(args)
    if args.format == "json":
        print(json.dumps(findings_table(findings), indent=2))
    elif args.format == "csv":
        sys.stdout.write(to_csv(findings_table(findings)))
    else:
        _print_findings(runs, findings)

    threshold = {"error": Severity.ERROR, "warning": Severity.WARNING}.get(args.fail_on)
    if threshold is not None and any(f.severity.rank >= threshold.rank for f in findings):
        return 1
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    runs, findings = _load(args)
    group_by = _split(args.by)
    rows = summary_table(runs, findings, group_by, args.confidence)
    print(to_markdown(rows))
    if args.out:
        out = Path(args.out)
        write_csv(rows, out / "summary.csv")
        write_csv(runs_table(runs, findings), out / "runs.csv")
        write_csv(timing_table(runs), out / "timing.csv")
        write_csv(findings_table(findings), out / "findings.csv")
        print(f"Wrote summary, runs, timing and findings tables to {out}/", file=sys.stderr)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    runs, findings = _load(args)
    red = _redactor(args)
    out = Path(args.out)
    invalid = invalid_run_ids(findings)
    flagged = {f.run_id for f in findings}
    selected = {
        "all": runs,
        "invalid": [r for r in runs if r.run_id in invalid],
        "flagged": [r for r in runs if r.run_id in flagged],
    }[args.only]

    paths = write_traces(selected, findings, out / "traces", red)
    write_csv(runs_table(runs, findings), out / "runs.csv")
    write_csv(timing_table(runs), out / "timing.csv")
    write_csv(findings_table(findings), out / "findings.csv")
    if args.jsonl:
        write_jsonl(selected, findings, out / "runs.jsonl", red)
    print(f"Wrote {len(paths)} trace(s) and tables to {out}/", file=sys.stderr)
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    runs, findings = _load(args)
    opts = BundleOptions(
        include_invalid=not args.no_invalid,
        include_warned=not args.no_warned,
        sample_valid=args.sample,
        seed=args.seed,
        run_ids=args.run_id or [],
        title=args.title,
        notes=args.notes,
    )
    manifest = build_bundle(runs, findings, Path(args.out), opts, _redactor(args), _split(args.by))
    c = manifest["counts"]
    print(
        f"Wrote {args.out}: {c['traces']} trace(s) from {c['runs']} runs ({c['invalid']} invalid).",
        file=sys.stderr,
    )
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    hits = scan_paths([Path(p) for p in args.paths], _redactor(args))
    for h in hits:
        print(f"{h.path}:{h.line}: {h.name} ({h.preview})")
    if hits:
        print(f"{len(hits)} possible secret(s) found.", file=sys.stderr)
        return 1
    print("No secrets found.", file=sys.stderr)
    return 0


def cmd_list_checks(_: argparse.Namespace) -> int:
    width = max(len(n) for n in REGISTRY)
    for check in REGISTRY.values():
        print(f"{check.name:<{width}}  {check.severity.value:<7}  {check.description}")
    return 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _load(args: argparse.Namespace) -> tuple[list[Run], list[Finding]]:
    skipped: list[NotAnInspectLog] = []

    def report_skip(exc: NotAnInspectLog) -> None:
        skipped.append(exc)
        print(f"evalsift: skipped {exc}", file=sys.stderr)

    runs = load_runs(args.logs, on_skip=report_skip)
    if not runs:
        detail = f" ({len(skipped)} file(s) skipped as unreadable)" if skipped else ""
        raise ValueError(f"no runs found in the given logs{detail}")
    overrides: dict[str, Severity] = {}
    for item in args.severity or []:
        name, _, level = item.partition("=")
        try:
            overrides[name] = Severity(level)
        except ValueError:
            raise ValueError(
                f"bad --severity {item!r}; expected CHECK=error|warning|info"
            ) from None
    config = CheckConfig(
        repeat_threshold=args.repeat_threshold,
        severity_overrides=overrides,
        disabled=set(args.disable or []),
    )
    return runs, run_checks(runs, config)


def _redactor(args: argparse.Namespace) -> Redactor:
    return Redactor(extra_patterns=args.redact_pattern or [], include_env=args.redact_env)


def _print_findings(runs: list[Run], findings: list[Finding]) -> None:
    invalid = invalid_run_ids(findings)
    rows = findings_table(findings)
    if rows:
        print(to_markdown(rows, ["run_id", "turn", "severity", "check", "message"]))
    n_warn = sum(f.severity is Severity.WARNING for f in findings)
    print(
        f"{len(runs)} runs · {len(invalid)} invalid · "
        f"{len({f.run_id for f in findings if f.severity is Severity.WARNING})} with warnings "
        f"({n_warn} warning findings)"
    )


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalsift",
        description="Find invalid runs and failed turns in agent evaluation logs.",
    )
    p.add_argument("--version", action="version", version=f"evalsift {__version__}")
    sub = p.add_subparsers(metavar="COMMAND")

    checks = argparse.ArgumentParser(add_help=False)
    checks.add_argument("logs", nargs="+", help="Inspect log files (.eval/.json) or directories")
    checks.add_argument(
        "--severity",
        action="append",
        metavar="CHECK=LEVEL",
        help="override a check's severity, e.g. limit_hit=error (repeatable)",
    )
    checks.add_argument(
        "--disable", action="append", metavar="CHECK", help="skip a check (repeatable)"
    )
    checks.add_argument(
        "--repeat-threshold",
        type=int,
        default=3,
        metavar="N",
        help="identical consecutive tool calls that count as a loop (default 3)",
    )

    redact = argparse.ArgumentParser(add_help=False)
    redact.add_argument(
        "--redact-env",
        action="store_true",
        help="also mask current values of env vars named like *KEY*, *TOKEN*, *SECRET*",
    )
    redact.add_argument(
        "--redact-pattern",
        action="append",
        metavar="REGEX",
        help="extra secret pattern (repeatable)",
    )

    c = sub.add_parser("check", parents=[checks], help="flag invalid runs and failed turns")
    c.add_argument("--format", choices=["text", "json", "csv"], default="text")
    c.add_argument(
        "--fail-on",
        choices=["error", "warning", "never"],
        default="error",
        help="exit 1 if any finding is at least this severe (default: error)",
    )
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("summary", parents=[checks], help="rates with Wilson intervals")
    s.add_argument(
        "--by", default="task,model", help="group keys: task, model, epoch, metadata.<field>"
    )
    s.add_argument("--confidence", type=float, default=0.95)
    s.add_argument("--out", metavar="DIR", help="also write CSV tables here")
    s.set_defaults(func=cmd_summary)

    e = sub.add_parser("export", parents=[checks, redact], help="write traces and tables")
    e.add_argument("-o", "--out", required=True, metavar="DIR")
    e.add_argument("--only", choices=["all", "invalid", "flagged"], default="all")
    e.add_argument("--jsonl", action="store_true", help="also write normalised runs as JSONL")
    e.set_defaults(func=cmd_export)

    b = sub.add_parser("bundle", parents=[checks, redact], help="build a redacted review zip")
    b.add_argument("-o", "--out", required=True, metavar="FILE.zip")
    b.add_argument("--sample", type=int, default=5, help="random clean runs to include (default 5)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--run-id", action="append", help="always include this run (repeatable)")
    b.add_argument("--no-invalid", action="store_true", help="don't auto-include invalid runs")
    b.add_argument("--no-warned", action="store_true", help="don't auto-include runs with warnings")
    b.add_argument("--by", default="task,model")
    b.add_argument("--title", default="Evaluation review bundle")
    b.add_argument("--notes", default="", help="text to put at the top of the bundle README")
    b.set_defaults(func=cmd_bundle)

    sc = sub.add_parser("scan", parents=[redact], help="scan files for secrets")
    sc.add_argument("paths", nargs="+")
    sc.set_defaults(func=cmd_scan)

    lc = sub.add_parser("checks", help="list available checks")
    lc.set_defaults(func=cmd_list_checks)
    return p


if __name__ == "__main__":
    raise SystemExit(main())
