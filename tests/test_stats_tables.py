from __future__ import annotations

import math

import pytest

from evalsift.model import Finding, Run
from evalsift.stats import wilson
from evalsift.tables import findings_table, runs_table, summary_table, timing_table, to_markdown


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        # Reference values from statsmodels.proportion_confint(method="wilson").
        (5, 10, 0.236593, 0.763407),
        (0, 10, 0.0, 0.277533),
        (10, 10, 0.722467, 1.0),
        (1, 20, 0.008881, 0.236131),
        (42, 100, 0.327984, 0.517935),
    ],
)
def test_wilson_reference_values(k: int, n: int, low: float, high: float) -> None:
    p = wilson(k, n)
    assert p.low == pytest.approx(low, abs=1e-5)
    assert p.high == pytest.approx(high, abs=1e-5)
    assert p.estimate == k / n


def test_wilson_bounds_and_width() -> None:
    narrow, wide = wilson(5, 10, 0.80), wilson(5, 10, 0.99)
    assert wide.low < narrow.low and wide.high > narrow.high
    for k in range(0, 31):
        p = wilson(k, 30)
        assert 0.0 <= p.low <= p.estimate <= p.high <= 1.0


def test_wilson_empty_and_invalid() -> None:
    assert math.isnan(wilson(0, 0).estimate)
    assert str(wilson(0, 0)) == "n/a (n=0)"
    with pytest.raises(ValueError, match="successes"):
        wilson(3, 2)
    with pytest.raises(ValueError, match="confidence"):
        wilson(1, 2, confidence=1.0)


def test_summary_excludes_invalid_from_rates(runs: list[Run], findings: list[Finding]) -> None:
    rows = {r["task"]: r for r in summary_table(runs, findings)}
    crash = rows["crash"]
    assert (crash["runs"], crash["valid"], crash["invalid_rate"]) == (1, 0, "1.000")
    assert crash["match:n"] == 0 and crash["match:rate"] == ""
    clean = rows["clean"]
    assert (clean["match:rate"], clean["match:n"]) == ("1.000", 1)
    assert clean["match:ci"].startswith("[0.207")


def test_summary_pooled_by_model(runs: list[Run], findings: list[Finding]) -> None:
    (row,) = summary_table(runs, findings, group_by=["model"])
    assert row["runs"] == len(runs)
    assert row["valid"] == len(runs) - 1
    # clean, tool_error, loop, leak, reasoning are correct; crash is excluded.
    assert row["match:n"] == len(runs) - 1


def test_bad_group_key(runs: list[Run], findings: list[Finding]) -> None:
    with pytest.raises(ValueError, match="cannot group"):
        summary_table(runs, findings, group_by=["colour"])


def test_runs_and_timing_tables(runs: list[Run], findings: list[Finding]) -> None:
    rows = {r["sample_id"]: r for r in runs_table(runs, findings)}
    assert rows["crash"]["valid"] is False
    assert rows["tool_error"]["warnings"] == 1
    assert rows["clean"]["score:match"] == "C"
    timing = {r["run_id"]: r for r in timing_table(runs)}
    clean = next(r for r in runs if r.sample_id == "clean")
    assert timing[clean.run_id]["slowest_turn"] in (0, 1)


def test_markdown_escapes_pipes(findings: list[Finding]) -> None:
    md = to_markdown([{"a": "x|y", "b": "line\nbreak"}])
    assert "x\\|y" in md and "line break" in md
    assert to_markdown([]) == "_No rows._\n"
    rows = findings_table(findings)
    for run_id in {r["run_id"] for r in rows}:
        sev = [r["severity"] for r in rows if r["run_id"] == run_id]
        assert sev == sorted(sev, key=["error", "warning", "info"].index)
