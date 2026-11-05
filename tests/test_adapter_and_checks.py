from __future__ import annotations

from pathlib import Path

import pytest

from evalsift.adapters import NotAnInspectLog, discover, load_runs
from evalsift.checks import REGISTRY, CheckConfig, invalid_run_ids, run_checks
from evalsift.model import Finding, Run, Score, Severity, Turn

# What each scripted scenario must produce. Exact sets, so a new false
# positive fails the test as surely as a missed detection.
EXPECTED: dict[str, set[str]] = {
    "clean": set(),
    "wrong": set(),
    "tool_error": {"tool_error"},
    "loop": {"repeated_tool_call"},
    "limit": {"limit_hit", "repeated_tool_call"},
    "crash": {"sample_error"},
    "empty": {"empty_final"},
    "leak": set(),
    "truncated": {"truncated_output"},
    "reasoning": set(),
}


def test_every_scenario_loaded(by_sample: dict[str, Run]) -> None:
    assert set(by_sample) == set(EXPECTED)


@pytest.mark.parametrize(("sample", "expected"), sorted(EXPECTED.items()))
def test_scenario_findings(
    sample: str, expected: set[str], by_sample: dict[str, Run], findings: list[Finding]
) -> None:
    run_id = by_sample[sample].run_id
    assert {f.check for f in findings if f.run_id == run_id} == expected


def test_only_crash_is_invalid(by_sample: dict[str, Run], findings: list[Finding]) -> None:
    assert invalid_run_ids(findings) == {by_sample["crash"].run_id}


def test_turns_and_tool_results(by_sample: dict[str, Run]) -> None:
    clean = by_sample["clean"]
    assert [len(t.tool_calls) for t in clean.turns] == [1, 0]
    call, result = clean.turns[0].tool_calls[0], clean.turns[0].tool_results[0]
    assert (call.function, call.arguments) == ("add", {"x": 1, "y": 2})
    assert result.call_id == call.id
    assert result.content == "3"
    assert clean.turns[-1].text == "3"
    assert [s.value for s in clean.scores] == ["C"]


def test_reasoning_extracted(by_sample: dict[str, Run]) -> None:
    turn = by_sample["reasoning"].turns[0]
    assert turn.reasoning == ["1 plus 2 is 3."]
    assert turn.text == "3"


def test_timing_and_usage(by_sample: dict[str, Run]) -> None:
    run = by_sample["clean"]
    assert run.total_time_s is not None and run.total_time_s > 0
    assert all(t.latency_s is not None for t in run.turns)
    assert run.input_tokens > 0 and run.output_tokens > 0


def test_limit_described(by_sample: dict[str, Run]) -> None:
    assert by_sample["limit"].limit == "message=6"


def test_tool_error_details(by_sample: dict[str, Run], findings: list[Finding]) -> None:
    (f,) = [f for f in findings if f.run_id == by_sample["tool_error"].run_id]
    assert f.turn == 0
    assert "record not found" in f.message


def test_severity_override_makes_limit_invalid(runs: list[Run], by_sample: dict[str, Run]) -> None:
    config = CheckConfig(severity_overrides={"limit_hit": Severity.ERROR})
    invalid = invalid_run_ids(run_checks(runs, config))
    assert by_sample["limit"].run_id in invalid


def test_disable_and_threshold(runs: list[Run], by_sample: dict[str, Run]) -> None:
    fs = run_checks(runs, CheckConfig(disabled={"repeated_tool_call"}))
    assert not any(f.check == "repeated_tool_call" for f in fs)
    # The loop scenario repeats 4 times: a threshold of 5 must not fire.
    fs = run_checks([by_sample["loop"]], CheckConfig(repeat_threshold=5))
    assert not any(f.check == "repeated_tool_call" for f in fs)


def test_unknown_check_name_rejected(runs: list[Run]) -> None:
    with pytest.raises(ValueError, match="unknown check"):
        run_checks(runs, CheckConfig(disabled={"nope"}))


def test_registry_has_descriptions() -> None:
    assert all(c.description for c in REGISTRY.values())


def test_discover_skips_index_files(log_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "logs.json").write_text("{}")
    assert discover([tmp_path]) == []
    assert len(discover([log_dir])) == len(EXPECTED)


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_runs([tmp_path / "absent"])


def test_epochs_get_distinct_run_ids(tmp_path: Path) -> None:
    import logfactory

    logfactory.build(tmp_path, {"clean": logfactory.SCENARIOS["clean"]}, epochs=3)
    runs = load_runs([tmp_path])
    assert sorted(r.epoch for r in runs) == [1, 2, 3]
    assert len({r.run_id for r in runs}) == 3


def _run(turns: list[Turn], **kw: object) -> Run:
    return Run(
        run_id="r",
        source="",
        task="t",
        model="m",
        sample_id="s",
        epoch=1,
        turns=turns,
        scores=[Score("s", "C")],
        **kw,
    )  # type: ignore[arg-type]


def test_missing_score_and_no_turns() -> None:
    fs = run_checks(
        [Run(run_id="r", source="", task="t", model="m", sample_id="s", epoch=1, turns=[])]
    )
    assert {f.check for f in fs} == {"missing_score", "no_turns"}
    assert all(f.severity is Severity.ERROR for f in fs)


def test_content_filter_and_empty_turn() -> None:
    turns = [Turn(0, ""), Turn(1, "", stop_reason="content_filter")]
    checks = {f.check for f in run_checks([_run(turns)])}
    assert {"empty_turn", "content_filter", "empty_final"} <= checks


def test_score_binary_mapping() -> None:
    assert [Score("s", v).as_binary() for v in ["C", "I", True, False, 1, 0, 0.5, "P", None]] == [
        1,
        0,
        1,
        0,
        1,
        0,
        None,
        None,
        None,
    ]


# ---------------------------------------------------------------- non-log files

# Each is a way a file with a log extension can fail to be an Inspect log.
JUNK = {
    "config.json": '{"lr": 0.1, "seed": 3}',  # valid JSON, wrong shape
    "empty.json": "{}",
    "broken.json": "not json",
    "zero.json": "",
    "fake.eval": "hello",  # .eval must be a zip
}


def _junk_dir(tmp_path: Path, log_dir: Path) -> Path:
    import shutil

    shutil.copy(next(log_dir.glob("*.eval")), tmp_path / "real.eval")
    (tmp_path / "nested").mkdir()
    for name, body in JUNK.items():
        (tmp_path / "nested" / name).write_text(body)
    return tmp_path


def test_junk_in_folder_is_skipped(tmp_path: Path, log_dir: Path) -> None:
    folder = _junk_dir(tmp_path, log_dir)
    skipped: list[NotAnInspectLog] = []
    runs = load_runs([folder], on_skip=skipped.append)
    assert len(runs) == 1  # the real log still loads
    assert sorted(e.path.name for e in skipped) == sorted(JUNK)
    assert all(e.reason for e in skipped)


def test_default_skip_handler_warns(tmp_path: Path, log_dir: Path) -> None:
    folder = _junk_dir(tmp_path, log_dir)
    with pytest.warns(UserWarning, match="not a readable Inspect log") as record:
        load_runs([folder])
    assert len(record) == len(JUNK)


@pytest.mark.parametrize("name", sorted(JUNK))
def test_named_junk_file_is_an_error(name: str, tmp_path: Path) -> None:
    path = tmp_path / name
    path.write_text(JUNK[name])
    with pytest.raises(NotAnInspectLog, match=name):
        load_runs([path])


def test_explicit_name_wins_over_folder_scan(tmp_path: Path, log_dir: Path) -> None:
    folder = _junk_dir(tmp_path, log_dir)
    with pytest.raises(NotAnInspectLog):
        load_runs([folder, folder / "nested" / "config.json"], on_skip=lambda e: None)


def test_conversion_bugs_are_not_swallowed(
    tmp_path: Path, log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evalsift.adapters.inspect as adapter

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bug in evalsift")

    monkeypatch.setattr(adapter, "_sample_to_run", broken)
    with pytest.raises(RuntimeError, match="bug in evalsift"):
        load_runs([_junk_dir(tmp_path, log_dir)], on_skip=lambda e: None)


def test_cli_reports_skips(
    tmp_path: Path, log_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from evalsift.cli import main

    folder = _junk_dir(tmp_path, log_dir)
    assert main(["check", str(folder), "--fail-on", "never"]) == 0
    err = capsys.readouterr().err
    assert err.count("evalsift: skipped") == len(JUNK)

    assert main(["check", str(folder / "nested")]) == 2
    assert f"{len(JUNK)} file(s) skipped" in capsys.readouterr().err

    assert main(["check", str(folder / "nested" / "fake.eval")]) == 2
    assert "not a readable Inspect log" in capsys.readouterr().err
