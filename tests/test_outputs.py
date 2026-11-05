from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from logfactory import FAKE_KEY

from evalsift.bundle import BundleOptions, SecretLeakError, build_bundle, select_runs
from evalsift.cli import main
from evalsift.export import trace_markdown, write_jsonl, write_traces
from evalsift.model import Finding, Run
from evalsift.redact import Redactor, scan_paths

# Built by concatenation so the repo itself contains no scannable secrets.
SAMPLES = {
    "anthropic_key": "sk-" + "ant-api03-" + "x" * 30,
    "openrouter_key": "sk-" + "or-v1-" + "a1" * 20,
    "openai_key": FAKE_KEY,
    "github_token": "gh" + "p_" + "Z" * 36,
    "huggingface_token": "h" + "f_" + "q" * 34,
    "aws_access_key": "AK" + "IA" + "ABCDEFGHIJKLMNOP",
}


# ---------------------------------------------------------------- redaction


@pytest.mark.parametrize(("kind", "secret"), sorted(SAMPLES.items()))
def test_known_formats_redacted(kind: str, secret: str) -> None:
    out = Redactor().redact(f"before {secret} after")
    assert out == f"before [REDACTED:{kind}] after"


BEARER = "abcdefgh" + "ijklmnopqrstuvwxyz"
PASSWORD = "hunter2" + "hunter2!"


def test_bearer_and_assignment_keep_context() -> None:
    red = Redactor()
    assert red.redact(f"Authorization: Bearer {BEARER}") == (
        "Authorization: Bearer [REDACTED:bearer_token]"
    )
    assert red.redact(f'password = "{PASSWORD}"') == 'password = "[REDACTED:assigned_secret]"'


@pytest.mark.parametrize(
    "text",
    [
        *SAMPLES.values(),
        f"Authorization: Bearer {BEARER}",
        f'password = "{PASSWORD}"',
        f"api_key={FAKE_KEY}",
    ],
)
def test_redaction_is_idempotent(text: str) -> None:
    # Placeholders must never look like secrets, or bundle verification would
    # refuse every bundle that contains a redaction.
    red = Redactor()
    once = red.redact(text)
    assert red.find(once) == []
    assert red.redact(once) == once


@pytest.mark.parametrize(
    "code",
    [
        "secret = text[m.start : m.end]",
        "token = tokenizer.encode(prompt)",
        "password = get_password_from_vault()",
    ],
)
def test_code_assignments_not_flagged(code: str) -> None:
    assert Redactor().find(code) == []


def test_no_false_positive_on_ordinary_text() -> None:
    text = "The task asked for sk-learn tips and a token budget of 4000."
    assert Redactor().redact(text) == text


def test_env_values_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SERVICE_TOKEN", "shapeless-but-secret-value")
    text = "got shapeless-but-secret-value back"
    assert Redactor().redact(text) == text
    assert Redactor(include_env=True).redact(text) == "got [REDACTED:env:MY_SERVICE_TOKEN] back"


def test_custom_pattern_and_overlaps() -> None:
    red = Redactor(extra_patterns=[r"case-\d{6}"])
    assert red.redact("ref case-123456") == "ref [REDACTED:custom_0]"
    # api_key=sk-... matches two patterns; it must be masked once, not mangled.
    out = Redactor().redact(f"api_key={FAKE_KEY}")
    assert out.count("REDACTED") == 1 and FAKE_KEY not in out


def test_scan_reports_location_without_secret(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text(f"ok\nkey {FAKE_KEY}\n")
    (tmp_path / "b.bin").write_bytes(b"\xff\xfe\x00")
    (hit,) = scan_paths([tmp_path])
    assert (hit.path.name, hit.line, hit.name) == ("a.txt", 2, "openai_key")
    assert FAKE_KEY not in hit.preview


# ---------------------------------------------------------------- export


def test_trace_markdown(by_sample: dict[str, Run], findings: list[Finding]) -> None:
    md = trace_markdown(by_sample["tool_error"], findings)
    assert "## Turn 0" in md and "**Tool call** `lookup`" in md
    assert "⚠️ **tool_error**" in md
    md = trace_markdown(by_sample["reasoning"], findings)
    assert "<summary>Reasoning</summary>" in md and "1 plus 2 is 3." in md
    md = trace_markdown(by_sample["crash"], findings)
    assert "## Run-level findings" in md and "❌ **sample_error**" in md


def test_leaked_key_never_exported(
    by_sample: dict[str, Run], findings: list[Finding], tmp_path: Path
) -> None:
    run = by_sample["leak"]
    assert FAKE_KEY in run.turns[0].tool_results[0].content  # the log really has it
    assert FAKE_KEY not in trace_markdown(run, findings)
    write_jsonl([run], findings, tmp_path / "runs.jsonl")
    text = (tmp_path / "runs.jsonl").read_text()
    assert FAKE_KEY not in text
    assert json.loads(text)["sample_id"] == "leak"


def test_fence_cannot_be_escaped(by_sample: dict[str, Run]) -> None:
    run = by_sample["clean"]
    run_copy = Run(**{**run.__dict__, "input_text": "```\n# injected heading\n```"})
    md = trace_markdown(run_copy)
    assert "````\n```\n# injected heading\n```\n````" in md


def test_write_traces_unique_names(
    runs: list[Run], findings: list[Finding], tmp_path: Path
) -> None:
    paths = write_traces(runs + runs[:1], findings, tmp_path)
    assert len(paths) == len(runs) + 1 == len(set(paths))


# ---------------------------------------------------------------- bundle


def test_bundle_contents(runs: list[Run], findings: list[Finding], tmp_path: Path) -> None:
    out = tmp_path / "review.zip"
    manifest = build_bundle(runs, findings, out, BundleOptions(sample_valid=2, seed=7))
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {
            "README.md",
            "manifest.json",
            "summary.csv",
            "runs.csv",
            "timing.csv",
            "findings.csv",
        } <= names
        traces = [n for n in names if n.startswith("traces/")]
        assert any("INVALID_crash" in n for n in traces)
        everything = "".join(zf.read(n).decode() for n in names)
    assert FAKE_KEY not in everything
    assert manifest["counts"] == {"runs": len(runs), "invalid": 1, "traces": len(traces)}
    assert all(len(s["sha256"]) == 64 for s in manifest["sources"])


def test_selection_is_deterministic(runs: list[Run], findings: list[Finding]) -> None:
    opts = BundleOptions(include_warned=False, sample_valid=3, seed=1)
    first = [r.run_id for r in select_runs(runs, findings, opts)]
    assert first == [r.run_id for r in select_runs(runs, findings, opts)]
    assert len(first) == 4  # the invalid crash run plus 3 sampled


def test_bundle_unknown_run_id(runs: list[Run], findings: list[Finding], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        build_bundle(runs, findings, tmp_path / "x.zip", BundleOptions(run_ids=["nope"]))


def test_bundle_refuses_unredactable(
    runs: list[Run], findings: list[Finding], tmp_path: Path
) -> None:
    class LeakyRedactor(Redactor):
        def redact(self, text: str) -> str:  # simulate a redaction bug
            return text

    out = tmp_path / "leaky.zip"
    with pytest.raises(SecretLeakError):
        build_bundle(runs, findings, out, redactor=LeakyRedactor())
    assert not out.exists()


# ---------------------------------------------------------------- CLI


def test_cli_check_exit_codes(log_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(log_dir)]) == 1  # the crash run is invalid
    out = capsys.readouterr().out
    assert "1 invalid" in out and "sample_error" in out
    assert main(["check", str(log_dir), "--fail-on", "never"]) == 0
    assert main(["check", str(log_dir), "--disable", "sample_error"]) == 0


def test_cli_check_json(log_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(log_dir), "--format", "json", "--fail-on", "never"])
    rows = json.loads(capsys.readouterr().out)
    assert {r["check"] for r in rows} >= {"sample_error", "tool_error"}


def test_cli_summary_and_export(
    log_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["summary", str(log_dir), "--by", "model", "--out", str(tmp_path / "t")]) == 0
    assert "invalid_rate" in capsys.readouterr().out
    assert (tmp_path / "t" / "summary.csv").exists()
    assert (
        main(["export", str(log_dir), "-o", str(tmp_path / "e"), "--only", "flagged", "--jsonl"])
        == 0
    )
    assert len(list((tmp_path / "e" / "traces").glob("*.md"))) == 6
    assert (tmp_path / "e" / "runs.jsonl").exists()


def test_cli_bundle_scan_and_list(
    log_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "b.zip"
    assert main(["bundle", str(log_dir), "-o", str(out), "--sample", "1"]) == 0
    assert out.exists()
    secret = tmp_path / "s.txt"
    secret.write_text(FAKE_KEY)
    assert main(["scan", str(secret)]) == 1
    assert (
        main(["scan", str(tmp_path / "e")] if (tmp_path / "e").exists() else ["scan", str(out)])
        == 0
    )
    assert main(["checks"]) == 0
    assert "repeated_tool_call" in capsys.readouterr().out


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(tmp_path / "absent")]) == 2
    assert main(["check", str(tmp_path)]) == 2  # empty directory: no runs
    assert main(["check", str(tmp_path), "--severity", "bad"]) == 2
    assert main([]) == 2
    assert "error" in capsys.readouterr().err
