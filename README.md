# evalsift
 
**Which of your agent runs actually measured the model and what do the valid ones say?**
 
Agent evaluations fail in two different ways, and most pipelines blur them. Some runs are
*invalid*: a tool crashed, the API dropped, no score was written. They tell you nothing about
the model and should be excluded. Others are *valid but rough*: a malformed tool call, a
loop, a truncated answer. Those are real model behaviour and worth reading. Counting an
infrastructure crash as a model failure (or silently dropping it) quietly changes your
headline number.

`evalsift` reads [Inspect](https://inspect.aisi.org.uk/) logs, separates the two, and reports
rates on valid runs only, with Wilson confidence intervals and an explicit count of what was
excluded.

## Result on the bundled demo

`make demo` builds 30 runs offline (10 scripted scenarios × 3 epochs, using Inspect's mock
model, so no API key or GPU) and runs the tool on them:

| model | runs | valid | invalid_rate | invalid_ci | runs_with_warnings | match:rate | match:ci | match:n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mockllm/model | 30 | 27 | 0.100 | [0.035, 0.256] | 15 | 0.556 | [0.373, 0.724] | 27 |

All 3 crashed runs are flagged invalid and excluded, giving a pass rate of 15/27. Scoring
them as failures would report 15/30 = 0.500 instead, a 5.6-point understatement caused purely
by infrastructure. The 15 runs with warnings (tool errors, loops, a hit message limit,
truncation, an empty answer) stay in the denominator and are flagged for reading.

## Limitations

- **Inspect logs only, for now.** Other formats need an adapter (see `src/evalsift/adapters/`).
- **Checks are structural, not semantic.** evalsift sees that a tool errored or a call
  repeated; it cannot tell whether an answer was *reasonable*. That is the scorer's job.
- **"Invalid" is a policy.** Hitting a message limit is a normal outcome in some benchmarks
  and a broken run in others. Defaults are conservative; set yours with
  `--severity limit_hit=error`.
- **Per-turn latency** comes from matching model events to assistant messages. If a solver
  rewrites the message list and the counts disagree, latency is left blank rather than guessed.
- **Redaction is pattern-based.** It catches known key formats, `Bearer` headers and
  `api_key=`-style assignments. A secret with no recognisable shape is only caught with
  `--redact-env`, which masks the live values of your `*KEY*`/`*TOKEN*` environment variables.
- **Wilson intervals assume independent runs.** Epochs of the same sample are correlated, so
  pooled intervals across epochs are optimistic. Group by `sample_id` to inspect this.
- **Demo data is scripted.** The numbers above show the mechanics, not any real model's
  performance.

## Install

```sh
pip install evalsift            # once published; until then:
pip install git+https://github.com/qymmore/evalsift
```

Or with Docker: `docker run --rm -v "$PWD/logs:/work/logs" evalsift check logs`

## Use

```sh
evalsift check   logs/                         # findings table; exits 1 if any run is invalid
evalsift summary logs/ --by task,model         # rates with 95% Wilson intervals
evalsift export  logs/ -o out/ --only flagged  # Markdown traces + runs/timing/findings CSVs
evalsift bundle  logs/ -o review.zip           # redacted zip for a reviewer or mentor
evalsift scan    review/                       # look for secrets before sharing anything
evalsift checks                                # list checks and default severities
```

`check` is built for CI: `--fail-on error|warning|never` sets the exit code, `--format json`
gives machine-readable output.

### Checks

| check | default | flags |
| --- | --- | --- |
| `sample_error` | error | the sample raised and did not finish |
| `missing_score` | error | no score, or a scorer returned nothing |
| `no_turns` | error | the model never replied |
| `limit_hit` | warning | a message, token, time or cost limit ended the run |
| `empty_final` | warning | final answer has no text and no tool call |
| `truncated_output` | warning | stopped on `max_tokens` or context length |
| `content_filter` | warning | provider filter stopped generation |
| `model_error` | warning | the API call for a turn errored |
| `tool_parse_error` | warning | tool-call arguments failed to parse |
| `tool_error` | warning | a tool returned an error |
| `unanswered_tool_call` | warning | a call never got a result (not reported when a limit or crash already explains it) |
| `repeated_tool_call` | warning | identical calls N turns in a row (`--repeat-threshold`, default 3) |
| `empty_turn` | warning | a mid-run turn with no text, reasoning or calls |

Override any severity with `--severity CHECK=error|warning|info`, skip one with `--disable`.

### Review bundles

`evalsift bundle` writes one zip a reviewer can open with no tooling: a README with the
summary table, full CSVs, Markdown traces (every invalid run, every run with warnings, and a
seeded random sample of clean runs), and `manifest.json` with SHA-256 hashes of the source
logs and the exact options used. Findings appear inline next to the turn they refer to, and
reasoning blocks are collapsed under each turn.

Everything is redacted before writing, then re-scanned. If anything still looks like a
secret, no file is written and the command fails.

## Reproduce

```sh
git clone https://github.com/qymmore/evalsift && cd evalsift
make install   # editable install + pre-commit hooks (gitleaks, ruff, mypy)
make demo      # builds demo logs offline and runs every command above
make test      # 75 tests, real Inspect logs generated offline
```

## Engineering

- Typed throughout (`mypy --strict`), linted and formatted with ruff.
- Tests run against real Inspect logs produced by Inspect's own logging code with a scripted
  mock model, so adapter tests track the real schema rather than hand-written fixtures.
  Coverage is enforced at 85% (currently about 94%).
- CI on Python 3.10–3.13: lint, types, tests, a CLI smoke test, a Docker build, and a
  full-history gitleaks scan. Releases publish to PyPI from version tags via trusted
  publishing, so no API token is stored anywhere.
- `.gitignore` excludes `*.eval`, `logs/` and `*.zip` so real logs and bundles are not
  committed by accident.

## Mistakes and negative results

Kept here on purpose: these are what the tests now guard against.

- **The redaction placeholder matched its own pattern.** An early `api_key=…` rule matched
  `[REDACTED:assigned_secret]`, so any bundle containing a redacted assignment would have been
  refused as a leak. Running `evalsift scan` on this repo caught it. Placeholders are now
  excluded, and a test asserts redaction is idempotent for every pattern.
- **The same rule flagged ordinary code** such as `secret = text[m.start : m.end]`. It now
  requires a credential-shaped value: no brackets or parentheses, at least one digit.
- **Hand-typed reference values for the Wilson tests were wrong** for two of five cases. The
  implementation was right; the references are now taken from `statsmodels`.
- **Multi-epoch demo runs drifted.** Inspect's mock model serves scripted outputs from one
  queue, so parallel epochs, or an epoch that crashed before using its outputs, shifted later
  epochs onto the wrong script. Each scenario now consumes exactly its own outputs and
  samples run one at a time. A useful reminder that fixture logic can fake a "finding".

## Extending

An adapter only needs to return `list[Run]` (see `evalsift/model.py`). Checks are plain
functions registered with a name and default severity:

```python
from evalsift.checks import register
from evalsift.model import Finding, Severity

@register("too_many_turns", Severity.WARNING, "Run used more than 40 turns.")
def too_many_turns(run, config):
    if len(run.turns) > 40:
        yield Finding(run.run_id, "too_many_turns", Severity.WARNING, f"{len(run.turns)} turns")
```

## License

MIT
