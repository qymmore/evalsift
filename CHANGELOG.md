# Changelog

## 0.1.0 (unreleased)

First release.

- Inspect adapter for `.eval` and `.json` logs, normalised into a source-independent run model.
- 13 checks separating invalid runs (excluded from rates) from failed turns (flagged), with
  per-check severity overrides.
- Summary tables with Wilson score intervals on valid runs only, grouped by task, model,
  epoch, sample or any metadata field.
- Markdown trace export with inline findings and reasoning blocks; runs, timing and findings
  CSVs; JSONL export.
- Redacted review bundles with source-log hashes, re-scanned before writing.
- `evalsift scan` for secrets in files you are about to share.
- Files with a log extension that aren't Inspect logs (a stray `config.json`, a corrupt
  `.eval`) are skipped with a message when found in a folder, and are a clean error when
  named directly, instead of crashing with a traceback.
