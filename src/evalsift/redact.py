"""Secret detection and redaction.

Agent transcripts routinely capture credentials: a tool that echoes its
environment, an agent that reads a config file, an error message with an
Authorization header. Everything evalsift writes for sharing goes through
``redact``, and ``scan`` checks arbitrary files before they leave your machine.

The patterns favour precision over recall. They catch well-known key formats
and explicit ``Bearer``/``api_key=`` assignments; they will not find a secret
with no recognisable shape. Pair this with a repository scanner (gitleaks runs
in this project's pre-commit and CI) rather than relying on it alone.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER = "[REDACTED:{name}]"


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    group: int = 0  # which group holds the secret; 0 = whole match


def _p(name: str, pattern: str, group: int = 0) -> Pattern:
    return Pattern(name, re.compile(pattern), group)


# Order matters: more specific prefixes first, so an Anthropic key is labelled
# anthropic rather than the generic OpenAI-style "sk-" pattern.
BUILTIN_PATTERNS: tuple[Pattern, ...] = (
    _p(
        "private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"
    ),
    _p("anthropic_key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    _p("openrouter_key", r"\bsk-or-v1-[A-Za-z0-9]{32,}"),
    _p("openai_key", r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}"),
    _p("github_token", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    _p("huggingface_token", r"\bhf_[A-Za-z0-9]{30,}"),
    _p("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}"),
    _p("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    _p("slack_token", r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    _p("jwt", r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    _p("bearer_token", r"(?i)\bbearer\s+([A-Za-z0-9_\-\.=+/]{20,})", group=1),
    _p(
        "assigned_secret",
        # The value must look like a credential: no brackets or parens (so
        # code such as `token = text[i]` is left alone), at least one digit,
        # and not an earlier redaction placeholder.
        r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*[\"']?"
        r"(?!\[REDACTED)(?=[^\s\"']*\d)([A-Za-z0-9_\-./+=!@#$%^&*~]{12,})",
        group=1,
    ),
)

_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


@dataclass(frozen=True)
class Match:
    name: str
    start: int
    end: int
    line: int


class Redactor:
    """Finds and masks secrets. Optionally also masks live environment values.

    ``include_env=True`` adds the current values of environment variables whose
    names look secret (``*_KEY``, ``*TOKEN*``...). This catches keys of any
    shape that you actually used, which pattern matching cannot.
    """

    def __init__(
        self,
        extra_patterns: Iterable[str] = (),
        include_env: bool = False,
        min_env_length: int = 12,
    ) -> None:
        self.patterns: list[Pattern] = list(BUILTIN_PATTERNS)
        for i, raw in enumerate(extra_patterns):
            self.patterns.append(_p(f"custom_{i}", raw))
        self._env_values: list[tuple[str, str]] = []
        if include_env:
            for name, value in os.environ.items():
                upper = name.upper()
                if len(value) >= min_env_length and any(h in upper for h in _SECRET_ENV_HINTS):
                    self._env_values.append((f"env:{name}", value))
            # Longest first, so a value containing another is masked whole.
            self._env_values.sort(key=lambda kv: -len(kv[1]))

    def find(self, text: str) -> list[Match]:
        spans: list[tuple[int, int, str]] = []
        for name, value in self._env_values:
            start = text.find(value)
            while start != -1:
                spans.append((start, start + len(value), name))
                start = text.find(value, start + 1)
        for pat in self.patterns:
            for m in pat.regex.finditer(text):
                s, e = m.span(pat.group)
                spans.append((s, e, pat.name))
        return [
            Match(name, s, e, text.count("\n", 0, s) + 1) for s, e, name in _non_overlapping(spans)
        ]

    def redact(self, text: str) -> str:
        if not text:
            return text
        out: list[str] = []
        cursor = 0
        for m in self.find(text):
            out.append(text[cursor : m.start])
            out.append(PLACEHOLDER.format(name=m.name))
            cursor = m.end
        out.append(text[cursor:])
        return "".join(out)


def _non_overlapping(spans: list[tuple[int, int, str]]) -> Iterator[tuple[int, int, str]]:
    """Keep the earliest-starting, then longest, span wherever spans overlap."""
    end = -1
    for s, e, name in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if s >= end:
            yield s, e, name
            end = e


@dataclass(frozen=True)
class FileFinding:
    path: Path
    line: int
    name: str
    preview: str


def scan_paths(paths: Iterable[Path], redactor: Redactor | None = None) -> list[FileFinding]:
    """Scan text files (recursively) and report where secrets appear.

    Previews show only the first and last two characters of each secret, so
    the report itself is safe to paste into an issue.
    """
    redactor = redactor or Redactor()
    results: list[FileFinding] = []
    for path in _iter_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; zip bundles are scanned before writing
        for m in redactor.find(text):
            secret = text[m.start : m.end]
            preview = f"{secret[:2]}…{secret[-2:]}" if len(secret) > 6 else "…"
            results.append(FileFinding(path, m.line, m.name, preview))
    return results


def _iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.is_file() and ".git" not in f.parts)
        elif p.is_file():
            yield p
