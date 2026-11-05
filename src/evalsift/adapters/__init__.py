"""Log-format adapters. Each exposes ``load_runs(paths) -> list[Run]``."""

from evalsift.adapters.inspect import NotAnInspectLog, discover, load_runs

__all__ = ["NotAnInspectLog", "discover", "load_runs"]
