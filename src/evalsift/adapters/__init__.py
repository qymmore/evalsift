"""Log-format adapters. Each exposes ``load_runs(paths) -> list[Run]``."""

from evalsift.adapters.inspect import discover, load_runs

__all__ = ["discover", "load_runs"]
