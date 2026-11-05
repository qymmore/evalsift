from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import logfactory

from evalsift.adapters import load_runs
from evalsift.checks import run_checks
from evalsift.model import Finding, Run


@pytest.fixture(scope="session")
def log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return logfactory.build(tmp_path_factory.mktemp("logs"))


@pytest.fixture(scope="session")
def runs(log_dir: Path) -> list[Run]:
    return load_runs([log_dir])


@pytest.fixture(scope="session")
def by_sample(runs: list[Run]) -> dict[str, Run]:
    return {r.sample_id: r for r in runs}


@pytest.fixture(scope="session")
def findings(runs: list[Run]) -> list[Finding]:
    return run_checks(runs)
