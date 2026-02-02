"""Smoke tests to verify basic project setup."""

import sys
from pathlib import Path


def test_python_version() -> None:
    """Verify Python version is 3.10+."""
    assert sys.version_info >= (3, 10), "Python 3.10+ required"


def test_project_structure() -> None:
    """Verify critical project directories exist."""
    project_root = Path(__file__).parent.parent.parent

    critical_dirs = [
        "src/tradebox",
        "src/tradebox/data",
        "src/tradebox/features",
        "src/tradebox/env",
        "src/tradebox/agents",
        "src/tradebox/backtest",
        "src/tradebox/risk",
        "src/tradebox/execution",
        "src/tradebox/monitoring",
        "src/tradebox/utils",
        "configs",
        "scripts",
        "tests/unit",
        "tests/integration",
    ]

    for dir_path in critical_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Missing directory: {dir_path}"


def test_can_import_tradebox() -> None:
    """Verify tradebox package can be imported."""
    try:
        import tradebox
        assert tradebox is not None
    except ImportError as e:
        assert False, f"Cannot import tradebox package: {e}"
