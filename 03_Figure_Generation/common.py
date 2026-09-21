"""Shared utilities for local/GitHub execution of figure-generation scripts."""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def discover_files(input_dir):
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def select_forecast_files(input_dir):
    """Discover forecast files recursively; model/station/horizon are parsed by each figure script."""
    return discover_files(input_dir)
