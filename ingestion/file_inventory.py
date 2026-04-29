"""
Auto-discover and catalog raw files from a directory.

Extracted from notebook cell 3 (01_phase1_ingestion.ipynb).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def inventory_dir(path: Path) -> pd.DataFrame:
    """
    Scan *path* for files and return a catalogue DataFrame with columns:
      name, path, size_bytes, suffix
    """
    rows = []
    for p in sorted(path.glob("*")):
        if p.is_file() and not p.name.startswith("."):
            rows.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "suffix": p.suffix.lower(),
                }
            )
    return pd.DataFrame(rows)


def inventory_recursive(path: Path, pattern: str = "*") -> pd.DataFrame:
    """Like ``inventory_dir`` but recursive (``rglob``)."""
    rows = []
    for p in sorted(path.rglob(pattern)):
        if p.is_file() and not p.name.startswith("."):
            rows.append(
                {
                    "name": p.name,
                    "relative_path": str(p.relative_to(path)),
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "suffix": p.suffix.lower(),
                }
            )
    return pd.DataFrame(rows)
