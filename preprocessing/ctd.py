"""
CTD preprocessing pipeline.

Extracted from notebook cells 11, 13, 22, 24 of 01_phase1_ingestion.ipynb.

Pipeline:
    load_ctd_raw  →  standardize_ctd_columns  →  summarize_ctd_profiles
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .common import (
    CTD_ALIAS_MAP,
    canonicalize_colname,
    read_tsv_with_header,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load raw CTD (notebook cell 11 + 13)
# ---------------------------------------------------------------------------
def load_ctd_raw(path: Path) -> pd.DataFrame:
    """
    Load the raw CTD TSV file, normalize the key columns
    (label→sample_id, date→ctd_date, depth→depth_m), and coerce types.
    """
    ctd = read_tsv_with_header(path)
    ctd.columns = [str(c).strip() for c in ctd.columns]

    # Find key columns defensively
    col_map = {c.lower(): c for c in ctd.columns}

    def _find(name_options: List[str]) -> Optional[str]:
        for opt in name_options:
            if opt.lower() in col_map:
                return col_map[opt.lower()]
        return None

    label_col = _find(["label"])
    date_col = _find(["date"])
    depth_col = _find(["depth", "depth(m)", "depth_m"])

    if label_col is None:
        raise ValueError("Could not find CTD label column.")
    if date_col is None:
        raise ValueError("Could not find CTD date column.")
    if depth_col is None:
        raise ValueError("Could not find CTD depth column.")

    ctd = ctd.rename(
        columns={
            label_col: "sample_id",
            date_col: "ctd_date",
            depth_col: "depth_m",
        }
    )

    ctd["ctd_date"] = pd.to_datetime(ctd["ctd_date"], errors="coerce")
    ctd["depth_m"] = pd.to_numeric(ctd["depth_m"], errors="coerce")

    # Convert all non-key columns to numeric where possible.
    for c in ctd.columns:
        if c not in {"sample_id", "ctd_date"}:
            ctd[c] = pd.to_numeric(ctd[c], errors="ignore")

    logger.info(
        "Loaded CTD: %d rows, %d columns, %d unique samples",
        len(ctd),
        len(ctd.columns),
        ctd["sample_id"].nunique(),
    )
    return ctd


# ---------------------------------------------------------------------------
# 2. Standardize column names (notebook cell 22)
# ---------------------------------------------------------------------------
def standardize_ctd_columns(ctd: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize CTD column names using ``CTD_ALIAS_MAP``.

    Columns not in the alias map are retained with their canonicalized names.
    """
    canonical = {}
    for col in ctd.columns:
        canon = canonicalize_colname(col)
        mapped = CTD_ALIAS_MAP.get(canon, canon)
        canonical[col] = mapped

    ctd_std = ctd.rename(columns=canonical)

    # Coerce numeric columns
    for c in ctd_std.columns:
        if c not in {"sample_id", "ctd_date"}:
            ctd_std[c] = pd.to_numeric(ctd_std[c], errors="coerce")

    logger.info("CTD standardized columns: %s", list(ctd_std.columns))
    return ctd_std


# ---------------------------------------------------------------------------
# 3. Summarize profiles (notebook cell 24)
# ---------------------------------------------------------------------------
def _value_at_surface(group: pd.DataFrame, col: str):
    """Return the measurement value at the shallowest depth."""
    s = group[["depth_m", col]].dropna().sort_values("depth_m", ascending=True)
    return s[col].iloc[0] if not s.empty else np.nan


def _value_at_bottom(group: pd.DataFrame, col: str):
    """Return the measurement value at the deepest depth."""
    s = group[["depth_m", col]].dropna().sort_values("depth_m", ascending=False)
    return s[col].iloc[0] if not s.empty else np.nan


def summarize_ctd_profiles(ctd_std: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-sample summary statistics for each CTD cast.

    Returns a DataFrame with one row per sample_id, containing
    surface/bottom/mean/min/max for each available variable.
    """
    candidate_vars = [
        "temperature",
        "salinity",
        "sigma_t",
        "chl_a",
        "do_percent",
        "do_mg_l",
        "turbidity",
        "ec",
        "ec25",
        "density",
        "orp",
        "ph",
        "par",
        "chl_flu",
        "voltage",
    ]

    available_vars = [c for c in candidate_vars if c in ctd_std.columns]
    rows = []

    for sample_id, g in ctd_std.groupby("sample_id", dropna=False):
        row = {
            "sample_id": sample_id,
            "ctd_date": g["ctd_date"].iloc[0] if "ctd_date" in g.columns else pd.NaT,
            "n_depth_points": len(g),
            "min_depth_m": g["depth_m"].min(),
            "max_depth_m": g["depth_m"].max(),
        }

        for col in available_vars:
            row[f"surface_{col}"] = _value_at_surface(g, col)
            row[f"bottom_{col}"] = _value_at_bottom(g, col)
            row[f"mean_{col}"] = pd.to_numeric(g[col], errors="coerce").mean()
            row[f"min_{col}"] = pd.to_numeric(g[col], errors="coerce").min()
            row[f"max_{col}"] = pd.to_numeric(g[col], errors="coerce").max()

        rows.append(row)

    summary = pd.DataFrame(rows)
    logger.info("CTD summary: %d samples", len(summary))
    return summary
