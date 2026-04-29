"""
Shared helpers for parsing sample IDs, reading TSVs, and normalizing columns.

Extracted from notebook cells 4, 22, 34 of 01_phase1_ingestion.ipynb.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Sample ID parsing (notebook cell 4)
# ---------------------------------------------------------------------------
SAMPLE_ID_RE = re.compile(
    r"^(?P<sample_id>\d{4}-\d{2}-[OIM]-[A-Za-z0-9]+)(?:\.(?P<replicate>\d+))?$"
)


def parse_sample_replicate(value: str) -> dict:
    """
    Parse sample-replicate strings like:
        2024-06-O-s4      →  sample_id=2024-06-O-s4, replicate=1
        2024-06-O-s4.1    →  sample_id=2024-06-O-s4, replicate=1
        2025-03-I-hm.2    →  sample_id=2025-03-I-hm, replicate=2
    """
    s = str(value).strip()
    m = SAMPLE_ID_RE.match(s)
    if not m:
        return {
            "sample_replicate": s,
            "sample_id": pd.NA,
            "replicate_no": pd.NA,
            "sample_year_month": pd.NA,
            "bay": pd.NA,
            "station_code": pd.NA,
        }

    sample_id = m.group("sample_id")
    replicate_no = m.group("replicate")

    parts = sample_id.split("-")
    year_month = f"{parts[0]}-{parts[1]}"
    bay = parts[2]
    station_code = parts[3]

    return {
        "sample_replicate": s,
        "sample_id": sample_id,
        "replicate_no": int(replicate_no) if replicate_no is not None else 1,
        "sample_year_month": year_month,
        "bay": bay,
        "station_code": station_code,
    }


# ---------------------------------------------------------------------------
# TSV I/O helpers (notebook cell 4)
# ---------------------------------------------------------------------------
def read_tsv_no_header(path: Path, columns: List[str]) -> pd.DataFrame:
    """Read a headerless TSV and assign the given column names."""
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] != len(columns):
        raise ValueError(
            f"{path.name}: expected {len(columns)} cols, got {df.shape[1]}"
        )
    df.columns = columns
    return df


def read_tsv_with_header(path: Path) -> pd.DataFrame:
    """Read a TSV that has a header row."""
    return pd.read_csv(path, sep="\t")


def add_sample_parsed_columns(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Parse a sample_replicate column and add derived fields."""
    parsed = df[source_col].apply(parse_sample_replicate).apply(pd.Series)
    out = df.copy()
    for c in parsed.columns:
        out[c] = parsed[c]
    return out


# ---------------------------------------------------------------------------
# Column name canonicalization (notebook cell 22)
# ---------------------------------------------------------------------------
def canonicalize_colname(col: str) -> str:
    """Normalize a column name to lowercase snake_case."""
    s = str(col).strip().lower()
    s = s.replace("chl-a", "chl_a")
    s = s.replace("sigmaT", "sigma_t")
    s = re.sub(r"[%()/]", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


CTD_ALIAS_MAP = {
    "sample_id": "sample_id",
    "label": "sample_id",
    "ctd_date": "ctd_date",
    "date": "ctd_date",
    "depth": "depth_m",
    "depth_m": "depth_m",
    "temperature": "temperature",
    "temp": "temperature",
    "salinity": "salinity",
    "sigma_t": "sigma_t",
    "sigmat": "sigma_t",
    "chl_a": "chl_a",
    "chla": "chl_a",
    "chl_flu": "chl_flu",
    "do": "do_percent",
    "do_percent": "do_percent",
    "do_mgl": "do_mg_l",
    "do_mg_l": "do_mg_l",
    "turbidity": "turbidity",
    "ec": "ec",
    "ec25": "ec25",
    "density": "density",
    "voltage": "voltage",
    "orp": "orp",
    "ph": "ph",
    "par": "par",
}


# ---------------------------------------------------------------------------
# Derive sample dimensions from IDs (notebook cell 34)
# ---------------------------------------------------------------------------
def derive_sample_dims(sample_ids: pd.Series) -> pd.DataFrame:
    """Extract year-month, bay code, and station code from sample IDs."""
    out = pd.DataFrame({"sample_id": pd.Series(sample_ids, dtype="string")})
    out["sample_year_month"] = out["sample_id"].str.extract(
        r"^(\d{4}-\d{2})", expand=False
    )
    out["bay"] = out["sample_id"].str.extract(
        r"^\d{4}-\d{2}-([OIM])-", expand=False
    )
    out["station_code"] = out["sample_id"].str.extract(
        r"^\d{4}-\d{2}-[OIM]-(.+)$", expand=False
    )
    return out


# ---------------------------------------------------------------------------
# Genus key normalization (notebook cell 31)
# ---------------------------------------------------------------------------
def normalize_genus_key(series: pd.Series) -> pd.Series:
    """Strip whitespace and replace blank/null-like strings with NA."""
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return s
