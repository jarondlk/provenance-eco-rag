"""
Shared pytest fixtures for the Onagawa Source Chat test suite.

All fixtures produce synthetic in-memory data — no real files,
database, or Ollama required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def sample_ctd_summary() -> pd.DataFrame:
    """5-row CTD summary DataFrame mimicking normalized/ctd_summary.parquet."""
    return pd.DataFrame({
        "sample_id": ["2024-04-O-s1", "2024-05-O-s1", "2024-06-O-s4",
                       "2024-07-O-s1", "2024-08-O-s4"],
        "ctd_date": pd.to_datetime([
            "2024-04-15", "2024-05-10", "2024-06-20",
            "2024-07-12", "2024-08-25",
        ]),
        "bay": ["O", "O", "O", "O", "O"],
        "station_code": ["s1", "s1", "s4", "s1", "s4"],
        "temperature": [10.5, 14.2, 18.7, 22.1, 24.3],
        "salinity": [33.2, 33.0, 32.8, 32.5, 32.1],
        "do_percent": [95.0, 88.0, 82.0, 75.0, 70.0],
        "chl_a": [1.2, 2.5, 5.1, 8.3, 3.2],
        "min_depth_m": [0.0, 0.0, 0.0, 0.0, 0.0],
        "max_depth_m": [20.0, 25.0, 22.0, 18.0, 30.0],
    })


@pytest.fixture
def sample_sst_daily() -> pd.DataFrame:
    """10-row SST daily summary DataFrame."""
    dates = pd.date_range("2024-04-10", periods=10, freq="D")
    return pd.DataFrame({
        "date_jst": dates.strftime("%Y-%m-%d").tolist(),
        "mean_sst": np.linspace(10.0, 12.0, 10).tolist(),
        "min_sst": np.linspace(9.5, 11.5, 10).tolist(),
        "max_sst": np.linspace(10.5, 12.5, 10).tolist(),
        "std_sst": [0.3] * 10,
        "n_obs": [24] * 10,
    })


@pytest.fixture
def sample_metagenome() -> pd.DataFrame:
    """5-row metagenome sample DataFrame with diversity indices."""
    return pd.DataFrame({
        "sample_id": ["2024-04-O-s1", "2024-05-O-s1", "2024-06-O-s4",
                       "2024-07-O-s1", "2024-08-O-s4"],
        "bay": ["O", "O", "O", "O", "O"],
        "shannon_h": [3.8, 4.1, 3.9, 1.6, 4.0],
        "simpson_1d": [0.92, 0.94, 0.93, 0.45, 0.93],
        "richness": [350, 420, 380, 90, 400],
    })


@pytest.fixture
def sample_analysis_docs(tmp_data_dir: Path) -> Path:
    """Create a temporary analysis_documents.jsonl file."""
    docs = [
        {"id": "analysis_trends", "analysis_type": "trend",
         "title": "CTD monthly temperature trends",
         "text": "Temperature rises from 10°C in April to 24°C in August."},
        {"id": "analysis_correlations", "analysis_type": "correlation",
         "title": "Taxa-environment correlations",
         "text": "Gyrodinium × temperature: ρ=−0.60, p=0.0001."},
    ]
    path = tmp_data_dir / "analysis_documents.jsonl"
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path


@pytest.fixture
def sample_reliability_docs(tmp_data_dir: Path) -> Path:
    """Create a temporary reliability_documents.jsonl file."""
    docs = [
        {"id": "reliability_sst_ctd_validation", "analysis_type": "cross_source_validation",
         "title": "SST ↔ CTD surface temperature cross-validation",
         "text": "24 paired observations, 100% agreement, mean |ΔT|=0.92°C."},
        {"id": "reliability_corroboration_summary", "analysis_type": "corroboration",
         "title": "Cross-source corroboration summary",
         "text": "207 observations: 37 verified, 20 supported, 150 standalone."},
    ]
    path = tmp_data_dir / "reliability_documents.jsonl"
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path
