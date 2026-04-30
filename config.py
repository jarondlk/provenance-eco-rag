"""
Centralized configuration for the Onagawa Source Chat framework.

All paths, database URLs, model settings, and environment variable
overrides are collected here so that every module imports from one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root – resolves relative to this file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data directory layout
# ---------------------------------------------------------------------------
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
RAW_CTD_DIR    = RAW_DIR / "ctd"
RAW_META_DIR   = RAW_DIR / "meta"
RAW_SST_DIR    = RAW_DIR / "sst"       # symlink or copy from onagawa_sst_subset/
NORMALIZED_DIR = DATA_DIR / "normalized"
CANONICAL_DIR  = DATA_DIR / "canonical"
SERVING_DIR    = DATA_DIR / "serving"
ANALYSIS_DIR   = DATA_DIR / "analysis"
PROVENANCE_DIR = DATA_DIR / "provenance"

# Satellite SST source (NetCDF subset files)
SST_NETCDF_DIR = PROJECT_ROOT / "onagawa_sst_subset"

# Raw Himawari .DAT files (optional, parsed via satpy)
HIMAWARI_RAW_DIR      = PROJECT_ROOT / "himawari_test_unzipped"

# ---------------------------------------------------------------------------
# Known raw file registry (matches notebook FILES dict)
# ---------------------------------------------------------------------------
RAW_FILES = {
    "ctd":                      RAW_CTD_DIR / "CTD_Onagawa.tsv",
    "runid":                    RAW_META_DIR / "runid.tsv",
    "read_summary":             RAW_META_DIR / "01.read_summary_gt1kb.tsv",
    "coverage_log":             RAW_META_DIR / "03.coverage.log.tsv",
    "kraken_genus_sample_tsv":  RAW_META_DIR / "Kraken.genus-sample.tsv",
    "kraken_genus_sample_txt":  RAW_META_DIR / "Kraken.genus-sample.txt",
    "kraken_upper_group_sample": RAW_META_DIR / "Kraken.upper_group-sample.txt",
    "kraken_genus_group":       RAW_META_DIR / "Kraken.genus-group.tsv",
    "metaeuk_genus_sample":     RAW_META_DIR / "MetaEuk.genus-sample.tsv",
    "genus_group":              RAW_META_DIR / "genus-group.tsv",
    "gn_consistency":           RAW_META_DIR / "gn.consistency.tsv",
    "km_consistency":           RAW_META_DIR / "km.consistency.tsv",
}

# ---------------------------------------------------------------------------
# Default Onagawa monitoring station coordinates
# ---------------------------------------------------------------------------
ONAGAWA_LAT = 38.42907415591698
ONAGAWA_LON = 141.4775733277202

# Regional bounding box for SST subset
SST_LAT_MIN, SST_LAT_MAX = 38.0, 39.0
SST_LON_MIN, SST_LON_MAX = 141.0, 142.0

# ---------------------------------------------------------------------------
# Database (PostgreSQL + pgvector)
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://onagawa:onagawa@localhost:5433/onagawa_rag",
)

# ---------------------------------------------------------------------------
# LLM / Embedding
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL      = os.environ.get("CHAT_MODEL", "qwen2.5:14b-instruct")

# Embedding dimension (nomic-embed-text → 768)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [NORMALIZED_DIR, CANONICAL_DIR, SERVING_DIR, PROVENANCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
