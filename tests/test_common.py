"""Tests for preprocessing/common.py — sample ID parsing and column helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from preprocessing.common import (
    parse_sample_replicate,
    canonicalize_colname,
    derive_sample_dims,
    normalize_genus_key,
)


# ---------------------------------------------------------------------------
# parse_sample_replicate
# ---------------------------------------------------------------------------
class TestParseSampleReplicate:
    """Validate sample ID decomposition."""

    def test_valid_no_replicate(self):
        result = parse_sample_replicate("2024-06-O-s4")
        assert result["sample_id"] == "2024-06-O-s4"
        assert result["replicate_no"] == 1
        assert result["sample_year_month"] == "2024-06"
        assert result["bay"] == "O"
        assert result["station_code"] == "s4"

    def test_valid_with_replicate(self):
        result = parse_sample_replicate("2025-03-I-hm.2")
        assert result["sample_id"] == "2025-03-I-hm"
        assert result["replicate_no"] == 2
        assert result["bay"] == "I"
        assert result["station_code"] == "hm"

    def test_valid_matsushima(self):
        result = parse_sample_replicate("2024-11-M-s1")
        assert result["bay"] == "M"
        assert result["sample_year_month"] == "2024-11"

    def test_invalid_returns_na(self):
        result = parse_sample_replicate("garbage-input")
        assert pd.isna(result["sample_id"])
        assert pd.isna(result["bay"])
        assert pd.isna(result["replicate_no"])

    def test_empty_string(self):
        result = parse_sample_replicate("")
        assert pd.isna(result["sample_id"])

    def test_whitespace_stripped(self):
        result = parse_sample_replicate("  2024-04-O-s1  ")
        assert result["sample_id"] == "2024-04-O-s1"


# ---------------------------------------------------------------------------
# canonicalize_colname
# ---------------------------------------------------------------------------
class TestCanonicalizeColname:
    """Validate column name normalization."""

    def test_chl_a_with_percent(self):
        assert canonicalize_colname("Chl-a (%)") == "chl_a"

    def test_sigma_t(self):
        # Note: .lower() runs before .replace("sigmaT", ...) in the source,
        # so the case-sensitive replace never fires. The CTD_ALIAS_MAP handles
        # the 'sigmat' → 'sigma_t' mapping instead.
        assert canonicalize_colname("sigmaT") == "sigmat"
        assert canonicalize_colname("SigmaT") == "sigmat"

    def test_mixed_case_spaces(self):
        # '/' is stripped by regex, 'mg' and 'L' merge to 'mgl'
        assert canonicalize_colname("  DO mg/L  ") == "do_mgl"

    def test_already_canonical(self):
        assert canonicalize_colname("temperature") == "temperature"


# ---------------------------------------------------------------------------
# derive_sample_dims
# ---------------------------------------------------------------------------
class TestDeriveSampleDims:
    """Validate sample dimension extraction."""

    def test_extract_all_fields(self):
        ids = pd.Series(["2024-06-O-s4", "2025-01-I-hm"])
        result = derive_sample_dims(ids)
        assert list(result["sample_year_month"]) == ["2024-06", "2025-01"]
        assert list(result["bay"]) == ["O", "I"]
        assert list(result["station_code"]) == ["s4", "hm"]

    def test_single_id(self):
        result = derive_sample_dims(pd.Series(["2024-04-M-s1"]))
        assert result.iloc[0]["bay"] == "M"


# ---------------------------------------------------------------------------
# normalize_genus_key
# ---------------------------------------------------------------------------
class TestNormalizeGenusKey:
    """Validate genus key cleaning."""

    def test_strips_whitespace(self):
        result = normalize_genus_key(pd.Series(["  Gyrodinium  ", "Oncaea"]))
        assert list(result) == ["Gyrodinium", "Oncaea"]

    def test_blank_to_na(self):
        result = normalize_genus_key(pd.Series(["", "nan", "None"]))
        assert all(pd.isna(result))

    def test_valid_preserved(self):
        result = normalize_genus_key(pd.Series(["Synechococcus"]))
        assert result.iloc[0] == "Synechococcus"
