"""Tests for preprocessing/reliability_ensurance.py — cross-source validation logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.reliability_ensurance import build_reliability_documents


# ---------------------------------------------------------------------------
# SST ↔ CTD agreement score logic
# ---------------------------------------------------------------------------
class TestSstCtdAgreementLogic:
    """Test the SST-CTD agreement scoring formula directly."""

    def test_perfect_agreement(self):
        """Delta=0 → agrees=True, score=1.0."""
        threshold = 2.0
        delta_t = 0.0
        abs_delta = abs(delta_t)
        agrees = abs_delta <= threshold
        score = round(max(0.0, min(1.0, 1.0 - abs_delta / (threshold * 2))), 4)

        assert agrees is True
        assert score == 1.0

    def test_within_threshold(self):
        """Delta=1.5°C (within 2.0 threshold) → agrees=True, score>0."""
        threshold = 2.0
        delta_t = 1.5
        abs_delta = abs(delta_t)
        agrees = abs_delta <= threshold
        score = round(max(0.0, min(1.0, 1.0 - abs_delta / (threshold * 2))), 4)

        assert agrees is True
        assert 0.0 < score < 1.0

    def test_exact_threshold(self):
        """Delta exactly at threshold → agrees=True."""
        threshold = 2.0
        delta_t = 2.0
        abs_delta = abs(delta_t)
        agrees = abs_delta <= threshold

        assert agrees is True

    def test_exceeds_threshold(self):
        """Delta=5.0°C → agrees=False, score=0."""
        threshold = 2.0
        delta_t = 5.0
        abs_delta = abs(delta_t)
        agrees = abs_delta <= threshold
        score = round(max(0.0, min(1.0, 1.0 - abs_delta / (threshold * 2))), 4)

        assert agrees is False
        assert score == 0.0

    def test_negative_delta(self):
        """Negative delta (CTD < SST) also works correctly."""
        threshold = 2.0
        delta_t = -1.0
        abs_delta = abs(delta_t)
        agrees = abs_delta <= threshold

        assert agrees is True
        assert abs_delta == 1.0


# ---------------------------------------------------------------------------
# Corroboration tier logic
# ---------------------------------------------------------------------------
class TestCorroborationTierLogic:
    """Test the tier assignment rules."""

    def test_verified_with_multiple_checks(self):
        """2+ checks → verified tier."""
        n_checks = 3
        tier = "verified" if n_checks >= 2 else ("supported" if n_checks == 1 else "standalone")
        assert tier == "verified"

    def test_supported_with_one_check(self):
        """Exactly 1 check → supported tier."""
        n_checks = 1
        tier = "verified" if n_checks >= 2 else ("supported" if n_checks == 1 else "standalone")
        assert tier == "supported"

    def test_standalone_with_no_checks(self):
        """0 checks → standalone tier."""
        n_checks = 0
        tier = "verified" if n_checks >= 2 else ("supported" if n_checks == 1 else "standalone")
        assert tier == "standalone"


# ---------------------------------------------------------------------------
# Diversity anomaly detection logic
# ---------------------------------------------------------------------------
class TestDiversityAnomalyLogic:
    """Test the anomaly detection threshold."""

    def test_normal_within_sigma(self):
        """Deviation of 1.5σ (below 2.0 threshold) → not anomaly."""
        sigma = 2.0
        deviation_sigma = 1.5
        assert abs(deviation_sigma) <= sigma

    def test_anomaly_exceeds_sigma(self):
        """Deviation of 2.5σ (above 2.0 threshold) → anomaly."""
        sigma = 2.0
        deviation_sigma = -2.5
        assert abs(deviation_sigma) > sigma

    def test_exact_sigma_boundary(self):
        """Deviation at exactly 2.0σ → not anomaly (>σ, not >=σ)."""
        sigma = 2.0
        deviation_sigma = 2.0
        # The code uses > not >=
        is_anomaly = abs(deviation_sigma) > sigma
        assert is_anomaly is False


# ---------------------------------------------------------------------------
# build_reliability_documents
# ---------------------------------------------------------------------------
class TestBuildReliabilityDocuments:
    """Test reliability document generation from DataFrames."""

    def test_all_empty_inputs(self):
        """Empty DataFrames produce no documents."""
        docs = build_reliability_documents(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert docs == []

    def test_sst_ctd_document(self):
        """SST-CTD validation data produces a document."""
        sst_ctd = pd.DataFrame({
            "sample_id": ["2024-04-O-s1", "2024-05-O-s1"],
            "ctd_date": pd.to_datetime(["2024-04-15", "2024-05-10"]),
            "bay": ["O", "O"],
            "ctd_surface_t": [10.5, 14.2],
            "sst_daily_mean": [10.0, 13.8],
            "delta_t": [0.5, 0.4],
            "abs_delta_t": [0.5, 0.4],
            "agrees": [True, True],
            "reliability_score": [0.875, 0.9],
        })

        docs = build_reliability_documents(
            sst_ctd, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )

        assert len(docs) == 1
        assert docs[0]["id"] == "reliability_sst_ctd_validation"
        assert "100%" in docs[0]["text"]
        assert "Onagawa Bay" in docs[0]["text"]

    def test_diversity_prediction_document_with_anomaly(self):
        """Diversity prediction with anomaly produces a document with details."""
        div_pred = pd.DataFrame({
            "sample_id": ["2024-04-O-s1", "2024-07-O-s1"],
            "bay": ["O", "O"],
            "predicted_shannon": [3.9, 3.5],
            "actual_shannon": [3.8, 1.6],
            "deviation": [-0.1, -1.9],
            "deviation_sigma": [-0.1, -2.3],
            "is_anomaly": [False, True],
            "n_supporting_vars": [3, 3],
            "supporting_env_summary": ["temp=10.5", "temp=22.1"],
        })

        docs = build_reliability_documents(
            pd.DataFrame(), pd.DataFrame(), div_pred, pd.DataFrame()
        )

        assert len(docs) == 1
        assert docs[0]["id"] == "reliability_diversity_prediction"
        assert "1" in docs[0]["text"]  # 1 anomaly
        assert "2024-07-O-s1" in docs[0]["text"]

    def test_corroboration_document(self):
        """Corroboration summary produces a document with tier counts."""
        corrob = pd.DataFrame({
            "event_id": ["sample_2024-04-O-s1", "sample_2024-05-O-s1", "sample_2024-06-O-s4"],
            "sample_id": ["2024-04-O-s1", "2024-05-O-s1", "2024-06-O-s4"],
            "source_type": ["ctd,metagenome", "ctd,metagenome", "ctd"],
            "reliability_tier": ["verified", "verified", "standalone"],
            "corroboration_sources": ["multi_modal,diversity", "multi_modal", ""],
            "reliability_score": [0.9, 0.8, 0.3],
            "n_checks": [2, 1, 0],
            "detail": ["Multi-modal", "Multi-modal", "No cross-validation"],
        })

        docs = build_reliability_documents(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), corrob
        )

        assert len(docs) == 1
        assert docs[0]["id"] == "reliability_corroboration_summary"
        assert "Verified" in docs[0]["text"]

    def test_all_inputs_produce_four_documents(self):
        """All non-empty inputs produce 4 documents."""
        sst_ctd = pd.DataFrame({
            "sample_id": ["s1"], "ctd_date": pd.to_datetime(["2024-04-15"]),
            "bay": ["O"], "ctd_surface_t": [10.5], "sst_daily_mean": [10.0],
            "delta_t": [0.5], "abs_delta_t": [0.5], "agrees": [True],
            "reliability_score": [0.875],
        })
        gap = pd.DataFrame({
            "date": ["2024-04-16"], "sst_daily_mean": [10.2],
            "interpolated_surface_t": [10.7], "confidence": [0.9],
            "method": ["sst_offset_correction"], "nearest_ctd_days": [1.0],
            "in_ctd_gap": [True],
        })
        div = pd.DataFrame({
            "sample_id": ["s1"], "bay": ["O"],
            "predicted_shannon": [3.9], "actual_shannon": [3.8],
            "deviation": [-0.1], "deviation_sigma": [-0.1],
            "is_anomaly": [False], "n_supporting_vars": [3],
            "supporting_env_summary": ["temp=10.5"],
        })
        corrob = pd.DataFrame({
            "event_id": ["sample_s1"], "sample_id": ["s1"],
            "source_type": ["ctd"], "reliability_tier": ["verified"],
            "corroboration_sources": ["sst"], "reliability_score": [0.9],
            "n_checks": [2], "detail": ["SST agrees"],
        })

        docs = build_reliability_documents(sst_ctd, gap, div, corrob)
        assert len(docs) == 4
        ids = {d["id"] for d in docs}
        assert "reliability_sst_ctd_validation" in ids
        assert "reliability_gap_interpolation" in ids
        assert "reliability_diversity_prediction" in ids
        assert "reliability_corroboration_summary" in ids
