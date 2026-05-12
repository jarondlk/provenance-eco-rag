"""Tests for schema/anchor_event.py — spatiotemporal anchor creation."""
from __future__ import annotations

import pandas as pd
import pytest

from schema.anchor_event import build_anchor_events, BAY_COORDS


class TestBuildAnchorEvents:
    """Validate anchor event construction from sample data."""

    def _make_sample_registry(self) -> pd.DataFrame:
        """Minimal sample registry with CTD + metagenome flags."""
        return pd.DataFrame({
            "sample_id": ["2024-04-O-s1", "2024-05-O-s4", "2024-06-I-hm"],
            "bay": ["O", "O", "I"],
            "station_code": ["s1", "s4", "hm"],
            "sample_year_month": ["2024-04", "2024-05", "2024-06"],
            "has_ctd": [True, True, False],
            "has_kraken": [True, True, True],
            "has_metaeuk": [True, True, True],
            "min_depth_m": [0.0, 0.0, pd.NA],
            "max_depth_m": [20.0, 25.0, pd.NA],
        })

    def test_sample_anchors_created(self):
        """Each sample_id produces one anchor event."""
        registry = self._make_sample_registry()
        result = build_anchor_events(registry)

        assert len(result) == 3
        assert all(result["event_id"].str.startswith("sample_"))

    def test_anchor_has_correct_fields(self):
        """Anchor events contain expected columns."""
        registry = self._make_sample_registry()
        result = build_anchor_events(registry)

        expected_cols = {"event_id", "time_start", "time_end", "lat", "lon",
                         "depth_min", "depth_max", "station_id", "sample_id",
                         "bay_code", "source_types"}
        assert expected_cols.issubset(set(result.columns))

    def test_bay_coordinates_assigned(self):
        """Onagawa Bay anchors get correct coordinates."""
        registry = self._make_sample_registry()
        result = build_anchor_events(registry)

        onagawa_rows = result[result["bay_code"] == "O"]
        assert all(onagawa_rows["lat"].notna())
        assert all(abs(onagawa_rows["lat"] - BAY_COORDS["O"][0]) < 0.01)

    def test_source_types_set(self):
        """Source types reflect CTD + metagenome presence."""
        registry = self._make_sample_registry()
        result = build_anchor_events(registry)

        # First sample has both CTD and metagenome
        row0 = result[result["sample_id"] == "2024-04-O-s1"].iloc[0]
        assert "ctd" in row0["source_types"]
        assert "metagenome" in row0["source_types"]

        # Third sample has only metagenome
        row2 = result[result["sample_id"] == "2024-06-I-hm"].iloc[0]
        assert "ctd" not in row2["source_types"]
        assert "metagenome" in row2["source_types"]

    def test_sst_anchors(self):
        """SST daily records produce separate SST anchors."""
        registry = self._make_sample_registry()
        sst_daily = pd.DataFrame({
            "date_jst": ["2024-04-10", "2024-04-11", "2024-04-12"],
            "mean_sst": [10.0, 10.5, 11.0],
        })

        result = build_anchor_events(registry, sst_daily=sst_daily)

        sst_anchors = result[result["event_id"].str.startswith("sst_")]
        sample_anchors = result[result["event_id"].str.startswith("sample_")]
        assert len(sst_anchors) == 3
        assert len(sample_anchors) == 3
        assert all(sst_anchors["source_types"] == "remote_sensing")

    def test_ctd_date_attached(self, sample_ctd_summary):
        """When CTD summary is provided, dates are attached to anchors."""
        registry = pd.DataFrame({
            "sample_id": ["2024-04-O-s1"],
            "bay": ["O"],
            "station_code": ["s1"],
            "sample_year_month": ["2024-04"],
            "has_ctd": [True],
            "has_kraken": [False],
            "has_metaeuk": [False],
            "min_depth_m": [0.0],
            "max_depth_m": [20.0],
        })

        result = build_anchor_events(registry, ctd_summary=sample_ctd_summary)
        assert result.iloc[0]["time_start"] == "2024-04-15"

    def test_empty_registry(self):
        """Empty registry produces empty output."""
        registry = pd.DataFrame(columns=[
            "sample_id", "bay", "station_code", "sample_year_month",
            "has_ctd", "has_kraken", "has_metaeuk", "min_depth_m", "max_depth_m",
        ])
        result = build_anchor_events(registry)
        assert result.empty
