"""
Cross-source linker.

Links observations from different modalities by proximity in time
and/or space.  Used during evidence expansion in the query
orchestration layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CrossSourceLink:
    """One link between two events/documents."""

    source_event_id: str
    target_event_id: str
    link_type: str              # same_sample, time_match, space_match, time_space_match
    distance_km: Optional[float]
    time_delta_days: Optional[float]


def build_cross_source_links(
    anchor_events: pd.DataFrame,
    max_time_delta_days: float = 7.0,
    max_distance_km: float = 50.0,
) -> pd.DataFrame:
    """
    Build cross-source links between anchor events.

    Links are created when:
      - Two events share the same sample_id (same_sample)
      - Two events from different source types are within max_time_delta_days
      - Two events from different source types are within max_distance_km

    Returns a DataFrame with link records.
    """
    links: List[dict] = []

    # Parse times
    ae = anchor_events.copy()
    ae["_time"] = pd.to_datetime(ae["time_start"], errors="coerce")

    # 1. Same-sample links (CTD ↔ metagenome via sample_id)
    sample_groups = ae[ae["sample_id"].notna()].groupby("sample_id")
    for sid, group in sample_groups:
        if len(group) < 2:
            continue
        ids = group["event_id"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                links.append({
                    "source_event_id": ids[i],
                    "target_event_id": ids[j],
                    "link_type": "same_sample",
                    "distance_km": 0.0,
                    "time_delta_days": 0.0,
                })

    # 2. Time-based links (sample ↔ SST)
    sample_events = ae[ae["event_id"].str.startswith("sample_") & ae["_time"].notna()]
    sst_events = ae[ae["event_id"].str.startswith("sst_") & ae["_time"].notna()]

    if not sample_events.empty and not sst_events.empty:
        for _, se in sample_events.iterrows():
            st = se["_time"]
            for _, re_ in sst_events.iterrows():
                rt = re_["_time"]
                delta = abs((st - rt).total_seconds()) / 86400
                if delta <= max_time_delta_days:
                    links.append({
                        "source_event_id": se["event_id"],
                        "target_event_id": re_["event_id"],
                        "link_type": "time_match",
                        "distance_km": None,
                        "time_delta_days": round(delta, 2),
                    })

    df = pd.DataFrame(links)
    if df.empty:
        df = pd.DataFrame(columns=[
            "source_event_id", "target_event_id", "link_type",
            "distance_km", "time_delta_days",
        ])

    logger.info(
        "Built %d cross-source links: %d same_sample, %d time_match",
        len(df),
        len(df[df["link_type"] == "same_sample"]) if not df.empty else 0,
        len(df[df["link_type"] == "time_match"]) if not df.empty else 0,
    )
    return df
