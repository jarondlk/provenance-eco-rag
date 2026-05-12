"""
Shared anchor_event model.

An anchor event ties together observations across modalities by
shared spatiotemporal coordinates.  Every CTD cast, metagenome sample,
and remote-sensing observation is linked to one or more anchor events.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnchorEvent:
    """One spatiotemporal observation event."""

    event_id: str
    time_start: Optional[str]   # ISO date or datetime
    time_end: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    depth_min: Optional[float]
    depth_max: Optional[float]
    station_id: Optional[str]   # e.g. "s1", "s4", "hm"
    sample_id: Optional[str]    # canonical sample_id
    bay_code: Optional[str]     # O / I / M
    source_types: List[str] = field(default_factory=list)  # ["ctd", "metagenome", ...]
    provenance_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bay code → approximate coordinates (best-effort defaults)
# ---------------------------------------------------------------------------
BAY_COORDS = {
    "O": (38.4449, 141.4474),   # Onagawa Bay
    "I": (38.4127, 141.3040),   # Ishinomaki Bay (approximate)
    "M": (38.3540, 141.0630),   # Matsushima Bay (approximate)
}


def build_anchor_events(
    sample_registry: pd.DataFrame,
    ctd_summary: Optional[pd.DataFrame] = None,
    sst_daily: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create anchor events from the sample registry.

    Each unique sample_id becomes one anchor event.  When CTD summary
    data is available, depth range and date are attached.  SST daily
    records create separate anchor events (one per day).

    Returns a DataFrame with anchor event rows.
    """
    rows: List[Dict[str, Any]] = []

    # --- Sample-based anchors (CTD + metagenome) ---
    for _, sr in sample_registry.iterrows():
        sid = sr.get("sample_id")
        if pd.isna(sid):
            continue

        bay = sr.get("bay", None)
        lat, lon = BAY_COORDS.get(bay, (None, None)) if bay else (None, None)

        # Time from CTD date or sample year-month
        time_start = None
        if ctd_summary is not None:
            ctd_row = ctd_summary.loc[ctd_summary["sample_id"] == sid]
            if not ctd_row.empty:
                d = ctd_row.iloc[0].get("ctd_date")
                if pd.notna(d):
                    time_start = str(d.date()) if hasattr(d, "date") else str(d)

        if time_start is None:
            ym = sr.get("sample_year_month")
            if pd.notna(ym):
                time_start = f"{ym}-01"

        depth_min = sr.get("min_depth_m") if pd.notna(sr.get("min_depth_m")) else None
        depth_max = sr.get("max_depth_m") if pd.notna(sr.get("max_depth_m")) else None

        source_types = []
        if sr.get("has_ctd"):
            source_types.append("ctd")
        if sr.get("has_kraken") or sr.get("has_metaeuk"):
            source_types.append("metagenome")

        rows.append({
            "event_id": f"sample_{sid}",
            "time_start": time_start,
            "time_end": time_start,
            "lat": lat,
            "lon": lon,
            "depth_min": float(depth_min) if depth_min is not None else None,
            "depth_max": float(depth_max) if depth_max is not None else None,
            "station_id": sr.get("station_code"),
            "sample_id": sid,
            "bay_code": bay,
            "source_types": ",".join(source_types),
        })

    # --- SST-based anchors (one per day) ---
    if sst_daily is not None and not sst_daily.empty:
        for _, dr in sst_daily.iterrows():
            d = dr.get("date_jst")
            rows.append({
                "event_id": f"sst_{d}",
                "time_start": str(d),
                "time_end": str(d),
                "lat": None,   # regional, not a point
                "lon": None,
                "depth_min": 0.0,
                "depth_max": 0.0,
                "station_id": None,
                "sample_id": None,
                "bay_code": None,
                "source_types": "remote_sensing",
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info("Built %d anchor events (%d sample-based, %d SST-based)",
                    len(df),
                    len(df[df["event_id"].str.startswith("sample_")]),
                    len(df[df["event_id"].str.startswith("sst_")]))
    else:
        logger.info("Built 0 anchor events")
    return df
