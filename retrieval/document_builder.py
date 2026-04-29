"""
Retrieval document builder.

Converts canonical tables into unified, LLM-facing text chunks with
provenance metadata.  Each document is a self-contained narrative
paragraph that the retriever can score and the LLM can cite.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDocument:
    """One retrieval-ready chunk for the LLM."""

    doc_id: str
    source_type: str            # "ctd", "metagenome", "remote_sensing"
    sample_id: Optional[str]
    event_id: Optional[str]
    time: Optional[str]         # ISO date
    lat: Optional[float]
    lon: Optional[float]
    bay: Optional[str]
    station: Optional[str]
    title: str
    text: str                   # the LLM-facing narrative
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bay code → human-readable name
# ---------------------------------------------------------------------------
BAY_NAMES = {
    "O": "Onagawa Bay",
    "I": "Ishinomaki Bay",
    "M": "Matsushima Bay",
}

BAY_COORDS = {
    "O": (38.4449, 141.4474),
    "I": (38.4127, 141.3040),
    "M": (38.3540, 141.0630),
}


# =====================================================================
# CTD cast summaries
# =====================================================================
def build_ctd_docs(
    ctd_summary: pd.DataFrame,
    ctd_profile: Optional[pd.DataFrame] = None,
) -> List[RetrievalDocument]:
    """
    Build one retrieval document per CTD cast (sample_id).
    """
    docs: List[RetrievalDocument] = []

    for _, row in ctd_summary.iterrows():
        sid = row.get("sample_id", "")
        if pd.isna(sid) or not sid:
            continue

        # Parse bay and station from sample_id
        parts = str(sid).split("-")
        bay = parts[2] if len(parts) >= 3 else None
        station = parts[3] if len(parts) >= 4 else None
        year_month = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else None
        bay_name = BAY_NAMES.get(bay, bay or "unknown bay")
        lat, lon = BAY_COORDS.get(bay, (None, None))

        date_str = ""
        if pd.notna(row.get("ctd_date")):
            d = row["ctd_date"]
            date_str = str(d.date()) if hasattr(d, "date") else str(d)

        # Build narrative
        lines = [f"CTD cast at {bay_name}, station {station}, sample {sid}."]
        if date_str:
            lines[0] = f"CTD cast on {date_str} at {bay_name}, station {station}, sample {sid}."

        n_depths = row.get("n_depth_points", 0)
        min_d = row.get("min_depth_m")
        max_d = row.get("max_depth_m")
        if pd.notna(n_depths) and pd.notna(max_d):
            lines.append(f"Profile: {int(n_depths)} depth points from {_fmt(min_d)}m to {_fmt(max_d)}m.")

        # Environmental variables
        var_lines = []
        for var_name, label, unit in [
            ("temperature", "Temperature", "°C"),
            ("salinity", "Salinity", "PSU"),
            ("do_percent", "Dissolved oxygen", "%"),
            ("do_mg_l", "DO", "mg/L"),
            ("chl_a", "Chlorophyll-a", "µg/L"),
            ("turbidity", "Turbidity", "NTU"),
            ("ph", "pH", ""),
            ("sigma_t", "Sigma-T", "kg/m³"),
        ]:
            surf = row.get(f"surface_{var_name}")
            bot = row.get(f"bottom_{var_name}")
            mean = row.get(f"mean_{var_name}")
            if pd.notna(surf) and pd.notna(bot):
                var_lines.append(
                    f"{label}: surface={_fmt(surf)}{unit}, bottom={_fmt(bot)}{unit}, mean={_fmt(mean)}{unit}"
                )
            elif pd.notna(mean):
                var_lines.append(f"{label}: mean={_fmt(mean)}{unit}")

        if var_lines:
            lines.append("Measurements: " + "; ".join(var_lines) + ".")

        text = " ".join(lines)
        title = f"CTD cast {sid}"
        if date_str:
            title += f" ({date_str})"

        docs.append(RetrievalDocument(
            doc_id=f"ctd_{sid}",
            source_type="ctd",
            sample_id=sid,
            event_id=f"sample_{sid}",
            time=date_str or None,
            lat=lat,
            lon=lon,
            bay=bay,
            station=station,
            title=title,
            text=text,
        ))

    logger.info("Built %d CTD retrieval documents", len(docs))
    return docs


# =====================================================================
# Metagenome sample summaries
# =====================================================================
def build_metagenome_docs(
    sample_context: pd.DataFrame,
) -> List[RetrievalDocument]:
    """
    Build one retrieval document per metagenome sample from the
    multisource context table.
    """
    docs: List[RetrievalDocument] = []

    for _, row in sample_context.iterrows():
        sid = row.get("sample_id", "")
        if pd.isna(sid) or not sid:
            continue

        # Only include samples that have metagenome data
        has_kraken = row.get("has_kraken", False)
        has_metaeuk = row.get("has_metaeuk", False)
        if not has_kraken and not has_metaeuk:
            continue

        parts = str(sid).split("-")
        bay = parts[2] if len(parts) >= 3 else None
        station = parts[3] if len(parts) >= 4 else None
        year_month = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else None
        bay_name = BAY_NAMES.get(bay, bay or "unknown bay")
        lat, lon = BAY_COORDS.get(bay, (None, None))

        lines = [f"Metagenome sample {sid} from {bay_name}, station {station} ({year_month})."]

        # QC info
        n_runs = row.get("n_runs")
        reads = row.get("sum_reads_gt1kb")
        if pd.notna(n_runs):
            lines.append(f"Sequencing: {int(n_runs)} run(s)")
            if pd.notna(reads):
                lines[-1] += f", {int(reads):,} reads >1kb"
            lines[-1] += "."

        # Top Kraken genera
        kraken_json = row.get("top_genus_10_json")
        if pd.notna(kraken_json) and isinstance(kraken_json, str) and kraken_json.strip():
            try:
                taxa = json.loads(kraken_json)
                if taxa:
                    top_list = [
                        f"{t['genus']} ({t['abundance_value']:.2f}%)"
                        + (f" [{t.get('upper_group_label', '')}]" if t.get("upper_group_label") else "")
                        for t in taxa[:5]
                    ]
                    lines.append("Top Kraken genera: " + ", ".join(top_list) + ".")
            except (json.JSONDecodeError, KeyError):
                pass

        # Top MetaEuk genera
        metaeuk_col = None
        for c in sample_context.columns:
            if "metaeuk" in c.lower() or c == "top_genus_10_json_y":
                # Handle suffix from merge
                pass
        # Try common column names from the merge
        for col_candidate in ["top_genus_10_json_y", "top_genus_10_json"]:
            if col_candidate in row.index and col_candidate != "top_genus_10_json":
                metaeuk_json = row.get(col_candidate)
                if pd.notna(metaeuk_json) and isinstance(metaeuk_json, str):
                    try:
                        taxa = json.loads(metaeuk_json)
                        if taxa:
                            top_list = [
                                f"{t['genus']} ({t['abundance_value']:.2f}%)"
                                for t in taxa[:5]
                            ]
                            lines.append("Top MetaEuk genera: " + ", ".join(top_list) + ".")
                    except (json.JSONDecodeError, KeyError):
                        pass
                break

        # Top upper groups
        upper_col = row.get("top_upper_group_10_json")
        if pd.notna(upper_col) and isinstance(upper_col, str):
            try:
                groups = json.loads(upper_col)
                if groups:
                    top_list = [
                        f"{g['upper_group']} ({g['abundance_value']:.2f}%)"
                        for g in groups[:5]
                    ]
                    lines.append("Dominant taxonomic groups: " + ", ".join(top_list) + ".")
            except (json.JSONDecodeError, KeyError):
                pass

        text = " ".join(lines)
        title = f"Metagenome {sid} ({year_month}, {bay_name})"

        # Determine time from CTD or year-month
        time_str = None
        ctd_date = row.get("ctd_date")
        if pd.notna(ctd_date):
            time_str = str(ctd_date.date()) if hasattr(ctd_date, "date") else str(ctd_date)
        elif year_month:
            time_str = f"{year_month}-01"

        docs.append(RetrievalDocument(
            doc_id=f"meta_{sid}",
            source_type="metagenome",
            sample_id=sid,
            event_id=f"sample_{sid}",
            time=time_str,
            lat=lat,
            lon=lon,
            bay=bay,
            station=station,
            title=title,
            text=text,
        ))

    logger.info("Built %d metagenome retrieval documents", len(docs))
    return docs


# =====================================================================
# SST daily summaries
# =====================================================================
def build_sst_docs(
    sst_daily: pd.DataFrame,
    sst_point: Optional[pd.DataFrame] = None,
    target_lat: float = 38.4291,
    target_lon: float = 141.4776,
) -> List[RetrievalDocument]:
    """
    Build one retrieval document per SST day.
    """
    docs: List[RetrievalDocument] = []

    # Pre-index point data by date if available
    point_by_date: Dict[str, List[float]] = {}
    if sst_point is not None and not sst_point.empty:
        for _, pr in sst_point.iterrows():
            d = str(pr["time_jst"].date()) if hasattr(pr["time_jst"], "date") else str(pr["time_jst"])[:10]
            point_by_date.setdefault(d, []).append(float(pr["sst"]))

    for _, row in sst_daily.iterrows():
        d = str(row["date_jst"])
        mean_sst = row.get("mean_sst")
        min_sst = row.get("min_sst")
        max_sst = row.get("max_sst")
        n_files = row.get("n_files", 0)

        lines = [f"Satellite SST for the Onagawa region on {d}."]
        lines.append(
            f"Regional statistics: mean={_fmt(mean_sst)}°C, "
            f"min={_fmt(min_sst)}°C, max={_fmt(max_sst)}°C "
            f"(from {int(n_files)} hourly observations)."
        )

        # Add point SST if available
        if d in point_by_date:
            point_vals = point_by_date[d]
            mean_pt = np.mean(point_vals)
            lines.append(
                f"At monitoring station ({target_lat:.4f}°N, {target_lon:.4f}°E): "
                f"daily mean SST = {mean_pt:.2f}°C ({len(point_vals)} measurements)."
            )

        text = " ".join(lines)
        title = f"Satellite SST {d}"

        docs.append(RetrievalDocument(
            doc_id=f"sst_{d}",
            source_type="remote_sensing",
            sample_id=None,
            event_id=f"sst_{d}",
            time=d,
            lat=target_lat,
            lon=target_lon,
            bay=None,
            station=None,
            title=title,
            text=text,
        ))

    logger.info("Built %d SST retrieval documents", len(docs))
    return docs


# =====================================================================
# Unified builder
# =====================================================================
def build_all_documents(
    ctd_summary: pd.DataFrame,
    sample_context: pd.DataFrame,
    sst_daily: Optional[pd.DataFrame] = None,
    sst_point: Optional[pd.DataFrame] = None,
) -> List[RetrievalDocument]:
    """Build all retrieval documents from all sources."""
    docs = []
    docs.extend(build_ctd_docs(ctd_summary))
    docs.extend(build_metagenome_docs(sample_context))
    if sst_daily is not None:
        docs.extend(build_sst_docs(sst_daily, sst_point))
    logger.info("Total retrieval documents: %d", len(docs))
    return docs


def documents_to_dataframe(docs: List[RetrievalDocument]) -> pd.DataFrame:
    """Convert retrieval documents to a DataFrame."""
    rows = []
    for d in docs:
        rows.append({
            "doc_id": d.doc_id,
            "source_type": d.source_type,
            "sample_id": d.sample_id,
            "event_id": d.event_id,
            "time": d.time,
            "lat": d.lat,
            "lon": d.lon,
            "bay": d.bay,
            "station": d.station,
            "title": d.title,
            "text": d.text,
        })
    return pd.DataFrame(rows)


def documents_to_jsonl(docs: List[RetrievalDocument], path) -> None:
    """Write documents as JSONL (compatible with existing mock_store)."""
    import json
    from pathlib import Path

    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for d in docs:
            obj = {
                "id": d.doc_id,
                "title": d.title,
                "date": d.time or "",
                "location": BAY_NAMES.get(d.bay, d.bay or "Onagawa region"),
                "url": "",
                "text": d.text,
                "lat": d.lat,
                "lon": d.lon,
                "source_type": d.source_type,
                "sample_id": d.sample_id,
                "event_id": d.event_id,
            }
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d documents to %s", len(docs), p)


# =====================================================================
# Helpers
# =====================================================================
def _fmt(v, decimals: int = 2) -> str:
    """Format a numeric value, returning '–' for NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.{decimals}f}"
