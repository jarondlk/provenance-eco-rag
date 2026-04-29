from __future__ import annotations

import os
import json
import html
import re
from pathlib import Path
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt

from engines import ChatParams, OllamaEngine, RagEngine
from rag.mock_store import load_jsonl, SourceDoc
from rag.bm25_retriever import BM25Retriever


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Onagawa Source Chat", layout="wide")
DEFAULT_SYSTEM = "You are an assistant for marine survey Q&A. Prefer precise, cited answers."


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# -----------------------------
# Helpers
# -----------------------------
def _stringify(x: Any) -> str:
    """Convert any value into a safe scalar string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)

    if hasattr(x, "__array__"):
        try:
            return json.dumps(np.asarray(x).tolist(), ensure_ascii=False, default=str)
        except Exception:
            return str(x)

    if isinstance(x, (dict, list, tuple, set)):
        return json.dumps(x, ensure_ascii=False, default=str)

    return str(x)


def _truncate(s: str, n: int = 180) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s


def _parse_iso_date(s: str) -> Optional[date]:
    """Parse YYYY-MM-DD to date; return None on failure."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def docs_to_rows(docs: List[SourceDoc]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in docs:
        text = _stringify(getattr(d, "text", ""))
        lat = _to_float(getattr(d, "lat", None))
        lon = _to_float(getattr(d, "lon", None))

        rows.append(
            {
                "id": _stringify(getattr(d, "id", "")),
                "title": _stringify(getattr(d, "title", "")),
                "date": _stringify(getattr(d, "date", "")),
                "location": _stringify(getattr(d, "location", "")),
                "url": _stringify(getattr(d, "url", "")),
                "lat": lat,
                "lon": lon,
                "text_len": len(text),
                "preview": _truncate(text, 180),
                "_full_text": text,
                "_date_obj": _parse_iso_date(_stringify(getattr(d, "date", ""))),
            }
        )
    return rows


def render_html_table(
    rows: List[Dict[str, Any]],
    columns: List[Tuple[str, str]],
    *,
    max_rows: int = 50,
    table_height_px: int = 420,
) -> None:
    """Render a scrollable HTML table."""
    shown = rows[:max_rows]

    thead = "".join(
        f"<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd;'>{html.escape(lbl)}</th>"
        for _, lbl in columns
    )

    def td(val: Any) -> str:
        return html.escape(_stringify(val))

    tbody_rows = []
    for r in shown:
        tds = "".join(
            f"<td style='vertical-align:top;padding:6px;border-bottom:1px solid #f0f0f0;'>{td(r.get(k,''))}</td>"
            for k, _ in columns
        )
        tbody_rows.append(f"<tr>{tds}</tr>")

    tbody = "".join(tbody_rows)
    table_html = f"""
    <div style="height:{table_height_px}px; overflow:auto; border:1px solid #e6e6e6; border-radius:8px;">
      <table style="border-collapse:collapse; width:100%; font-size:13px;">
        <thead style="position:sticky; top:0; background:#fafafa; z-index:1;">
          <tr>{thead}</tr>
        </thead>
        <tbody>
          {tbody}
        </tbody>
      </table>
    </div>
    <div style="margin-top:6px; color:#666; font-size:12px;">
      Showing {len(shown)} / {len(rows)} rows
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def filter_rows(
    rows: List[Dict[str, Any]],
    keyword: str,
    locations: List[str],
    date_range: Optional[Tuple[date, date]],
) -> List[Dict[str, Any]]:
    kw = (keyword or "").strip().lower()
    loc_set = set(locations or [])

    out: List[Dict[str, Any]] = []
    for r in rows:
        if loc_set and r.get("location", "") not in loc_set:
            continue

        if date_range is not None:
            d = r.get("_date_obj")
            if d is None:
                continue
            start, end = date_range
            if d < start or d > end:
                continue

        if kw:
            hay = (r.get("title", "") + " " + r.get("preview", "")).lower()
            if kw not in hay:
                continue

        out.append(r)
    return out


def rows_to_jsonl(rows: List[Dict[str, Any]]) -> str:
    out_lines = []
    for r in rows:
        r2 = {k: v for k, v in r.items() if not k.startswith("_")}
        out_lines.append(json.dumps(r2, ensure_ascii=False, default=str))
    return "\n".join(out_lines)


def render_sources_map(rows: List[Dict[str, Any]], *, height: int = 520) -> None:
    points = [
        {
            "lat": r["lat"],
            "lon": r["lon"],
            "title": r["title"],
            "date": r["date"],
            "location": r["location"],
            "id": r["id"],
        }
        for r in rows
        if isinstance(r.get("lat"), (int, float)) and isinstance(r.get("lon"), (int, float))
    ]

    if not points:
        st.info("No rows have lat/lon yet.")
        return

    try:
        st.map(points, latitude="lat", longitude="lon", height=height)
        return
    except Exception:
        pass

    center_lat = sum(p["lat"] for p in points) / len(points)
    center_lon = sum(p["lon"] for p in points) / len(points)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        id="sources",
        get_position="[lon, lat]",
        pickable=True,
        auto_highlight=True,
        get_radius=120,
    )

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=0,
        ),
        layers=[layer],
        tooltip={"text": "{title}\n{date}\n{location}\n{id}"},
    )

    st.pydeck_chart(deck, height=height)


# -----------------------------
# Ocean model helpers
# -----------------------------
def _parse_model_time_from_filename(path: str | Path) -> pd.Timestamp:
    name = Path(path).name
    m = re.search(r"(\d{8})_(\d{4})", name)
    if not m:
        raise ValueError(f"Could not parse timestamp from filename: {name}")
    return pd.to_datetime(f"{m.group(1)} {m.group(2)}", format="%Y%m%d %H%M")


@st.cache_data(show_spinner=False)
def list_model_files(model_root: str) -> List[str]:
    root = Path(model_root)
    if not root.exists():
        return []
    return [str(p) for p in sorted(root.rglob("*.nc"))]


@st.cache_data(show_spinner=False)
def load_model_point_timeseries(model_root: str, target_lat: float, target_lon: float) -> pd.DataFrame:
    files = list_model_files(model_root)
    rows: List[Dict[str, Any]] = []

    for fp in files:
        ds = xr.open_dataset(fp, engine="netcdf4", decode_times=False)
        try:
            point = ds["SST"].sel(
                latitude=target_lat,
                longitude=target_lon,
                method="nearest",
            ).isel(time=0, depth=0)

            nearest_lat = float(ds["latitude"].sel(latitude=target_lat, method="nearest").values)
            nearest_lon = float(ds["longitude"].sel(longitude=target_lon, method="nearest").values)

            t_utc = _parse_model_time_from_filename(fp)
            rows.append(
                {
                    "file": Path(fp).name,
                    "time_utc": t_utc,
                    "time_jst": t_utc + pd.Timedelta(hours=9),
                    "sst": float(point.values),
                    "nearest_lat": nearest_lat,
                    "nearest_lon": nearest_lon,
                }
            )
        finally:
            ds.close()

    if not rows:
        return pd.DataFrame(
            columns=["file", "time_utc", "time_jst", "sst", "nearest_lat", "nearest_lon"]
        )

    return pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_model_frame(model_root: str, file_index: int) -> Dict[str, Any]:
    files = list_model_files(model_root)
    if not files:
        raise FileNotFoundError(f"No .nc files found under {model_root}")

    fp = files[file_index]
    ds = xr.open_dataset(fp, engine="netcdf4", decode_times=False)
    try:
        sst = ds["SST"].isel(time=0, depth=0).values.astype("float32")
        lon = ds["longitude"].values.astype("float32")
        lat = ds["latitude"].values.astype("float32")
        t_utc = _parse_model_time_from_filename(fp)

        out = {
            "file": Path(fp).name,
            "time_utc": t_utc,
            "time_jst": t_utc + pd.Timedelta(hours=9),
            "sst": sst,
            "lon": lon,
            "lat": lat,
        }
        return out
    finally:
        ds.close()


@st.cache_data(show_spinner=False)
def estimate_model_color_range(model_root: str, stride: int = 24) -> Tuple[float, float]:
    files = list_model_files(model_root)
    if not files:
        return 0.0, 1.0

    sample_files = files[::max(1, stride)]
    mins: List[float] = []
    maxs: List[float] = []

    for fp in sample_files:
        ds = xr.open_dataset(fp, engine="netcdf4", decode_times=False)
        try:
            arr = ds["SST"].isel(time=0, depth=0).values.astype("float32")
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            mins.append(float(np.percentile(arr, 2)))
            maxs.append(float(np.percentile(arr, 98)))
        finally:
            ds.close()

    if not mins or not maxs:
        return 0.0, 1.0

    return float(np.min(mins)), float(np.max(maxs))


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

ollama_base_url = st.sidebar.text_input(
    "Ollama base URL",
    value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
)

data_path = st.sidebar.text_input("Data path", value="data/mock_sources.jsonl")
top_k_sources = st.sidebar.slider("Top-K sources (retrieval)", 1, 10, 4, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Ocean model")
model_root = st.sidebar.text_input("Ocean model path", value="onagawa_sst_subset")
default_model_lat = 38.42907415591698
default_model_lon = 141.4775733277202
model_lat = st.sidebar.number_input("Model point latitude", value=default_model_lat, format="%.6f")
model_lon = st.sidebar.number_input("Model point longitude", value=default_model_lon, format="%.6f")
model_stride = st.sidebar.slider("Model color-range stride", 1, 48, 24, 1)

with st.sidebar.expander("Model & sampling", expanded=True):
    engine = OllamaEngine(ollama_base_url)
    try:
        models = engine.list_models()
    except Exception:
        models = []

    model = st.selectbox(
        "Model",
        options=(models or ["llama3.1"]),
        index=0,
        placeholder="qwen2.5:14b-instruct",
    )

    temperature = st.slider("temperature", 0.0, 2.0, 0.0, 0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 0.9, 0.01)
    top_k = st.slider("top_k", 0, 100, 40, 1)
    repeat_penalty = st.slider("repeat_penalty", 0.8, 2.0, 1.1, 0.05)

with st.sidebar.expander("Limits", expanded=False):
    num_predict = st.slider("num_predict", 16, 4096, 700, 16)
    num_ctx = st.slider("num_ctx", 256, 32768, 8192, 256)
    keep_last_n_messages = st.slider("Keep last N messages", 2, 60, 20, 1)

with st.sidebar.expander("System prompt", expanded=False):
    system_prompt = st.text_area("system_prompt", value=DEFAULT_SYSTEM, height=140)

stream = st.sidebar.checkbox("Stream response", value=True)

if st.sidebar.button("Reset chat"):
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.rerun()


# -----------------------------
# Load corpus + retriever
# -----------------------------
try:
    docs: List[SourceDoc] = load_jsonl(data_path)
except Exception as e:
    st.error(f"Failed to load JSONL from '{data_path}': {e}")
    st.stop()

rows = docs_to_rows(docs)

try:
    retriever = BM25Retriever(docs)
except Exception as e:
    st.error(f"Failed to build retriever: {e}")
    st.stop()

rag_engine = RagEngine(engine, retriever)

params = ChatParams(
    model=model,
    system_prompt=system_prompt,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    repeat_penalty=repeat_penalty,
    num_predict=num_predict,
    num_ctx=num_ctx,
    seed=None,
    stream=stream,
    keep_last_n_messages=keep_last_n_messages,
)


# -----------------------------
# Main UI
# -----------------------------
st.title("Onagawa Source Chat")
st.caption("Prelim tested model: qwen2.5:14b-instruct")

tab_chat, tab_sources, tab_stats, tab_model = st.tabs(
    ["Chat", "Sources DB", "Stats", "Ocean Model"]
)


# =============================
# TAB: Chat
# =============================
with tab_chat:
    col_chat, col_right = st.columns([2.2, 1.0], gap="large", border=True)

    with col_chat:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if st.session_state.pending_prompt:
            user_text = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""

                retrieved = retriever.search(user_text, k=top_k_sources)
                with st.expander("Retrieved sources", expanded=False):
                    for r in retrieved:
                        d = r.doc
                        st.markdown(
                            f"**[{_stringify(d.id)}] {_stringify(d.title)}**  \n"
                            f"Date: {_stringify(d.date)} | Location: {_stringify(d.location)}  \n"
                            f"URL: {_stringify(d.url)}"
                        )
                        st.markdown(f"Score: `{r.score:.4f}`")
                        st.markdown(f"> {_stringify(r.excerpt)}")

                try:
                    for chunk in rag_engine.chat(st.session_state.messages, params, k=top_k_sources):
                        full += chunk
                        placeholder.markdown(full)
                except Exception as e:
                    st.error(f"Ollama call failed: {e}")
                    full = full or "Error calling Ollama."

                st.session_state.messages.append({"role": "assistant", "content": full})

    with col_right:
        st.subheader("Corpus quick view")
        render_html_table(
            rows,
            columns=[
                ("id", "ID"),
                ("title", "Title"),
                ("date", "Date"),
                ("location", "Location"),
                ("preview", "Preview"),
            ],
            max_rows=25,
            table_height_px=720,
        )


# =============================
# TAB: Sources DB
# =============================
with tab_sources:
    st.subheader("Sources DB Explorer")

    c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
    with c1:
        q = st.text_input("Keyword filter (title/preview)", value="")

    with c2:
        locs = sorted({r.get("location", "") for r in rows if r.get("location", "")})
        selected_locs = st.multiselect("Location", options=locs, default=locs)

    with c3:
        parsed_dates = sorted([r["_date_obj"] for r in rows if r.get("_date_obj") is not None])
        if parsed_dates:
            dmin, dmax = parsed_dates[0], parsed_dates[-1]
            picked = st.date_input("Date range", value=(dmin, dmax))
            date_range = (picked[0], picked[1]) if isinstance(picked, (tuple, list)) and len(picked) == 2 else None
        else:
            st.caption("No parseable dates (YYYY-MM-DD) found.")
            date_range = None

    filtered_sources = filter_rows(rows, q, selected_locs, date_range)

    st.markdown("### Table")
    render_html_table(
        filtered_sources,
        columns=[
            ("id", "ID"),
            ("title", "Title"),
            ("date", "Date"),
            ("location", "Location"),
            ("url", "URL"),
            ("text_len", "Len"),
            ("preview", "Preview"),
        ],
        max_rows=200,
        table_height_px=520,
    )

    st.markdown("### Source detail")
    ids = [r["id"] for r in filtered_sources] if filtered_sources else [r["id"] for r in rows]

    if ids:
        selected_id = st.selectbox("Select a source ID", options=ids, index=0)
        r = next((x for x in rows if x["id"] == selected_id), None)
        if r:
            st.markdown(f"**[{r['id']}] {r['title']}**")
            st.markdown(
                f"- **Date:** {r['date']}\n"
                f"- **Location:** {r['location']}\n"
                f"- **URL:** {r['url']}"
            )
            st.markdown("**Text**")
            st.write(r["_full_text"])
    else:
        st.info("No sources match the current filters.")

    st.download_button(
        "Download sources JSONL",
        data=rows_to_jsonl(rows),
        file_name="sources_export.jsonl",
        mime="application/x-ndjson",
    )

    st.markdown("### Map")
    render_sources_map(filtered_sources, height=520)


# =============================
# TAB: Stats
# =============================
with tab_stats:
    st.subheader("Corpus stats")

    n_docs = len(rows)
    loc_counts = Counter(r.get("location", "") for r in rows if r.get("location", ""))
    avg_len = int(sum(r.get("text_len", 0) for r in rows) / n_docs) if n_docs else 0

    c1, c2, c3 = st.columns(3, border=True)
    c1.metric("Documents", n_docs)
    c2.metric("Locations", len(loc_counts))
    c3.metric("Avg text length", avg_len)

    st.markdown("### Map")
    render_sources_map(rows, height=520)

    st.markdown("### Docs by location")
    if loc_counts:
        for loc, cnt in loc_counts.most_common():
            st.markdown(f"- **{loc}**: {cnt}")
    else:
        st.caption("No location values found.")

    st.markdown("### Docs by month")
    month_counts = Counter()
    for r in rows:
        d = r.get("_date_obj")
        if d is not None:
            month_counts[f"{d.year:04d}-{d.month:02d}"] += 1

    if month_counts:
        for m in sorted(month_counts.keys()):
            st.markdown(f"- **{m}**: {month_counts[m]}")
    else:
        st.caption("No parseable dates to compute monthly counts.")

    st.subheader("Abstract Temp - DELETE LATER")
    abstract = """
        Coastal fisheries and biodiversity programs generate high-value observations, but in practice those data are hard to use: they are fragmented across logs and reports, inconsistently structured, and require domain + database expertise to answer simple operational questions (e.g., “what was most abundant in Onagawa in August?”). This creates a gap between data collection and actionable ecological insight, and it slows collaboration across field teams, analysts, and researchers.
    We present Onagawa Source Chat, a provenance-preserving question-answering application that combines retrieval-augmented generation (RAG) with an LLM served via Ollama to enable fast, source-backed exploration of the ANEMONE fisheries database. The system indexes survey and fishery documents with minimal metadata (ID, title, date, location, URL, text). For each query, a BM25 retriever selects top-K relevant sources and passes scored excerpts to the LLM, producing grounded answers accompanied by inspectable citations. A Streamlit interface supports streaming responses, adjustable decoding/context parameters, a searchable “Sources DB” (filters by keyword/location/date), and corpus-level statistics for quality checks.
    """
    st.caption(abstract)


# =============================
# TAB: Ocean Model
# =============================
with tab_model:
    st.subheader("Onagawa Ocean Model")

    model_files = list_model_files(model_root)

    c1, c2, c3 = st.columns(3, border=True)
    c1.metric("Model files", len(model_files))
    c2.metric("Point lat", f"{model_lat:.4f}")
    c3.metric("Point lon", f"{model_lon:.4f}")

    if not model_files:
        st.warning(f"No NetCDF files found under: {Path(model_root).resolve()}")
    else:
        ts = load_model_point_timeseries(model_root, model_lat, model_lon)

        if ts.empty:
            st.info("No model rows loaded.")
        else:
            start_jst = ts["time_jst"].min()
            end_jst = ts["time_jst"].max()

            c4, c5, c6, c7 = st.columns(4, border=True)
            c4.metric("Frames", len(ts))
            c5.metric("Min SST", f"{ts['sst'].min():.2f}")
            c6.metric("Max SST", f"{ts['sst'].max():.2f}")
            c7.metric("Mean SST", f"{ts['sst'].mean():.2f}")

            st.caption(
                f"Nearest model grid point: "
                f"({ts.loc[0, 'nearest_lat']:.4f}, {ts.loc[0, 'nearest_lon']:.4f}) | "
                f"Time range: {start_jst} to {end_jst} JST"
            )

            st.markdown("### Point SST time series")
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            ax_ts.plot(ts["time_jst"], ts["sst"])
            ax_ts.set_xlabel("Time (JST)")
            ax_ts.set_ylabel("SST")
            ax_ts.set_title("Onagawa SST time series")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            with st.expander("Show point time-series table", expanded=False):
                st.dataframe(ts, use_container_width=True, height=320)

            st.download_button(
                "Download point SST CSV",
                data=ts.to_csv(index=False),
                file_name="onagawa_model_point_timeseries.csv",
                mime="text/csv",
            )

        st.markdown("### Regional SST map")
        file_times_jst = [(_parse_model_time_from_filename(fp) + pd.Timedelta(hours=9)) for fp in model_files]
        default_idx = len(model_files) - 1

        frame_idx = st.slider("Frame index", 0, len(model_files) - 1, default_idx, 1)
        frame = load_model_frame(model_root, frame_idx)

        global_vmin, global_vmax = estimate_model_color_range(model_root, stride=model_stride)

        fig_map, ax_map = plt.subplots(figsize=(7, 7))
        im = ax_map.imshow(
            frame["sst"],
            origin="lower",
            extent=[
                float(np.min(frame["lon"])),
                float(np.max(frame["lon"])),
                float(np.min(frame["lat"])),
                float(np.max(frame["lat"])),
            ],
            vmin=global_vmin,
            vmax=global_vmax,
            aspect="auto",
            cmap="viridis",
        )
        ax_map.scatter(model_lon, model_lat, marker="x", s=100)
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title(f"SST map {frame['time_jst']} JST")
        fig_map.colorbar(im, ax=ax_map, label="SST")
        st.pyplot(fig_map, clear_figure=True)

        st.caption(f"File: {frame['file']}")

        with st.expander("Model file timestamps", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    {
                        "index": list(range(len(model_files))),
                        "time_jst": file_times_jst,
                        "file": [Path(fp).name for fp in model_files],
                    }
                ),
                use_container_width=True,
                height=320,
            )


# -----------------------------
# Global pinned chat input
# -----------------------------
prompt = st.chat_input('Ask: "During August in Onagawa, which species is most abundant?"')
if prompt:
    st.session_state.pending_prompt = prompt
    st.rerun()