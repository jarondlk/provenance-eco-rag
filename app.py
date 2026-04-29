"""
Onagawa Source Chat v2 – Provenance-Aware Marine RAG
"""
from __future__ import annotations
import os, sys, json, html, re
from pathlib import Path
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Page config ──
st.set_page_config(page_title="Onagawa Source Chat", layout="wide", page_icon="🌊")

# ── Session state ──
for k, v in [("messages", []), ("pending_prompt", None), ("retriever", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════
def _s(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    return str(x)

def _trunc(s: str, n: int = 180) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

def _fmt(v, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "–"
    return f"{v:.{d}f}"


# ═══════════════════════════════════════════
# Data loading (cached)
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading documents…")
def load_documents() -> List[dict]:
    p = config.SERVING_DIR / "retrieval_documents.jsonl"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

@st.cache_data(show_spinner="Loading sample registry…")
def load_sample_registry() -> pd.DataFrame:
    p = config.SERVING_DIR / "sample_registry.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner="Loading CTD summary…")
def load_ctd_summary() -> pd.DataFrame:
    p = config.NORMALIZED_DIR / "ctd_summary.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner="Loading CTD profiles…")
def load_ctd_profiles() -> pd.DataFrame:
    p = config.NORMALIZED_DIR / "ctd_profile_standardized.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner="Loading SST data…")
def load_sst_timeseries() -> pd.DataFrame:
    p = config.NORMALIZED_DIR / "sst_point_timeseries.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner="Loading SST daily…")
def load_sst_daily() -> pd.DataFrame:
    p = config.NORMALIZED_DIR / "sst_daily_summary.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner="Loading sample context…")
def load_sample_context() -> pd.DataFrame:
    p = config.SERVING_DIR / "sample_multisource_context.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def get_retriever():
    """Get or create the local retriever singleton."""
    if st.session_state.retriever is None:
        from retrieval.local_retriever import LocalRetriever
        r = LocalRetriever()
        r.load()
        r.ensure_embeddings()
        st.session_state.retriever = r
    return st.session_state.retriever


# ═══════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════
st.sidebar.header("⚙️ Settings")

ollama_url = st.sidebar.text_input(
    "Ollama URL", value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
)
config.OLLAMA_BASE_URL = ollama_url

# Try to list models
try:
    import requests
    resp = requests.get(f"{ollama_url}/api/tags", timeout=3)
    model_names = [m["name"] for m in resp.json().get("models", [])]
except Exception:
    model_names = []

with st.sidebar.expander("🤖 Model", expanded=True):
    model = st.selectbox(
        "Chat model",
        options=model_names or ["qwen2.5:14b-instruct"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.05)
    top_k_sources = st.slider("Top-K sources", 1, 15, 6, 1)

with st.sidebar.expander("🔍 Filters", expanded=False):
    filter_source = st.selectbox("Source type", ["All", "ctd", "metagenome", "remote_sensing"])
    filter_bay = st.selectbox("Bay", ["All", "O – Onagawa", "I – Ishinomaki", "M – Matsushima"])

if st.sidebar.button("🗑️ Reset chat"):
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.rerun()


# ═══════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════
docs = load_documents()
registry = load_sample_registry()
ctd_summary = load_ctd_summary()
ctd_profiles = load_ctd_profiles()
sst_ts = load_sst_timeseries()
sst_daily = load_sst_daily()
sample_ctx = load_sample_context()
retriever = get_retriever()


# ═══════════════════════════════════════════
# Main UI
# ═══════════════════════════════════════════
st.title("🌊 Onagawa Source Chat")
st.caption("Provenance-aware marine RAG — CTD · Metagenome · Satellite SST")

tab_chat, tab_explore, tab_ctd, tab_taxa, tab_sst, tab_stats = st.tabs(
    ["💬 Chat", "📋 Evidence Explorer", "🌡️ CTD Profiles", "🧬 Taxa", "🛰️ SST", "📊 Stats"]
)


# ═══════════════════════════════════════════
# TAB: Chat
# ═══════════════════════════════════════════
with tab_chat:
    col_chat, col_sources = st.columns([2.5, 1.0], gap="large")

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

                # Retrieve
                src_filter = None if filter_source == "All" else filter_source
                bay_filter = None if filter_bay == "All" else filter_bay[0]
                retrieved = retriever.search(user_text, k=top_k_sources,
                                             source_type=src_filter, bay=bay_filter)

                with st.expander(f"📎 Retrieved {len(retrieved)} sources", expanded=False):
                    for r in retrieved:
                        src_icon = {"ctd": "🌡️", "metagenome": "🧬", "remote_sensing": "🛰️"}.get(
                            r.get("source_type", ""), "📄")
                        st.markdown(
                            f"**{src_icon} [{r.get('id', r.get('doc_id', ''))}]** "
                            f"{r.get('title', '')}  \n"
                            f"Score: `{r.get('score', 0):.4f}` | {r.get('time', r.get('date', ''))}"
                        )
                        st.caption(_trunc(r.get("text", ""), 300))

                # Build prompt
                from orchestration.unified import build_prompt
                prompt_text = build_prompt(user_text, retrieved)

                # Call Ollama
                full = ""
                try:
                    import requests
                    resp = requests.post(
                        f"{ollama_url}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt_text}],
                            "stream": True,
                            "options": {"temperature": temperature},
                        },
                        stream=True, timeout=120,
                    )
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            full += token
                            placeholder.markdown(full)
                except Exception as e:
                    if not full:
                        full = f"⚠️ LLM error: {e}"
                    placeholder.markdown(full)

                st.session_state.messages.append({"role": "assistant", "content": full})

    with col_sources:
        st.subheader("📚 Corpus")
        st.metric("Documents", len(docs))
        src_counts = Counter(d.get("source_type", "") for d in docs)
        for src, cnt in sorted(src_counts.items()):
            icon = {"ctd": "🌡️", "metagenome": "🧬", "remote_sensing": "🛰️"}.get(src, "📄")
            st.caption(f"{icon} {src}: {cnt}")


# ═══════════════════════════════════════════
# TAB: Evidence Explorer
# ═══════════════════════════════════════════
with tab_explore:
    st.subheader("Evidence Explorer")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        eq = st.text_input("🔍 Search documents", placeholder="e.g. dinoflagellate, temperature June")
    with c2:
        esrc = st.selectbox("Source type ", ["All", "ctd", "metagenome", "remote_sensing"], key="esrc")
    with c3:
        ebay = st.selectbox("Bay ", ["All", "O", "I", "M"], key="ebay")

    if eq:
        sf = None if esrc == "All" else esrc
        bf = None if ebay == "All" else ebay
        results = retriever.search(eq, k=20, source_type=sf, bay=bf)
        st.caption(f"Found {len(results)} results")
        for r in results:
            icon = {"ctd": "🌡️", "metagenome": "🧬", "remote_sensing": "🛰️"}.get(r.get("source_type"), "📄")
            with st.expander(f"{icon} {r.get('title', r.get('id', ''))} — score {r.get('score',0):.4f}"):
                st.markdown(r.get("text", ""))
                st.caption(f"doc_id: {r.get('id', r.get('doc_id', ''))} | "
                           f"sample: {r.get('sample_id', '–')} | "
                           f"event: {r.get('event_id', '–')}")
    else:
        st.info("Enter a search query above to explore the evidence base.")


# ═══════════════════════════════════════════
# TAB: CTD Profiles
# ═══════════════════════════════════════════
with tab_ctd:
    st.subheader("🌡️ CTD Depth Profiles")
    if ctd_profiles.empty:
        st.warning("No CTD profile data available.")
    else:
        samples = sorted(ctd_profiles["sample_id"].dropna().unique())
        selected = st.selectbox("Select sample", samples, index=0)
        prof = ctd_profiles[ctd_profiles["sample_id"] == selected].sort_values("depth_m")

        if not prof.empty:
            vars_available = [c for c in ["temperature", "salinity", "do_percent", "chl_a",
                                           "turbidity", "sigma_t"] if c in prof.columns]
            selected_vars = st.multiselect("Variables", vars_available,
                                            default=vars_available[:3])
            if selected_vars:
                fig, axes = plt.subplots(1, len(selected_vars),
                                          figsize=(4 * len(selected_vars), 6), sharey=True)
                if len(selected_vars) == 1:
                    axes = [axes]
                for ax, var in zip(axes, selected_vars):
                    vals = pd.to_numeric(prof[var], errors="coerce")
                    ax.plot(vals, prof["depth_m"], "o-", markersize=3)
                    ax.set_xlabel(var)
                    ax.invert_yaxis()
                    if ax == axes[0]:
                        ax.set_ylabel("Depth (m)")
                    ax.grid(True, alpha=0.3)
                fig.suptitle(f"CTD Profile: {selected}", fontsize=14)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

            # Summary stats
            if not ctd_summary.empty:
                row = ctd_summary[ctd_summary["sample_id"] == selected]
                if not row.empty:
                    r = row.iloc[0]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Depth points", int(r.get("n_depth_points", 0)))
                    c2.metric("Surface T", f"{_fmt(r.get('surface_temperature'))}°C")
                    c3.metric("Bottom T", f"{_fmt(r.get('bottom_temperature'))}°C")
                    c4.metric("Mean Sal", f"{_fmt(r.get('mean_salinity'))} PSU")


# ═══════════════════════════════════════════
# TAB: Taxa
# ═══════════════════════════════════════════
with tab_taxa:
    st.subheader("🧬 Taxonomic Composition")
    if sample_ctx.empty:
        st.warning("No metagenome data available.")
    else:
        meta_samples = sample_ctx[sample_ctx["has_kraken"] == True]["sample_id"].dropna().unique()
        meta_samples = sorted(meta_samples)
        if meta_samples:
            sel_sample = st.selectbox("Select metagenome sample", meta_samples, key="taxa_sample")
            row = sample_ctx[sample_ctx["sample_id"] == sel_sample].iloc[0]

            col_kr, col_me = st.columns(2)

            # Kraken top genera
            with col_kr:
                st.markdown("**Kraken top genera**")
                kr_json = row.get("top_genus_10_json_x")
                if pd.notna(kr_json) and isinstance(kr_json, str):
                    try:
                        taxa = json.loads(kr_json)
                        if taxa:
                            names = [t["genus"] for t in taxa]
                            vals = [t["abundance_value"] for t in taxa]
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.barh(names[::-1], vals[::-1], color="#2196F3")
                            ax.set_xlabel("Abundance (%)")
                            ax.set_title(f"Kraken – {sel_sample}")
                            fig.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                    except Exception:
                        st.caption("Could not parse Kraken data")

            # MetaEuk top genera
            with col_me:
                st.markdown("**MetaEuk top genera**")
                me_json = row.get("top_genus_10_json_y")
                if pd.notna(me_json) and isinstance(me_json, str):
                    try:
                        taxa = json.loads(me_json)
                        if taxa:
                            names = [t["genus"] for t in taxa]
                            vals = [t["abundance_value"] for t in taxa]
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.barh(names[::-1], vals[::-1], color="#4CAF50")
                            ax.set_xlabel("Abundance (%)")
                            ax.set_title(f"MetaEuk – {sel_sample}")
                            fig.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                    except Exception:
                        st.caption("Could not parse MetaEuk data")

            # Upper groups
            ug_json = row.get("top_upper_group_10_json")
            if pd.notna(ug_json) and isinstance(ug_json, str):
                try:
                    groups = json.loads(ug_json)
                    if groups:
                        st.markdown("**Dominant taxonomic groups**")
                        names = [g["upper_group"] for g in groups]
                        vals = [g["abundance_value"] for g in groups]
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.barh(names[::-1], vals[::-1], color="#FF9800")
                        ax.set_xlabel("Abundance (%)")
                        ax.set_title(f"Upper groups – {sel_sample}")
                        fig.tight_layout()
                        st.pyplot(fig, clear_figure=True)
                except Exception:
                    pass
        else:
            st.info("No metagenome samples found.")


# ═══════════════════════════════════════════
# TAB: SST
# ═══════════════════════════════════════════
with tab_sst:
    st.subheader("🛰️ Satellite SST")
    if sst_ts.empty:
        st.warning("No SST data. Run `python scripts/ingest.py` to process SST files.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", len(sst_ts))
        c2.metric("Min SST", f"{sst_ts['sst'].min():.1f}°C")
        c3.metric("Max SST", f"{sst_ts['sst'].max():.1f}°C")
        c4.metric("Mean SST", f"{sst_ts['sst'].mean():.1f}°C")

        st.markdown("### Point SST time series")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(sst_ts["time_jst"], sst_ts["sst"], linewidth=0.8, color="#1976D2")
        ax.set_xlabel("Time (JST)")
        ax.set_ylabel("SST (°C)")
        ax.set_title(f"Onagawa monitoring point ({config.ONAGAWA_LAT:.4f}°N, {config.ONAGAWA_LON:.4f}°E)")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        if not sst_daily.empty:
            st.markdown("### Daily regional summary")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.fill_between(pd.to_datetime(sst_daily["date_jst"]),
                             sst_daily["min_sst"], sst_daily["max_sst"],
                             alpha=0.2, color="#1976D2", label="min–max range")
            ax2.plot(pd.to_datetime(sst_daily["date_jst"]),
                     sst_daily["mean_sst"], color="#1976D2", linewidth=1.5, label="mean")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("SST (°C)")
            ax2.set_title("Regional SST daily summary")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.autofmt_xdate()
            fig2.tight_layout()
            st.pyplot(fig2, clear_figure=True)


# ═══════════════════════════════════════════
# TAB: Stats
# ═══════════════════════════════════════════
with tab_stats:
    st.subheader("📊 Corpus Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total documents", len(docs))
    src_counts = Counter(d.get("source_type", "") for d in docs)
    c2.metric("CTD docs", src_counts.get("ctd", 0))
    c3.metric("Metagenome docs", src_counts.get("metagenome", 0))
    c4.metric("SST docs", src_counts.get("remote_sensing", 0))

    if not registry.empty:
        st.markdown("### Sample coverage")
        cov = registry[["sample_id", "bay", "has_run_qc", "has_kraken", "has_metaeuk", "has_ctd"]].copy()
        cov = cov.rename(columns={"has_run_qc": "QC", "has_kraken": "Kraken",
                                   "has_metaeuk": "MetaEuk", "has_ctd": "CTD"})
        st.dataframe(cov, use_container_width=True, height=400)

        st.markdown("### Samples by bay")
        bay_counts = registry["bay"].value_counts()
        bay_names = {"O": "Onagawa", "I": "Ishinomaki", "M": "Matsushima"}
        for bay, cnt in bay_counts.items():
            st.markdown(f"- **{bay_names.get(bay, bay)}** ({bay}): {cnt} samples")

    st.markdown("### Provenance")
    prov_path = config.PROVENANCE_DIR / "provenance.jsonl"
    if prov_path.exists():
        with open(prov_path) as f:
            prov_count = sum(1 for _ in f)
        st.metric("Registered files", prov_count)
    else:
        st.caption("No provenance registry found.")


# ═══════════════════════════════════════════
# Chat input (pinned at bottom)
# ═══════════════════════════════════════════
prompt = st.chat_input('Ask: "What was the temperature in Onagawa Bay in June 2024?"')
if prompt:
    st.session_state.pending_prompt = prompt
    st.rerun()
