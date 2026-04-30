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

# Try to list models (filter out embedding-only models)
_EMBED_ONLY = {"nomic-embed-text", "mxbai-embed-large", "all-minilm", "snowflake-arctic-embed"}
try:
    import requests
    resp = requests.get(f"{ollama_url}/api/tags", timeout=3)
    model_names = [m["name"] for m in resp.json().get("models", [])
                   if not any(e in m["name"] for e in _EMBED_ONLY)]
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

# Backend status
@st.cache_data(ttl=30, show_spinner=False)
def _check_pg():
    try:
        from sqlalchemy import create_engine, text
        e = create_engine(config.DATABASE_URL, pool_pre_ping=True)
        with e.connect() as c:
            n = c.execute(text(
                "SELECT count(*) FROM retrieval_document WHERE embedding IS NOT NULL"
            )).scalar()
        return n
    except Exception:
        return 0

pg_embed_count = _check_pg()
if pg_embed_count > 0:
    st.sidebar.success(f"🔗 pgvector active ({pg_embed_count} embedded docs)")
    USE_PG = True
else:
    st.sidebar.info("📂 Local BM25 search")
    USE_PG = False


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

tab_chat, tab_explore, tab_ctd, tab_taxa, tab_sst, tab_db, tab_stats = st.tabs(
    ["💬 Chat", "📋 Evidence Explorer", "🌡️ CTD Profiles", "🧬 Taxa", "🛰️ SST", "🗄️ Database", "📊 Stats"]
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
                if USE_PG:
                    from orchestration.unified import retrieve
                    retrieved = retrieve(user_text, k=top_k_sources,
                                         source_type=src_filter, bay=bay_filter)
                else:
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
        if USE_PG:
            from orchestration.unified import retrieve
            results = retrieve(eq, k=20, source_type=sf, bay=bf)
        else:
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
        st.dataframe(cov, width="stretch", height=400)

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
# TAB: Database Explorer
# ═══════════════════════════════════════════
with tab_db:
    st.subheader("🗄️ Database Explorer")

    if not USE_PG:
        st.warning("PostgreSQL is not connected. Start the database with `podman compose up -d`.")
    else:
        from sqlalchemy import create_engine, text, inspect
        _db_engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)

        # ── Connection & Schema Overview ──
        with _db_engine.connect() as _conn:
            pg_ver = _conn.execute(text("SELECT version()")).scalar()
        inspector = inspect(_db_engine)
        table_names = inspector.get_table_names()

        st.caption(f"Connected: `{pg_ver[:60]}…` • Port **{config.DATABASE_URL.split(':')[-1].split('/')[0]}** • Tables: **{len(table_names)}**")

        # ── Sub-tabs inside Database ──
        db_sub1, db_sub2, db_sub3, db_sub4 = st.tabs(
            ["📋 Table Browser", "🔎 SQL Console", "📐 Schema", "📊 Embeddings"]
        )

        # ──────────────────────────────────
        # Sub-tab 1: Table Browser
        # ──────────────────────────────────
        with db_sub1:
            col_tbl, col_opts = st.columns([1, 3])

            with col_tbl:
                selected_table = st.selectbox("Table", table_names, index=0, key="db_table")

                # Row count
                with _db_engine.connect() as _conn:
                    row_count = _conn.execute(text(f"SELECT count(*) FROM {selected_table}")).scalar()
                st.metric("Rows", f"{row_count:,}")

                # Column info
                columns_info = inspector.get_columns(selected_table)
                col_names = [c["name"] for c in columns_info]
                st.caption(f"{len(col_names)} columns")

            with col_opts:
                # Filters
                fc1, fc2, fc3 = st.columns([2, 1, 1])
                with fc1:
                    where_clause = st.text_input(
                        "WHERE clause (optional)",
                        placeholder="e.g. bay = 'O' AND source_type = 'ctd'",
                        key="db_where",
                    )
                with fc2:
                    order_col = st.selectbox("ORDER BY", [""] + col_names, index=0, key="db_order")
                with fc3:
                    limit = st.number_input("LIMIT", min_value=1, max_value=5000, value=100, key="db_limit")

                # Build and execute query
                query_str = f"SELECT * FROM {selected_table}"
                if where_clause:
                    query_str += f" WHERE {where_clause}"
                if order_col:
                    query_str += f" ORDER BY {order_col}"
                query_str += f" LIMIT {limit}"

                try:
                    df_result = pd.read_sql(text(query_str), _db_engine)

                    # Hide embedding column (too large to display)
                    display_cols = [c for c in df_result.columns if c not in ("embedding", "text_tsv")]
                    st.dataframe(df_result[display_cols], width="stretch", height=420)

                    # Download
                    csv_data = df_result[display_cols].to_csv(index=False)
                    st.download_button(
                        f"⬇️ Download {selected_table} ({len(df_result)} rows)",
                        data=csv_data,
                        file_name=f"{selected_table}_export.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Query error: {e}")

                # Row detail inspector
                if "df_result" in dir() and not df_result.empty:
                    with st.expander("🔍 Row Inspector", expanded=False):
                        row_idx = st.number_input("Row index", 0, len(df_result) - 1, 0, key="db_row_idx")
                        row_data = df_result.iloc[row_idx]
                        for col_name, val in row_data.items():
                            if col_name in ("embedding", "text_tsv"):
                                st.text(f"{col_name}: [hidden – {type(val).__name__}]")
                            elif col_name == "text" and isinstance(val, str) and len(val) > 200:
                                st.markdown(f"**{col_name}:**")
                                st.text_area("Text content", val, height=150, key=f"db_detail_{col_name}")
                            else:
                                st.markdown(f"**{col_name}:** {val}")

        # ──────────────────────────────────
        # Sub-tab 2: SQL Console
        # ──────────────────────────────────
        with db_sub2:
            st.markdown("Run **read-only** SQL queries against the database.")

            default_sql = """-- Example queries:
-- SELECT source_type, count(*) FROM retrieval_document GROUP BY source_type;
-- SELECT sample_id, surface_temperature, mean_salinity FROM ctd_summary ORDER BY ctd_date LIMIT 20;
-- SELECT * FROM cross_source_link WHERE link_type = 'time_match' LIMIT 10;

SELECT source_type, count(*) AS n_docs,
       round(avg(length(text))) AS avg_text_len
FROM retrieval_document
GROUP BY source_type
ORDER BY n_docs DESC;"""

            sql_input = st.text_area("SQL Query", value=default_sql, height=180, key="db_sql")

            col_run, col_info = st.columns([1, 3])
            with col_run:
                run_sql = st.button("▶️ Run Query", key="db_run_sql")

            if run_sql and sql_input.strip():
                # Safety: block destructive statements
                sql_upper = sql_input.strip().upper()
                blocked = ["DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE", "CREATE"]
                if any(sql_upper.startswith(kw) for kw in blocked):
                    st.error("❌ Only SELECT queries are allowed.")
                else:
                    try:
                        import time as _time
                        t0 = _time.perf_counter()
                        df_sql = pd.read_sql(text(sql_input), _db_engine)
                        elapsed = _time.perf_counter() - t0

                        with col_info:
                            st.caption(f"✅ {len(df_sql)} rows in {elapsed:.3f}s")

                        # Hide embedding columns
                        display = [c for c in df_sql.columns if c not in ("embedding", "text_tsv")]
                        st.dataframe(df_sql[display], width="stretch", height=400)

                        st.download_button(
                            f"⬇️ Download result ({len(df_sql)} rows)",
                            data=df_sql[display].to_csv(index=False),
                            file_name="query_result.csv",
                            mime="text/csv",
                            key="db_sql_download",
                        )
                    except Exception as e:
                        st.error(f"SQL error: {e}")

        # ──────────────────────────────────
        # Sub-tab 3: Schema
        # ──────────────────────────────────
        with db_sub3:
            for tname in table_names:
                cols = inspector.get_columns(tname)
                pk = inspector.get_pk_constraint(tname)
                indexes = inspector.get_indexes(tname)

                with st.expander(f"📋 **{tname}** ({len(cols)} columns)", expanded=False):
                    # Column table
                    schema_rows = []
                    pk_cols = set(pk.get("constrained_columns", []))
                    for c in cols:
                        schema_rows.append({
                            "Column": c["name"],
                            "Type": str(c["type"]),
                            "Nullable": "✓" if c.get("nullable", True) else "✗",
                            "PK": "🔑" if c["name"] in pk_cols else "",
                            "Default": str(c.get("default", "") or ""),
                        })
                    st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

                    if indexes:
                        st.markdown("**Indexes:**")
                        for idx in indexes:
                            unique = "UNIQUE " if idx.get("unique") else ""
                            st.caption(f"  `{idx['name']}` — {unique}{', '.join(idx['column_names'])}")

        # ──────────────────────────────────
        # Sub-tab 4: Embedding Statistics
        # ──────────────────────────────────
        with db_sub4:
            with _db_engine.connect() as _conn:
                total_docs = _conn.execute(text("SELECT count(*) FROM retrieval_document")).scalar()
                embedded = _conn.execute(text(
                    "SELECT count(*) FROM retrieval_document WHERE embedding IS NOT NULL"
                )).scalar()
                fts_count = _conn.execute(text(
                    "SELECT count(*) FROM retrieval_document WHERE text_tsv IS NOT NULL"
                )).scalar()

                # Per source type
                src_stats = _conn.execute(text("""
                    SELECT source_type,
                           count(*) AS total,
                           count(embedding) AS with_embedding,
                           round(avg(length(text))) AS avg_text_len
                    FROM retrieval_document
                    GROUP BY source_type
                    ORDER BY total DESC
                """)).fetchall()

                # Cross-source links
                link_count = _conn.execute(text("SELECT count(*) FROM cross_source_link")).scalar()
                link_types = _conn.execute(text(
                    "SELECT link_type, count(*) FROM cross_source_link GROUP BY link_type"
                )).fetchall()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total docs", total_docs)
            c2.metric("With embeddings", f"{embedded}/{total_docs}")
            c3.metric("With FTS", f"{fts_count}/{total_docs}")
            c4.metric("Cross-source links", link_count)

            coverage_pct = (embedded / total_docs * 100) if total_docs > 0 else 0
            if coverage_pct >= 100:
                st.success(f"✅ 100% embedding coverage ({config.EMBEDDING_MODEL}, {config.EMBEDDING_DIM}-dim)")
            elif coverage_pct > 0:
                st.warning(f"⚠️ {coverage_pct:.0f}% embedding coverage — run `scripts/load_db.py --embed`")
            else:
                st.error("❌ No embeddings — run `scripts/load_db.py --embed`")

            st.markdown("### Documents by source type")
            emb_df = pd.DataFrame([
                {
                    "Source type": r[0],
                    "Total": r[1],
                    "Embedded": r[2],
                    "Avg text (chars)": int(r[3]) if r[3] else 0,
                    "Coverage": f"{r[2]/r[1]*100:.0f}%" if r[1] > 0 else "–",
                }
                for r in src_stats
            ])
            st.dataframe(emb_df, width="stretch", hide_index=True)

            st.markdown("### Cross-source links")
            for lt, cnt in link_types:
                st.markdown(f"- **{lt}**: {cnt:,} links")

            # Similarity test tool
            st.markdown("### 🧪 Similarity Probe")
            st.caption("Test what a query retrieves via vector vs FTS.")
            probe_q = st.text_input("Probe query", placeholder="e.g. chlorophyll bloom summer", key="db_probe")
            if probe_q:
                from retrieval.hybrid_retriever import hybrid_search
                probe_results = hybrid_search(probe_q, k=10)
                st.caption(f"Top 10 hybrid results (RRF fusion):")
                probe_data = []
                for r in probe_results:
                    probe_data.append({
                        "doc_id": r.doc_id,
                        "source": r.source_type,
                        "score": f"{r.score:.4f}",
                        "vec_rank": r.rank_sources.get("vector", "–"),
                        "fts_rank": r.rank_sources.get("fts", "–"),
                        "title": r.title[:80],
                    })
                st.dataframe(pd.DataFrame(probe_data), width="stretch", hide_index=True)


# ═══════════════════════════════════════════
# Chat input (pinned at bottom)
# ═══════════════════════════════════════════
prompt = st.chat_input('Ask: "What was the temperature in Onagawa Bay in June 2024?"')
if prompt:
    st.session_state.pending_prompt = prompt
    st.rerun()
