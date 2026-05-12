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
st.set_page_config(page_title="Onagawa Source Chat", layout="wide", page_icon="")

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
st.sidebar.header("Settings")

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

# ── Model Settings ──
with st.sidebar.expander("Model", expanded=True):
    model = st.selectbox(
        "Chat model",
        options=model_names or ["qwen2.5:14b-instruct"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.05,
                            help="Controls randomness. 0 = deterministic, higher = more creative.")
    top_p = st.slider("Top-P (nucleus sampling)", 0.0, 1.0, 0.9, 0.05,
                      help="Cumulative probability cutoff for token sampling. Lower = more focused.")
    repeat_penalty = st.slider("Repeat penalty", 0.5, 2.0, 1.1, 0.05,
                               help="Penalizes token repetition. Higher = less repetitive output.")
    num_ctx = st.select_slider("Context window", options=[2048, 4096, 8192, 16384, 32768],
                               value=8192,
                               help="Max tokens the model can process. Larger = more context but slower.")

# ── Retrieval Settings ──
with st.sidebar.expander("Retrieval", expanded=False):
    top_k_sources = st.slider("Top-K sources", 1, 20, 6, 1,
                              help="Number of documents retrieved per query.")

    st.caption("**Hybrid search weights**")
    vector_weight = st.slider("Vector weight", 0.0, 1.0, 0.6, 0.05,
                              help="Weight for semantic (embedding) similarity in RRF fusion.")
    fts_weight = st.slider("FTS weight", 0.0, 1.0, 0.4, 0.05,
                           help="Weight for keyword (full-text) search in RRF fusion.")

    rrf_k = st.slider("RRF-k constant", 1, 200, 60, 1,
                       help="Smoothing constant for Reciprocal Rank Fusion. "
                            "Higher = more uniform blending; lower = sharper rank differences.")

    inject_analysis = st.checkbox("Inject pre-analysis context", value=True,
                                  help="When enabled, precomputed ecological analyses are "
                                       "automatically added to prompts for complex queries.")
    inject_reliability = st.checkbox("Inject reliability context", value=True,
                                     help="When enabled, cross-source validation and "
                                          "corroboration results are added to prompts.")

# ── Filters ──
with st.sidebar.expander("Filters", expanded=False):
    filter_source = st.selectbox("Source type", ["All", "ctd", "metagenome", "remote_sensing"])
    filter_bay = st.selectbox("Bay", ["All", "O -- Onagawa", "I -- Ishinomaki", "M -- Matsushima"])

    st.caption("**Time range**")
    filter_time_from = st.text_input("From (YYYY-MM-DD)", value="",
                                     help="Filter documents from this date onward.")
    filter_time_to = st.text_input("To (YYYY-MM-DD)", value="",
                                   help="Filter documents up to this date.")

# ── Actions ──
st.sidebar.markdown("---")
if st.sidebar.button("Reset chat"):
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
    st.sidebar.success(f"pgvector active ({pg_embed_count} embedded docs)")
    USE_PG = True
else:
    st.sidebar.info("Local BM25 search")
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
st.title("Onagawa Source Chat")
st.caption("Provenance-aware marine RAG — CTD · Metagenome · Satellite SST")

tab_overview, tab_chat, tab_explore, tab_data, tab_analysis, tab_db, tab_stats = st.tabs(
    ["Overview", "Chat", "Evidence Explorer", "Data",
     "Pre-Analysis", "Database", "Stats"]
)


# ═══════════════════════════════════════════
# TAB: Overview
# ═══════════════════════════════════════════
with tab_overview:
    st.subheader("System Overview")
    st.caption("End-to-end architecture of the Onagawa Source Chat RAG pipeline.")

    # ─── Live pipeline statistics ───
    # Gather stats from loaded data and filesystem
    _prov_path = config.PROVENANCE_DIR / "provenance.jsonl"
    _prov_count = 0
    if _prov_path.exists():
        with open(_prov_path, "r") as _f:
            _prov_count = sum(1 for _ in _f)

    _norm_files = list(config.NORMALIZED_DIR.glob("*.parquet")) if config.NORMALIZED_DIR.exists() else []
    _analysis_files = list(config.ANALYSIS_DIR.glob("*.parquet")) if config.ANALYSIS_DIR.exists() else []
    _analysis_docs_path = config.ANALYSIS_DIR / "analysis_documents.jsonl"
    _n_analysis_docs = 0
    if _analysis_docs_path.exists():
        with open(_analysis_docs_path, "r") as _f:
            _n_analysis_docs = sum(1 for line in _f if line.strip())

    _anchor_path = config.CANONICAL_DIR / "anchor_events.parquet"
    _n_anchors = 0
    if _anchor_path.exists():
        _n_anchors = len(pd.read_parquet(_anchor_path))

    _links_path = config.CANONICAL_DIR / "cross_source_links.parquet"
    _n_links = 0
    if _links_path.exists():
        _n_links = len(pd.read_parquet(_links_path))

    _reliability_files = list(config.RELIABILITY_DIR.glob("*.parquet")) if config.RELIABILITY_DIR.exists() else []
    _rel_docs_path = config.RELIABILITY_DIR / "reliability_documents.jsonl"
    _n_rel_docs = 0
    if _rel_docs_path.exists():
        with open(_rel_docs_path, "r") as _f:
            _n_rel_docs = sum(1 for line in _f if line.strip())
    _corrob_path = config.RELIABILITY_DIR / "corroboration.parquet"
    _n_verified = 0
    _n_corrob_total = 0
    if _corrob_path.exists():
        _corrob_df = pd.read_parquet(_corrob_path)
        _n_corrob_total = len(_corrob_df)
        _n_verified = int((_corrob_df["reliability_tier"] == "verified").sum()) if "reliability_tier" in _corrob_df.columns else 0

    # ─── Pipeline stages as interactive expanders ───
    st.markdown("---")
    st.markdown("### Pipeline Architecture")
    st.caption("Click each stage to explore what happens at that layer.")

    # Stage 1: Data Sources
    with st.expander("Stage 1 -- Data Sources", expanded=True):
        st.markdown("""
This system integrates three marine monitoring data sources from Miyagi Prefecture, Japan:

| Source | Instrument / Platform | Format | Description |
|---|---|---|---|
| **CTD** | PlanDyo multi-sensor | TSV | Depth-resolved water column profiles (temperature, salinity, dissolved oxygen, chlorophyll-a, turbidity) |
| **Metagenome** | PlanDyo sequencing | TSV | Community composition via Kraken2 (prokaryotes) and MetaEuk (eukaryotes) at genus level |
| **SST** | Himawari-9 satellite | NetCDF | Hourly sea surface temperature at 0.02-degree resolution |
""")

        st.markdown("**Study sites:**")
        site_df = pd.DataFrame({
            "Bay": ["Onagawa (O)", "Ishinomaki (I)", "Matsushima (M)"],
            "Latitude": ["~38.44 N", "~38.41 N", "~38.35 N"],
            "Longitude": ["~141.45 E", "~141.30 E", "~141.06 E"],
            "Data": ["CTD + Metagenome + SST", "CTD + Metagenome", "CTD + Metagenome"],
        })
        st.dataframe(site_df, hide_index=True, width="stretch")

        # Live stats
        c1, c2, c3 = st.columns(3)
        _raw_ctd_count = len(list(config.RAW_CTD_DIR.glob("*"))) if config.RAW_CTD_DIR.exists() else 0
        _raw_meta_count = len(list(config.RAW_META_DIR.glob("*"))) if config.RAW_META_DIR.exists() else 0
        c1.metric("CTD raw files", _raw_ctd_count)
        c2.metric("Metagenome raw files", _raw_meta_count)
        c3.metric("Total registered (provenance)", f"{_prov_count:,}")

    # Stage 2: Ingestion & Provenance
    with st.expander("Stage 2 -- Ingestion and Provenance Tracking"):
        st.markdown("""
Every raw file is registered with a **SHA-256 hash** before any processing begins.
This creates an immutable provenance chain from raw bytes to final LLM answers.

**What happens:**
1. Each file is scanned and its SHA-256 checksum computed
2. Metadata recorded: filename, size, modification time, hash
3. Written to `provenance.jsonl` as an append-only log

**Why it matters:**  
If a source file changes upstream, the hash mismatch is immediately detectable,
ensuring reproducibility and audit trail for scientific claims.
""")

        c1, c2 = st.columns(2)
        c1.metric("Files registered", f"{_prov_count:,}")
        c2.metric("Registry location", "data/provenance/provenance.jsonl")

        if _prov_path.exists():
            with st.expander("Preview provenance records"):
                import json as _json
                _prov_sample = []
                with open(_prov_path, "r") as _f:
                    for i, line in enumerate(_f):
                        if i >= 5:
                            break
                        _prov_sample.append(_json.loads(line))
                st.json(_prov_sample)

    # Stage 3: Preprocessing
    with st.expander("Stage 3 -- Preprocessing"):
        st.markdown("""
Raw data is standardized into analysis-ready Parquet files:

**CTD Pipeline** (`preprocessing/ctd.py`):
- Parse PlanDyo TSV with Japanese column headers
- Standardize column names, units, and depth bins
- Compute per-cast summaries (surface/bottom T, mean salinity, depth range)

**Metagenome Pipeline** (`preprocessing/metagenome.py`):
- Parse Kraken2 genus-level abundance tables
- Parse MetaEuk genus-level abundance tables
- Compute upper taxonomic group summaries (Dinoflagellata, Bacillariophyta, etc.)
- Build per-sample QC context (total reads, classified fraction)

**SST Pipeline** (`preprocessing/remote_sensing.py`):
- Extract SST at monitoring coordinates from Himawari netCDF grids
- Compute daily regional summaries (mean, min, max over bounding box)
""")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Normalized files", len(_norm_files))
        c2.metric("CTD profiles", f"{len(ctd_profiles):,}" if not ctd_profiles.empty else "0")
        c3.metric("CTD casts", f"{len(ctd_summary):,}" if not ctd_summary.empty else "0")
        c4.metric("SST observations", f"{len(sst_ts):,}" if not sst_ts.empty else "0")

        if not ctd_profiles.empty:
            st.markdown("**Available CTD variables:**")
            ctd_vars = [c for c in ctd_profiles.columns
                        if c not in ["sample_id", "depth_m", "bay", "station", "date"]]
            st.code(", ".join(ctd_vars))

    # Stage 4: Canonical Schema & Linking
    with st.expander("Stage 4 -- Spatiotemporal Linking (Anchor Events)"):
        st.markdown("""
An **Anchor Event** is a canonical spatiotemporal record that connects observations
from different instruments taken at the same place and time.

**How it works:**
1. Each CTD cast and metagenome sample generates an anchor at (lat, lon, datetime)
2. Each SST observation generates an anchor at (monitoring point, datetime)
3. Cross-source links are created when anchors overlap temporally (same day/month)

This enables the system to answer questions like  
*"What was the microbial community like when SST was highest?"*  
by following links from SST anchors to matching metagenome anchors.
""")

        c1, c2, c3 = st.columns(3)
        c1.metric("Anchor events", f"{_n_anchors:,}")
        c2.metric("Cross-source links", f"{_n_links:,}")
        c3.metric("Link types", "same_sample, time_match")

    # Stage 5: Pre-Analysis
    with st.expander("Stage 5 -- Pre-Analysis (Ecological Relationships)"):
        st.markdown("""
Before retrieval, the system precomputes ecological relationships that span
multiple data sources. These analyses would be impossible to derive from
individual retrieved documents alone.

| Analysis | Method | Output |
|---|---|---|
| **CTD Trends** | Monthly mean/std aggregation per bay | Seasonal temperature, salinity, DO patterns |
| **Taxa-Env Correlations** | Spearman rank correlation | Which genera increase/decrease with environmental variables |
| **Diversity Indices** | Shannon H', Simpson 1-D, Richness, Evenness | Community diversity per sample |
| **Bay Comparison** | Cross-bay CTD aggregation | Environmental differences between study sites |
| **Co-occurrence** | Jaccard similarity (10-90% prevalence genera) | Which genera tend to appear together |

These results are stored as Parquet files and also serialized into
**analysis documents** (JSONL) that get injected into the LLM prompt
when the user asks complex ecosystem questions.
""")

        c1, c2, c3 = st.columns(3)
        c1.metric("Analysis outputs", len(_analysis_files))
        c2.metric("Analysis docs (for RAG)", _n_analysis_docs)
        c3.metric("Trigger keywords", "correlation, diversity, trend, seasonal, ...")

    # Stage 5b: Reliability Ensurance
    with st.expander("Stage 5b -- Reliability Ensurance (Cross-Source Validation)"):
        st.markdown("""
Alongside pre-analysis, the system runs **cross-source validation** to reinforce
data confidence. When one source has information that another lacks, the system
uses available data to predict, interpolate, or corroborate observations.

| Check | Method | Purpose |
|---|---|---|
| **SST ↔ CTD Validation** | Compare satellite SST with CTD surface temperature on matching dates | Verify instrument agreement |
| **Gap Interpolation** | Use continuous SST to fill between sparse CTD dates | Estimate conditions during field gaps |
| **Diversity Prediction** | Predict expected diversity from CTD environmental conditions | Detect ecological anomalies |
| **Corroboration Scoring** | Count independent sources confirming each observation | Assign reliability tiers |

**Reliability tiers:**
- **Verified** — Multi-source agreement (e.g., SST confirms CTD + diversity matches prediction)
- **Supported** — Partial corroboration (one cross-check passed)
- **Standalone** — Single source only, no cross-validation available

Results are injected into LLM prompts as `[reliability_*]` citations.
""")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reliability outputs", len(_reliability_files))
        c2.metric("Reliability docs", _n_rel_docs)
        c3.metric("Verified observations", _n_verified)
        c4.metric("Total checked", _n_corrob_total)

    # Stage 6: Retrieval Documents
    with st.expander("Stage 6 -- Retrieval Document Generation"):
        st.markdown("""
Raw data is transformed into **narrative text chunks** optimized for LLM comprehension.
Each document is a self-contained paragraph with statistics, not raw CSV rows.

**Example CTD document:**
> *CTD cast 2024-06-O-St1 at Onagawa Bay on 2024-06-15. Surface temperature: 18.5C,
> bottom temperature: 12.3C (stratification index: 6.2C). Mean salinity: 33.8 PSU.
> DO saturation: 95.2%. Chlorophyll-a peak: 2.1 ug/L at 5m depth.*

**Example Metagenome document:**
> *Metagenome sample 2024-06-O at Onagawa Bay. Kraken2 classification: 716 genera detected.
> Top 5 by abundance: Synechococcus (12.3%), Pelagibacter (8.7%), Candidatus Actinomarina (5.1%),
> Prochlorococcus (4.2%), Planktomarina (3.8%). Shannon diversity: 4.21.*

Each document carries metadata: source_type, sample_id, bay, station, datetime,
and a provenance event_id linking back to the original file.
""")

        n_ctd_docs = sum(1 for d in docs if d.get("source_type") == "ctd")
        n_meta_docs = sum(1 for d in docs if d.get("source_type") == "metagenome")
        n_sst_docs = sum(1 for d in docs if d.get("source_type") == "remote_sensing")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total documents", len(docs))
        c2.metric("CTD docs", n_ctd_docs)
        c3.metric("Metagenome docs", n_meta_docs)
        c4.metric("SST docs", n_sst_docs)

    # Stage 7: Storage & Embeddings
    with st.expander("Stage 7 -- PostgreSQL Storage and Vector Embeddings"):
        st.markdown(f"""
Documents are loaded into **PostgreSQL 16 + pgvector** for production search.

**Embedding model:** `{config.EMBEDDING_MODEL}` ({config.EMBEDDING_DIM}-dim vectors)

**Database schema (9 tables):**

| Table | Purpose |
|---|---|
| `retrieval_document` | Text chunks + {config.EMBEDDING_DIM}-dim embeddings + tsvector for FTS |
| `anchor_event` | Spatiotemporal linking records |
| `ctd_profile` | Full depth-resolved CTD measurements |
| `ctd_summary` | Per-cast aggregated statistics |
| `metagenome_sample` | Per-sample sequencing metadata + top taxa |
| `sst_point_observation` | Hourly satellite SST values |
| `sst_daily_summary` | Daily regional SST aggregates |
| `cross_source_link` | Temporal links between data sources |
| `provenance_record` | File registration records |

**Infrastructure:** Podman container (`pgvector/pgvector:pg16`) on port 5433.
""")

        c1, c2, c3 = st.columns(3)
        c1.metric("Embedding model", config.EMBEDDING_MODEL)
        c2.metric("Dimensions", config.EMBEDDING_DIM)
        c3.metric("Embedded docs", pg_embed_count if USE_PG else "N/A (PG offline)")

    # Stage 8: Hybrid Retrieval
    with st.expander("Stage 8 -- Hybrid Retrieval (Vector + FTS + RRF)"):
        st.markdown("""
The retrieval engine combines two complementary search strategies:

**1. Vector Search (Semantic)**
- User query is embedded via the same model (`nomic-embed-text`)
- Cosine similarity against all document embeddings
- Captures semantic meaning: "warm water" matches "high temperature"

**2. Full-Text Search (Lexical)**
- PostgreSQL `tsvector` + `ts_rank_cd` scoring
- Exact keyword matching with stemming and stop-word removal
- Captures specific terms: "Gyrodinium", "station St1"

**3. Reciprocal Rank Fusion (RRF)**
```
RRF_score(doc) = 1/(k + rank_vector) + 1/(k + rank_fts)   where k = 60
```
This combines both rankings into a single score, ensuring documents that rank
well in both searches are surfaced first.

**4. SQL Filters**
Users can filter by source type (CTD/metagenome/SST), bay (O/I/M),
and time range -- all applied at the SQL level before scoring.

**5. Analysis Context Injection**
For complex ecosystem queries (detected via keywords: *correlation*, *diversity*,
*seasonal*, *trend*, etc.), precomputed analysis summaries are automatically
injected as supplementary context alongside retrieved evidence.
""")

        st.markdown("**Fallback:** When PostgreSQL is unavailable, the system degrades gracefully "
                     "to local BM25 + numpy cosine search with the same document corpus.")

        c1, c2 = st.columns(2)
        c1.metric("Current backend", "pgvector (hybrid)" if USE_PG else "Local BM25")
        c2.metric("Top-K retrieval", top_k_sources)

    # Stage 9: LLM Prompting
    with st.expander("Stage 9 -- Provenance-Aware LLM Prompting"):
        st.markdown(f"""
The final stage constructs a structured prompt for the LLM:

**Current model:** `{model}`

**Prompt structure:**
1. **System instructions** -- Rules for citation, data types, and study site context
2. **Retrieved evidence** -- Top-K documents with `[doc_id]` tags for citation
3. **Analysis context** -- Precomputed ecological summaries (when relevant)
4. **User question** -- The original query

**Citation rules enforced:**
- Every claim must cite a source using `[doc_id]` notation
- Distinguish between CTD measurements, metagenome taxonomy, and satellite SST
- State data gaps explicitly and report values with units
- Use `[analysis_*]` notation when citing precomputed analyses

This ensures the LLM cannot hallucinate facts -- every statement in the answer
can be traced back through the document, to the anchor event, to the original
file with its SHA-256 hash.
""")

    # ─── Pipeline flow diagram ───
    st.markdown("---")
    st.markdown("### Pipeline Flow")

    n_ctd_docs = sum(1 for d in docs if d.get("source_type") == "ctd")
    n_meta_docs = sum(1 for d in docs if d.get("source_type") == "metagenome")
    n_sst_docs = sum(1 for d in docs if d.get("source_type") == "remote_sensing")

    # ─── Graphviz data flow diagram ───
    _embed_label = f"{pg_embed_count} vectors" if USE_PG else "Local numpy"
    _backend_label = "pgvector Hybrid" if USE_PG else "BM25 Local"

    _dot = f"""
    digraph pipeline {{
        rankdir=TB;
        bgcolor="white";
        fontname="Helvetica";
        node [fontname="Helvetica", fontsize=11, style="filled,rounded", shape=box,
              color="#64748b", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=9, color="#94a3b8", penwidth=1.2];

        // ── Data Sources ──
        subgraph cluster_sources {{
            label="Data Sources";
            labeljust=l;
            fontsize=13;
            fontcolor="#334155";
            style="dashed,rounded";
            color="#cbd5e1";
            bgcolor="#f8fafc";

            ctd_raw  [label="CTD\\n1 TSV, 10,955 profiles", fillcolor="#dbeafe"];
            meta_raw [label="Metagenome\\n11 TSV files", fillcolor="#dcfce7"];
            sst_raw  [label="Satellite SST\\n1,848 NetCDF", fillcolor="#fef3c7"];
        }}

        // ── Provenance ──
        provenance [label="Provenance Registry\\n{_prov_count:,} files, SHA-256 hashes", fillcolor="#f1f5f9"];

        // ── Preprocessing ──
        subgraph cluster_preprocess {{
            label="Preprocessing";
            labeljust=l;
            fontsize=13;
            fontcolor="#334155";
            style="dashed,rounded";
            color="#cbd5e1";
            bgcolor="#f8fafc";

            ctd_pp  [label="CTD Pipeline\\nstandardize, summaries", fillcolor="#dbeafe"];
            meta_pp [label="Metagenome Pipeline\\nKraken, MetaEuk, groups", fillcolor="#dcfce7"];
            sst_pp  [label="SST Pipeline\\npoint extraction, daily agg", fillcolor="#fef3c7"];
        }}

        // ── Normalized ──
        normalized [label="Normalized Parquets\\n{len(_norm_files)} files", fillcolor="#f1f5f9"];

        // ── Canonical ──
        anchors [label="Anchor Events\\n{_n_anchors:,} spatiotemporal anchors\\n{_n_links:,} cross-source links", fillcolor="#e0e7ff"];

        // ── Pre-Analysis ──
        preanalysis [label="Pre-Analysis\\n{len(_analysis_files)} ecological analyses\\n{_n_analysis_docs} RAG documents", fillcolor="#fae8ff"];

        // ── Reliability Ensurance ──
        reliability [label="Reliability Ensurance\\n{len(_reliability_files)} validation outputs\\n{_n_verified}/{_n_corrob_total} verified", fillcolor="#d1fae5", penwidth=1.5];

        // ── Retrieval Docs ──
        ret_docs [label="Retrieval Documents\\n{len(docs)} narrative chunks\\n({n_ctd_docs} CTD + {n_meta_docs} meta + {n_sst_docs} SST)", fillcolor="#f1f5f9"];

        // ── Storage ──
        subgraph cluster_storage {{
            label="PostgreSQL + pgvector";
            labeljust=l;
            fontsize=13;
            fontcolor="#334155";
            style="dashed,rounded";
            color="#cbd5e1";
            bgcolor="#f8fafc";

            embeddings [label="Vector Embeddings\\n{_embed_label}\\n{config.EMBEDDING_DIM}-dim {config.EMBEDDING_MODEL}", fillcolor="#dbeafe"];
            fts        [label="Full-Text Index\\ntsvector + ts_rank_cd", fillcolor="#dcfce7"];
            db_tables  [label="9 Relational Tables\\nprofiles, samples, links", fillcolor="#fef3c7"];
        }}

        // ── Retrieval ──
        retrieval [label="Hybrid Retrieval\\n{_backend_label}\\nVector + FTS + RRF (k=60)", fillcolor="#e0e7ff", penwidth=2];

        // ── LLM ──
        llm [label="LLM ({model})\\nProvenance-aware prompting\\nCitation-grounded answers",
             fillcolor="#fecdd3", penwidth=2, fontsize=12];

        // ── Edges ──
        ctd_raw  -> provenance;
        meta_raw -> provenance;
        sst_raw  -> provenance;

        provenance -> ctd_pp  [label="  registered"];
        provenance -> meta_pp;
        provenance -> sst_pp;

        ctd_pp  -> normalized;
        meta_pp -> normalized;
        sst_pp  -> normalized;

        normalized -> anchors   [label="  link"];
        normalized -> preanalysis [label="  analyze"];
        normalized -> reliability [label="  validate", color="#10b981"];

        preanalysis -> reliability [label="  correlations", style=dashed, color="#10b981"];

        anchors    -> ret_docs [label="  build"];
        normalized -> ret_docs;

        ret_docs    -> embeddings [label="  embed"];
        ret_docs    -> fts        [label="  index"];
        ret_docs    -> db_tables;
        preanalysis -> db_tables;

        embeddings -> retrieval [label="  cosine"];
        fts        -> retrieval [label="  rank"];

        preanalysis -> retrieval [label="  inject", style=dashed, color="#a855f7"];
        reliability -> retrieval [label="  inject", style=dashed, color="#10b981"];

        retrieval -> llm [label="  top-K + context", penwidth=2];
    }}
    """

    st.graphviz_chart(_dot, width="stretch")

    # ─── Summary metrics row ───
    st.markdown("#### Pipeline Summary")
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Raw files", f"{_prov_count:,}")
    mc2.metric("Parquets", len(_norm_files))
    mc3.metric("Anchors", f"{_n_anchors:,}")
    mc4.metric("Documents", len(docs))
    mc5.metric("Embeddings", pg_embed_count if USE_PG else "local")
    mc6.metric("Backend", "pgvector" if USE_PG else "BM25")


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
                t_from = filter_time_from.strip() or None
                t_to = filter_time_to.strip() or None
                if USE_PG:
                    from orchestration.unified import retrieve
                    retrieved = retrieve(user_text, k=top_k_sources,
                                         source_type=src_filter, bay=bay_filter,
                                         time_from=t_from, time_to=t_to,
                                         vector_weight=vector_weight,
                                         fts_weight=fts_weight, rrf_k=rrf_k)
                else:
                    retrieved = retriever.search(user_text, k=top_k_sources,
                                                 source_type=src_filter, bay=bay_filter,
                                                 time_from=t_from, time_to=t_to)

                # Count injected context docs
                _injected_analysis_docs = []
                _injected_reliability_docs = []
                if inject_analysis:
                    _adoc_path = config.ANALYSIS_DIR / "analysis_documents.jsonl"
                    if _adoc_path.exists():
                        import json as _json
                        with open(_adoc_path, encoding="utf-8") as _af:
                            _injected_analysis_docs = [_json.loads(l) for l in _af if l.strip()]
                if inject_reliability:
                    _rdoc_path = config.RELIABILITY_DIR / "reliability_documents.jsonl"
                    if _rdoc_path.exists():
                        import json as _json
                        with open(_rdoc_path, encoding="utf-8") as _rf:
                            _injected_reliability_docs = [_json.loads(l) for l in _rf if l.strip()]

                _total_sources = len(retrieved) + len(_injected_analysis_docs) + len(_injected_reliability_docs)
                with st.expander(f"All {_total_sources} sources feeding the LLM", expanded=False):
                    # Retrieved documents
                    st.markdown(f"**Retrieved Documents ({len(retrieved)})**")
                    for r in retrieved:
                        st.markdown(
                            f"**[{r.get('id', r.get('doc_id', ''))}]** "
                            f"{r.get('title', '')}  \n"
                            f"Score: `{r.get('score', 0):.4f}` | {r.get('source_type', '')} | {r.get('time', r.get('date', ''))}"
                        )
                        st.caption(_trunc(r.get("text", ""), 300))

                    # Pre-analysis context
                    if _injected_analysis_docs:
                        st.markdown("---")
                        st.markdown(f"**Pre-Analysis Context ({len(_injected_analysis_docs)})**")
                        for ad in _injected_analysis_docs:
                            st.markdown(
                                f"**[{ad.get('id', '')}]** "
                                f"{ad.get('title', ad.get('analysis_type', ''))}"
                            )
                            st.caption(_trunc(ad.get("text", ""), 300))

                    # Reliability context
                    if _injected_reliability_docs:
                        st.markdown("---")
                        st.markdown(f"**Reliability Context ({len(_injected_reliability_docs)})**")
                        for rd in _injected_reliability_docs:
                            st.markdown(
                                f"**[{rd.get('id', '')}]** "
                                f"{rd.get('title', rd.get('analysis_type', ''))}"
                            )
                            st.caption(_trunc(rd.get("text", ""), 300))

                # Build prompt
                from orchestration.unified import build_prompt
                prompt_text = build_prompt(user_text, retrieved,
                                           inject_analysis=inject_analysis,
                                           inject_reliability=inject_reliability)

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
                            "options": {
                                "temperature": temperature,
                                "top_p": top_p,
                                "repeat_penalty": repeat_penalty,
                                "num_ctx": num_ctx,
                            },
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
                        full = f"LLM error: {e}"
                    placeholder.markdown(full)

                st.session_state.messages.append({"role": "assistant", "content": full})

    with col_sources:
        st.subheader("Corpus")
        st.metric("Documents", len(docs))
        src_counts = Counter(d.get("source_type", "") for d in docs)
        for src, cnt in sorted(src_counts.items()):
            icon = {"ctd": "", "metagenome": "", "remote_sensing": ""}.get(src, "")
            st.caption(f"{icon} {src}: {cnt}")


# ═══════════════════════════════════════════
# TAB: Evidence Explorer
# ═══════════════════════════════════════════
with tab_explore:
    st.subheader("Evidence Explorer")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        eq = st.text_input("Search documents", placeholder="e.g. dinoflagellate, temperature June")
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
            icon = {"ctd": "", "metagenome": "", "remote_sensing": ""}.get(r.get("source_type"), "")
            with st.expander(f"{icon} {r.get('title', r.get('id', ''))} — score {r.get('score',0):.4f}"):
                st.markdown(r.get("text", ""))
                st.caption(f"doc_id: {r.get('id', r.get('doc_id', ''))} | "
                           f"sample: {r.get('sample_id', '–')} | "
                           f"event: {r.get('event_id', '–')}")
    else:
        st.info("Enter a search query above to explore the evidence base.")


# ═══════════════════════════════════════════
# TAB: Data (CTD + Taxa + SST)
# ═══════════════════════════════════════════
with tab_data:
    st.subheader("Data")
    st.caption("Browse CTD profiles, metagenome taxonomy, and satellite SST observations.")

    data_tab_ctd, data_tab_taxa, data_tab_sst = st.tabs(
        ["CTD Profiles", "Taxa", "SST"]
    )

    # ── Sub-tab: CTD Profiles ──
    with data_tab_ctd:
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

    # ── Sub-tab: Taxa ──
    with data_tab_taxa:
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

    # ── Sub-tab: SST ──
    with data_tab_sst:
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
                                 alpha=0.2, color="#1976D2", label="min-max range")
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
    st.subheader("Corpus Statistics")

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
# TAB: Pre-Analysis
# ═══════════════════════════════════════════
with tab_analysis:
    st.subheader("Pre-Analysis")
    st.caption("Precomputed ecological relationships across CTD, metagenome, and SST data.")

    # Load analysis data
    _analysis_dir = config.ANALYSIS_DIR
    _has_analysis = _analysis_dir.exists() and any(_analysis_dir.glob("*.parquet"))

    if not _has_analysis:
        st.warning("No pre-analysis data found. Run `python scripts/run_pre_analysis.py` to compute.")
    else:
        pa_tab1, pa_tab2, pa_tab3, pa_tab4, pa_tab5 = st.tabs(
            ["CTD Trends", "Correlations", "Diversity", "Co-occurrence", "Reliability"]
        )

        # ── Sub-tab 1: CTD Trends ──
        with pa_tab1:
            trends_path = _analysis_dir / "ctd_monthly_trends.parquet"
            if trends_path.exists():
                trends_df = pd.read_parquet(trends_path)

                st.markdown("### Monthly CTD variable trends by bay")

                # Variable selector
                trend_vars = [
                    ("mean_temperature", "Temperature (°C)", "#e74c3c"),
                    ("mean_salinity", "Salinity (PSU)", "#3498db"),
                    ("mean_do_percent", "Dissolved Oxygen (%)", "#2ecc71"),
                    ("mean_chl_a", "Chlorophyll-a (μg/L)", "#27ae60"),
                    ("mean_turbidity", "Turbidity", "#95a5a6"),
                ]

                sel_var = st.selectbox(
                    "Variable",
                    [v[1] for v in trend_vars],
                    key="pa_trend_var",
                )
                var_key = [v[0] for v in trend_vars if v[1] == sel_var][0]
                var_color = [v[2] for v in trend_vars if v[1] == sel_var][0]

                mean_col = f"{var_key}_mean"
                std_col = f"{var_key}_std"

                if mean_col in trends_df.columns:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(12, 4.5))

                    for bay in sorted(trends_df["bay"].dropna().unique()):
                        bd = trends_df[trends_df["bay"] == bay].sort_values("year_month")
                        x = range(len(bd))
                        y = bd[mean_col].values
                        yerr = bd[std_col].values if std_col in bd.columns else None

                        bay_label = {"O": "Onagawa", "I": "Ishinomaki", "M": "Matsushima"}.get(bay, bay)
                        ax.plot(x, y, "o-", label=bay_label, markersize=5)
                        if yerr is not None:
                            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)

                        ax.set_xticks(range(len(bd)))
                        ax.set_xticklabels(bd["year_month"].values, rotation=45, ha="right",
                                           fontsize=8)

                    ax.set_ylabel(sel_var, fontsize=11)
                    ax.set_xlabel("Month", fontsize=11)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)

                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                    # Stratification index
                    if "strat_index_mean" in trends_df.columns:
                        with st.expander("Thermal Stratification Index", expanded=False):
                            st.caption("Surface T − Bottom T: positive = stratified, ~0 = mixed")
                            strat = trends_df.dropna(subset=["strat_index_mean"]).sort_values("year_month")
                            fig2, ax2 = plt.subplots(figsize=(12, 3))

                            for bay in sorted(strat["bay"].dropna().unique()):
                                bs = strat[strat["bay"] == bay]
                                ax2.bar(range(len(bs)), bs["strat_index_mean"].values,
                                        alpha=0.7, label={"O": "Onagawa", "I": "Ishinomaki", "M": "Matsushima"}.get(bay, bay))
                                ax2.set_xticks(range(len(bs)))
                                ax2.set_xticklabels(bs["year_month"].values, rotation=45, ha="right",
                                                    fontsize=7)

                            ax2.axhline(0, color="#888", linewidth=0.8)
                            ax2.set_ylabel("ΔT (°C)", fontsize=10)
                            ax2.legend(fontsize=8)

                            fig2.tight_layout()
                            st.pyplot(fig2, width="stretch")
                            plt.close(fig2)

                    # Data table
                    with st.expander("Raw trend data"):
                        st.dataframe(trends_df, width="stretch", height=300)
            else:
                st.info("No CTD trend data available.")

        # ── Sub-tab 2: Correlations ──
        with pa_tab2:
            corr_path = _analysis_dir / "taxa_env_correlations.parquet"
            if corr_path.exists():
                corr_df = pd.read_parquet(corr_path)

                st.markdown("### Taxa–Environment Correlations (Spearman)")

                # Summary metrics
                n_sig = corr_df[corr_df["significant"]].shape[0]
                n_total = len(corr_df)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total pairs tested", n_total)
                c2.metric("Significant (p<0.05)", n_sig)
                c3.metric("% Significant", f"{n_sig/n_total*100:.0f}%" if n_total > 0 else "–")

                # Heatmap
                import matplotlib.pyplot as plt

                pivot = corr_df.pivot_table(
                    index="genus", columns="env_variable",
                    values="spearman_rho", aggfunc="first"
                )
                # Clean up column names for display
                pivot.columns = [c.replace("mean_", "") for c in pivot.columns]

                fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))

                im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=9)

                # Annotate cells
                sig_pivot = corr_df.pivot_table(
                    index="genus", columns="env_variable",
                    values="significant", aggfunc="first"
                )
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.iloc[i, j]
                        if pd.notna(val):
                            genus = pivot.index[i]
                            env = corr_df["env_variable"].unique()[j] if j < len(corr_df["env_variable"].unique()) else ""
                            is_sig = sig_pivot.iloc[i, j] if i < len(sig_pivot) and j < len(sig_pivot.columns) else False
                            txt = f"{val:.2f}"
                            if is_sig:
                                txt += "*"
                            color = "white" if abs(val) > 0.5 else "black"
                            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Spearman ρ")
                ax.set_title("Taxa × Environment Correlation Heatmap", fontsize=13, pad=12)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                st.caption("* = p < 0.05. Positive ρ = genus abundance increases with variable.")

                # Significant correlations table
                with st.expander("Significant correlations (p<0.05)"):
                    sig = corr_df[corr_df["significant"]].sort_values("p_value")
                    st.dataframe(sig, width="stretch", hide_index=True)
            else:
                st.info("No correlation data. Run `python scripts/run_pre_analysis.py`.")

        # ── Sub-tab 3: Diversity ──
        with pa_tab3:
            div_path = _analysis_dir / "diversity_indices.parquet"
            if div_path.exists():
                div_df = pd.read_parquet(div_path)

                st.markdown("### Community Diversity Indices")

                # Source selector
                div_source = st.selectbox("Method", div_df["source"].unique().tolist(), key="pa_div_src")
                sd = div_df[div_df["source"] == div_source].copy()

                # Summary metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Samples", len(sd))
                mc2.metric("Mean Shannon H'", f"{sd['shannon_h'].mean():.3f}")
                mc3.metric("Mean Simpson", f"{sd['simpson_1d'].mean():.4f}")
                mc4.metric("Mean Richness", f"{sd['richness'].mean():.0f}")

                # Shannon diversity over time
                import matplotlib.pyplot as plt

                sd_sorted = sd.sort_values("year_month")

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                for ax_i, (metric, label, color) in enumerate([
                    ("shannon_h", "Shannon H'", "#e74c3c"),
                    ("richness", "Genus Richness", "#3498db"),
                ]):
                    ax = axes[ax_i]

                    for bay in sorted(sd_sorted["bay"].dropna().unique()):
                        bsd = sd_sorted[sd_sorted["bay"] == bay]
                        bay_label = {"O": "Onagawa", "I": "Ishinomaki", "M": "Matsushima"}.get(bay, bay)
                        ax.scatter(range(len(bsd)), bsd[metric].values, s=20, label=bay_label, alpha=0.7)

                    ax.set_ylabel(label, fontsize=11)
                    ax.set_xlabel("Sample", fontsize=10)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                # Data table
                with st.expander("Full diversity data"):
                    st.dataframe(sd, width="stretch", height=300, hide_index=True)
            else:
                st.info("No diversity data. Run `python scripts/run_pre_analysis.py`.")

        # ── Sub-tab 4: Co-occurrence ──
        with pa_tab4:
            cooc_path = _analysis_dir / "taxa_cooccurrence.parquet"
            if cooc_path.exists():
                cooc_df = pd.read_parquet(cooc_path)

                st.markdown("### Taxa Co-occurrence (Jaccard Similarity)")
                st.caption("Jaccard index: proportion of samples where both genera are present. 1.0 = always co-occur, 0.0 = never.")

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 10))

                # Mask diagonal for cleaner viz
                vals = cooc_df.values.copy()
                np.fill_diagonal(vals, np.nan)

                im = ax.imshow(vals, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
                ax.set_xticks(range(len(cooc_df.columns)))
                ax.set_xticklabels(cooc_df.columns, rotation=90, fontsize=8)
                ax.set_yticks(range(len(cooc_df.index)))
                ax.set_yticklabels(cooc_df.index, fontsize=8)

                cbar = fig.colorbar(im, ax=ax, shrink=0.7)
                cbar.set_label("Jaccard Index")
                ax.set_title("Genus Co-occurrence Heatmap (Top 30)", fontsize=13, pad=12)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                # Top co-occurring pairs
                with st.expander("Top co-occurring pairs"):
                    pairs = []
                    genera = cooc_df.index.tolist()
                    for i in range(len(genera)):
                        for j in range(i + 1, len(genera)):
                            pairs.append({
                                "Genus A": genera[i],
                                "Genus B": genera[j],
                                "Jaccard": round(cooc_df.iloc[i, j], 4),
                            })
                    pairs_df = pd.DataFrame(pairs).sort_values("Jaccard", ascending=False).head(20)
                    st.dataframe(pairs_df, width="stretch", hide_index=True)
            else:
                st.info("No co-occurrence data. Run `python scripts/run_pre_analysis.py`.")

        # ── Sub-tab 5: Reliability ──
        with pa_tab5:
            _rel_dir = config.RELIABILITY_DIR
            _has_reliability = _rel_dir.exists() and any(_rel_dir.glob("*.parquet"))

            if not _has_reliability:
                st.warning("No reliability data found. Run `python scripts/run_reliability.py` to compute.")
            else:
                st.markdown("### Cross-Source Reliability Ensurance")
                st.caption("Validation, interpolation, and corroboration across CTD, metagenome, and SST sources.")

                # ── SST ↔ CTD Validation ──
                sst_ctd_path = _rel_dir / "sst_ctd_validation.parquet"
                if sst_ctd_path.exists():
                    sst_ctd_df = pd.read_parquet(sst_ctd_path)
                    if not sst_ctd_df.empty:
                        with st.expander("SST ↔ CTD Surface Temperature Validation", expanded=True):
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            n_agree = int(sst_ctd_df["agrees"].sum())
                            mc1.metric("Paired obs", len(sst_ctd_df))
                            mc2.metric("Agreement", f"{n_agree}/{len(sst_ctd_df)}")
                            mc3.metric("Mean |ΔT|", f"{sst_ctd_df['abs_delta_t'].mean():.2f}°C")
                            mc4.metric("Mean score", f"{sst_ctd_df['reliability_score'].mean():.3f}")

                            import matplotlib.pyplot as plt

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                            # Scatter: CTD vs SST
                            colors = ["#10b981" if a else "#ef4444" for a in sst_ctd_df["agrees"]]
                            ax1.scatter(sst_ctd_df["sst_daily_mean"], sst_ctd_df["ctd_surface_t"],
                                       c=colors, s=40, alpha=0.8, edgecolors="white", linewidth=0.5)
                            lims = [
                                min(sst_ctd_df["sst_daily_mean"].min(), sst_ctd_df["ctd_surface_t"].min()) - 1,
                                max(sst_ctd_df["sst_daily_mean"].max(), sst_ctd_df["ctd_surface_t"].max()) + 1,
                            ]
                            ax1.plot(lims, lims, "--", color="#6b7280", linewidth=1, label="1:1 line")
                            threshold = config.SST_CTD_AGREEMENT_THRESHOLD
                            ax1.fill_between(lims, [l - threshold for l in lims],
                                           [l + threshold for l in lims],
                                           alpha=0.1, color="#10b981", label=f"±{threshold}°C band")
                            ax1.set_xlabel("Satellite SST (°C)", fontsize=11)
                            ax1.set_ylabel("CTD Surface T (°C)", fontsize=11)
                            ax1.set_title("SST vs CTD Surface Temperature", fontsize=12)
                            ax1.legend(fontsize=9)
                            ax1.grid(True, alpha=0.3)

                            # Bar: ΔT by sample
                            bar_colors = ["#10b981" if a else "#ef4444" for a in sst_ctd_df["agrees"]]
                            ax2.bar(range(len(sst_ctd_df)), sst_ctd_df["delta_t"].values,
                                   color=bar_colors, alpha=0.8)
                            ax2.axhline(0, color="#6b7280", linewidth=0.8)
                            ax2.axhline(threshold, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.5)
                            ax2.axhline(-threshold, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.5)
                            ax2.set_xlabel("Observation", fontsize=11)
                            ax2.set_ylabel("ΔT (CTD − SST) °C", fontsize=11)
                            ax2.set_title("Temperature Difference per Cast", fontsize=12)
                            ax2.grid(True, alpha=0.3)

                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                            with st.expander("Validation data"):
                                st.dataframe(sst_ctd_df, width="stretch", hide_index=True)

                # ── Gap Interpolation ──
                gap_path = _rel_dir / "gap_interpolation.parquet"
                if gap_path.exists():
                    gap_df = pd.read_parquet(gap_path)
                    if not gap_df.empty:
                        with st.expander("Temporal Gap Interpolation", expanded=False):
                            gap_df["date_dt"] = pd.to_datetime(gap_df["date"])
                            gaps_only = gap_df[gap_df["in_ctd_gap"]]

                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("SST days", len(gap_df))
                            mc2.metric("In CTD gaps", len(gaps_only))
                            mc3.metric("Mean confidence",
                                      f"{gaps_only['confidence'].mean():.3f}" if not gaps_only.empty else "–")

                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(figsize=(14, 4))
                            ax.plot(gap_df["date_dt"], gap_df["sst_daily_mean"],
                                   linewidth=0.8, color="#3b82f6", alpha=0.6, label="SST daily mean")
                            ax.plot(gap_df["date_dt"], gap_df["interpolated_surface_t"],
                                   linewidth=1.2, color="#10b981", label="Interpolated surface T")

                            # Mark CTD observation dates
                            ctd_s_path = config.NORMALIZED_DIR / "ctd_summary.parquet"
                            if ctd_s_path.exists():
                                ctd_s = pd.read_parquet(ctd_s_path)
                                ctd_s["ctd_date"] = pd.to_datetime(ctd_s["ctd_date"], errors="coerce")
                                ctd_dates = ctd_s["ctd_date"].dropna().unique()
                                for cd in ctd_dates:
                                    ax.axvline(cd, color="#ef4444", alpha=0.3, linewidth=0.5)
                                ax.axvline(cd, color="#ef4444", alpha=0.3, linewidth=0.5, label="CTD dates")

                            ax.set_xlabel("Date", fontsize=11)
                            ax.set_ylabel("Temperature (°C)", fontsize=11)
                            ax.set_title("SST-based Gap Interpolation", fontsize=12)
                            ax.legend(fontsize=9)
                            ax.grid(True, alpha=0.3)
                            fig.autofmt_xdate()
                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                # ── Diversity Prediction ──
                div_pred_path = _rel_dir / "diversity_prediction.parquet"
                if div_pred_path.exists():
                    div_pred_df = pd.read_parquet(div_pred_path)
                    if not div_pred_df.empty:
                        with st.expander("Diversity Prediction vs Actual", expanded=False):
                            n_anom = int(div_pred_df["is_anomaly"].sum())
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Samples", len(div_pred_df))
                            mc2.metric("Anomalies", n_anom)
                            mc3.metric("Mean |deviation|",
                                      f"{div_pred_df['deviation_sigma'].abs().mean():.2f}σ")

                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(figsize=(8, 6))
                            normal = div_pred_df[~div_pred_df["is_anomaly"]]
                            anomaly = div_pred_df[div_pred_df["is_anomaly"]]

                            ax.scatter(normal["predicted_shannon"], normal["actual_shannon"],
                                      c="#10b981", s=50, alpha=0.8, label="Normal", edgecolors="white")
                            if not anomaly.empty:
                                ax.scatter(anomaly["predicted_shannon"], anomaly["actual_shannon"],
                                          c="#ef4444", s=80, alpha=0.9, label="Anomaly",
                                          edgecolors="white", marker="D")
                                for _, r in anomaly.iterrows():
                                    ax.annotate(r["sample_id"], (r["predicted_shannon"], r["actual_shannon"]),
                                               fontsize=7, alpha=0.8, xytext=(5, 5),
                                               textcoords="offset points")

                            lims = [
                                min(div_pred_df["predicted_shannon"].min(),
                                    div_pred_df["actual_shannon"].min()) - 0.2,
                                max(div_pred_df["predicted_shannon"].max(),
                                    div_pred_df["actual_shannon"].max()) + 0.2,
                            ]
                            ax.plot(lims, lims, "--", color="#6b7280", linewidth=1, label="1:1")
                            ax.set_xlabel("Predicted Shannon H'", fontsize=11)
                            ax.set_ylabel("Actual Shannon H'", fontsize=11)
                            ax.set_title("Diversity: Predicted vs Actual", fontsize=12)
                            ax.legend(fontsize=9)
                            ax.grid(True, alpha=0.3)
                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                            with st.expander("Prediction data"):
                                st.dataframe(div_pred_df, width="stretch", hide_index=True)

                # ── Corroboration Summary ──
                corrob_path = _rel_dir / "corroboration.parquet"
                if corrob_path.exists():
                    corrob_df = pd.read_parquet(corrob_path)
                    if not corrob_df.empty:
                        with st.expander("Corroboration Summary", expanded=False):
                            tier_counts = corrob_df["reliability_tier"].value_counts().to_dict()
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Total", len(corrob_df))
                            mc2.metric("Verified", tier_counts.get("verified", 0))
                            mc3.metric("Supported", tier_counts.get("supported", 0))
                            mc4.metric("Standalone", tier_counts.get("standalone", 0))

                            import matplotlib.pyplot as plt

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                            # Pie chart of tiers
                            tier_labels = ["Verified", "Supported", "Standalone"]
                            tier_vals = [tier_counts.get(t.lower(), 0) for t in tier_labels]
                            tier_colors = ["#10b981", "#f59e0b", "#ef4444"]
                            ax1.pie(tier_vals, labels=tier_labels, colors=tier_colors,
                                   autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
                            ax1.set_title("Reliability Tiers", fontsize=12)

                            # Score distribution
                            ax2.hist(corrob_df["reliability_score"], bins=20,
                                    color="#3b82f6", alpha=0.7, edgecolor="white")
                            ax2.set_xlabel("Reliability Score", fontsize=11)
                            ax2.set_ylabel("Count", fontsize=11)
                            ax2.set_title("Score Distribution", fontsize=12)
                            ax2.grid(True, alpha=0.3)

                            fig.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                            st.dataframe(
                                corrob_df[["sample_id", "source_type", "reliability_tier",
                                          "reliability_score", "n_checks", "detail"]],
                                width="stretch", hide_index=True, height=400,
                            )


# ═══════════════════════════════════════════
# TAB: Database Explorer
# ═══════════════════════════════════════════
with tab_db:
    st.subheader("Database Explorer")

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
            ["Table Browser", "SQL Console", "Schema", "Embeddings"]
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
                        f"Download {selected_table} ({len(df_result)} rows)",
                        data=csv_data,
                        file_name=f"{selected_table}_export.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Query error: {e}")

                # Row detail inspector
                if "df_result" in dir() and not df_result.empty:
                    with st.expander("Row Inspector", expanded=False):
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
                run_sql = st.button("Run Query", key="db_run_sql")

            if run_sql and sql_input.strip():
                # Safety: block destructive statements
                sql_upper = sql_input.strip().upper()
                blocked = ["DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE", "CREATE"]
                if any(sql_upper.startswith(kw) for kw in blocked):
                    st.error("Only SELECT queries are allowed.")
                else:
                    try:
                        import time as _time
                        t0 = _time.perf_counter()
                        df_sql = pd.read_sql(text(sql_input), _db_engine)
                        elapsed = _time.perf_counter() - t0

                        with col_info:
                            st.caption(f"{len(df_sql)} rows in {elapsed:.3f}s")

                        # Hide embedding columns
                        display = [c for c in df_sql.columns if c not in ("embedding", "text_tsv")]
                        st.dataframe(df_sql[display], width="stretch", height=400)

                        st.download_button(
                            f"Download result ({len(df_sql)} rows)",
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

                with st.expander(f"**{tname}** ({len(cols)} columns)", expanded=False):
                    # Column table
                    schema_rows = []
                    pk_cols = set(pk.get("constrained_columns", []))
                    for c in cols:
                        schema_rows.append({
                            "Column": c["name"],
                            "Type": str(c["type"]),
                            "Nullable": "✓" if c.get("nullable", True) else "✗",
                            "PK": "" if c["name"] in pk_cols else "",
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
                st.success(f"100% embedding coverage ({config.EMBEDDING_MODEL}, {config.EMBEDDING_DIM}-dim)")
            elif coverage_pct > 0:
                st.warning(f"{coverage_pct:.0f}% embedding coverage — run `scripts/load_db.py --embed`")
            else:
                st.error("No embeddings — run `scripts/load_db.py --embed`")

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
            st.markdown("### Similarity Probe")
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
