# provenance-eco-rag

Onagawa Source Chat is a provenance-aware Retrieval-Augmented Generation (RAG) application for marine environmental monitoring in Miyagi Prefecture, Japan. It transforms fragmented field data (CTD water profiles, metagenome sequencing results, and satellite sea surface temperature observations) into a searchable, citation-grounded question-answering system.

Every LLM-generated answer links back to its original data sources with traceable provenance. For complex ecosystem questions, the system draws on **precomputed ecological analyses** (correlations, diversity indices, temporal trends) and **cross-source reliability validation** (SST↔CTD agreement, diversity prediction, corroboration scoring) that go beyond what any single retrieved document can provide.

## Study Sites

* **Onagawa Bay** (~38.44 N, 141.45 E): CTD, Metagenome, SST
* **Ishinomaki Bay** (~38.41 N, 141.30 E): CTD, Metagenome
* **Matsushima Bay** (~38.35 N, 141.06 E): CTD, Metagenome

## Architecture

The system follows a 9-layer pipeline:
1. **Ingestion**: Provenance registry via SHA-256 hashes.
2. **Preprocessing**: Standardization of CTD profiles, metagenome abundance (Kraken/MetaEuk), and SST netCDFs.
3. **Canonical Schema**: Anchor events for spatiotemporal linking.
4. **Pre-Analysis**: Precomputed ecological relationships (trends, correlations, diversity, co-occurrence).
5. **Reliability Ensurance**: Cross-source validation, gap interpolation, diversity prediction, and corroboration scoring.
6. **Retrieval Documents**: Generation of narrative text chunks from raw data.
7. **Database**: PostgreSQL 16 + pgvector storing embeddings and metadata.
8. **Retrieval**: Hybrid search (vector + Full-Text Search + Reciprocal Rank Fusion) with analysis and reliability context injection.
9. **Application**: Streamlit interface with chat, data exploration, and analytics.

## Technology Stack

* **Language**: Python 3.12
* **Database**: PostgreSQL 16 + pgvector
* **Container**: Podman (or Docker)
* **LLM**: Ollama (local) - qwen2.5:14b-instruct
* **Embeddings**: nomic-embed-text (768-dim)
* **Libraries**: Pandas, Parquet, xarray, netCDF4, SciPy, SQLAlchemy, Streamlit

## Setup and Installation

### Prerequisites

* Python 3.12
* Podman or Docker
* Ollama

### Environment Setup

1. Install Python dependencies:
   ```bash
   pip install streamlit pandas sqlalchemy psycopg2-binary pgvector xarray netcdf4 requests numpy matplotlib scipy
   ```

2. Start the database infrastructure:
   ```bash
   podman machine start
   podman compose up -d
   ```
   This will start PostgreSQL with pgvector on port 5433.

3. Pull the embedding model and chat model via Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:14b-instruct
   ```

### Data Pipeline Execution

Run the following scripts in order to build the database from raw data:

```bash
# 1. Ingestion and Preprocessing
python scripts/ingest.py

# 2. Build retrieval documents and spatiotemporal links
python scripts/build_retrieval_docs.py

# 3. Precompute ecological analyses
python scripts/run_pre_analysis.py

# 4. Run cross-source reliability validation
python scripts/run_reliability.py

# 5. Populate database and generate embeddings
python scripts/load_db.py --reset --embed
```

## Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The application provides a multi-tab interface:
* **Overview**: System pipeline architecture (Graphviz) with live metrics covering all 9 pipeline stages.
* **Chat**: Streaming LLM chat with provenance-aware RAG, source citations, and automatic analysis/reliability context injection.
* **Evidence Explorer**: Search documents by keyword, source type, and bay.
* **Data**: Interactive CTD depth profiles, metagenome composition (Kraken/MetaEuk), and SST time series.
* **Pre-Analysis**: Ecological trends, correlations, diversity, co-occurrence matrices, and cross-source reliability validation.
* **Database**: Read-only SQL console, table browser, schema inspector, and embedding statistics.
* **Stats**: Corpus metrics and provenance tracking.

### Application Screenshots

![Overview Tab](docs/screenshots/overview_tab.png)
*System Overview tab showing the high-level metrics and configuration sidebar.*

![Data (CTD) Tab](docs/screenshots/data_ctd_tab.png)
*Data tab showing interactive depth profiles for CTD measurements.*

![Pre-Analysis Tab](docs/screenshots/pre_analysis_tab.png)
*Pre-Analysis tab showing precomputed ecological relationships and taxa-environment correlations.*

![Database Explorer Tab](docs/screenshots/database_tab.png)
*Database tab allowing inspection of the underlying relational tables.*

![Stats Tab](docs/screenshots/stats_tab.png)
*Stats tab detailing corpus statistics and data sample coverage.*

## Retrieval System Details

The hybrid retrieval system merges vector search (cosine similarity via pgvector) and full-text search (tsvector via PostgreSQL) using Reciprocal Rank Fusion (RRF).

For complex queries containing keywords like "correlation", "diversity", or "trend", the query orchestrator automatically injects precomputed analysis summaries as supplementary context. For queries involving data confidence, validation, or anomalies, cross-source reliability documents are also injected — providing the LLM with corroboration evidence and data quality assessments.

---

# IN DEPTH INFORMATION ABOUT THE PROJECT

> **Last updated:** 2026-05-11 · **Codebase:** 37 Python files, 7,178 lines
>

## What This Project Is

**Onagawa Source Chat** is a provenance-aware Retrieval-Augmented Generation (RAG) application for marine environmental monitoring in Miyagi Prefecture, Japan. It transforms fragmented field data — CTD water profiles, metagenome sequencing results, and satellite sea surface temperature (SST) observations — into a searchable, citation-grounded question-answering system.

Every LLM-generated answer links back to its original data sources with traceable provenance. For complex ecosystem questions, the system also draws on **precomputed ecological analyses** (correlations, diversity indices, temporal trends) and **cross-source reliability validation** (SST↔CTD agreement, gap interpolation, diversity prediction, corroboration scoring) that go beyond what any single retrieved document can provide.

### Study Sites

| Bay | Code | Coordinates | Data coverage |
| --- | --- | --- | --- |
| Onagawa Bay | O | ~38.44°N, 141.45°E | CTD + Metagenome + SST |
| Ishinomaki Bay | I | ~38.41°N, 141.30°E | CTD + Metagenome |
| Matsushima Bay | M | ~38.35°N, 141.06°E | CTD + Metagenome |

---

## Architecture

The system follows a **9-layer pipeline** from raw data to grounded LLM responses:

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        CTD["PlanDyo CTD\\n1 TSV, 10,955 profiles"]
        META["PlanDyo Metagenome\\n11 TSV files"]
        SST["Himawari SST\\n1,848 NetCDF files"]
    end

    subgraph L1["Layer 1: Ingestion"]
        PROV["Provenance Registry\\nSHA-256, 1,849 records"]
    end

    subgraph L2["Layer 2: Preprocessing"]
        CTD_PP["CTD Pipeline\\nstandardize, profiles, summaries"]
        META_PP["Metagenome Pipeline\\nQC, Kraken, MetaEuk, groups"]
        SST_PP["SST Pipeline\\nNetCDF → point + regional"]
    end

    subgraph L3["Layer 3: Canonical Schema"]
        ANCHOR["Anchor Events\\n286 spatiotemporal anchors"]
        NORM["16 Parquet files"]
    end

    subgraph L3b["Layer 3.5: Pre-Analysis"]
        TRENDS["CTD Trends\\n27 monthly aggregates"]
        CORR["Correlations\\n100 Spearman pairs"]
        DIV["Diversity\\n164 Shannon/Simpson"]
        COOC["Co-occurrence\\n30×30 Jaccard"]
    end

    subgraph L3c["Layer 3.5b: Reliability Ensurance"]
        SSTCTD["SST↔CTD Validation\\n24 paired, 100% agree"]
        GAP["Gap Interpolation\\n79 days filled"]
        DIVPRED["Diversity Prediction\\n37 samples, 1 anomaly"]
        CORROB["Corroboration\\n37 verified / 207 total"]
    end

    subgraph L4["Layer 4: Retrieval Documents"]
        DOCS["323 Text Documents\\n162 CTD + 82 meta + 79 SST"]
        LINKS["496 Cross-Source Links"]
        ADOCS["5 Analysis + 4 Reliability Documents"]
    end

    subgraph L5["Layer 5: Database"]
        PG["PostgreSQL 16 + pgvector\\n9 tables, 768-dim embeddings"]
    end

    subgraph L6["Layer 6: Retrieval"]
        HYB["Hybrid Retriever\\nVector + FTS + RRF fusion\\n+ Analysis + Reliability injection"]
    end

    subgraph L7["Layer 7: Application"]
        APP["Streamlit App\\n7 tabs, streaming LLM chat"]
    end

    Sources --> L1 --> L2 --> L3 --> L3b --> L3c --> L4 --> L5 --> L6 --> L7
```

---

## Technology Stack

| Component | Technology |
| --- | --- |
| **Language** | Python 3.12 (37 files, 7,178 lines) |
| **Database** | PostgreSQL 16 + pgvector (cosine similarity) |
| **Container** | Podman 5.4.2 (Apple HV) |
| **LLM** | Ollama (local) — qwen2.5:14b-instruct |
| **Embeddings** | nomic-embed-text (768-dim, 274MB) |
| **Data** | Pandas, Parquet, xarray, netCDF4, SciPy |
| **ORM** | SQLAlchemy 2.x |
| **UI** | Streamlit |
| **Search** | pgvector cosine + tsvector FTS + Reciprocal Rank Fusion |

---

## Project Structure

```
source_chat_agt/
├── app.py                              # Streamlit application (7 tabs, ~1,780 lines)
├── config.py                           # Centralized config (paths, DB, models, thresholds)
├── docker-compose.yml                  # Podman/Docker – PostgreSQL + pgvector
│
├── ingestion/
│   ├── provenance.py                   # SHA-256 file registration (JSONL)
│   └── file_inventory.py              # Directory scanner
│
├── preprocessing/
│   ├── common.py                       # Sample ID parsing, TSV I/O
│   ├── ctd.py                          # CTD load → standardize → summaries
│   ├── metagenome.py                  # Kraken/MetaEuk abundance, QC, groups
│   ├── remote_sensing.py             # NetCDF SST extraction
│   ├── pre_analysis.py               # ★ Ecological pre-analysis (5 analyses)
│   └── reliability_ensurance.py      # ★ Cross-source validation (4 engines)
│
├── schema/
│   └── anchor_event.py                # Spatiotemporal linking
│
├── retrieval/
│   ├── document_builder.py            # CTD/meta/SST → narrative text chunks
│   ├── cross_source_linker.py         # same_sample + time_match links
│   ├── hybrid_retriever.py            # pgvector + FTS + SQL filters + RRF
│   └── local_retriever.py            # BM25 + numpy fallback (no PG needed)
│
├── db/
│   ├── models.py                       # 9 SQLAlchemy ORM tables
│   ├── connection.py                  # Engine, sessions, init_db
│   └── vector_store.py               # Ollama embedding + cosine search
│
├── orchestration/
│   ├── query_orchestrator.py          # Evidence expansion via cross-source links
│   └── unified.py                     # Auto-detect PG vs local, prompt builder
│                                        + ★ analysis + reliability context injection
│
├── scripts/
│   ├── ingest.py                       # Full ingestion pipeline CLI
│   ├── build_retrieval_docs.py        # Anchor events + docs + links
│   ├── load_db.py                     # Populate PostgreSQL + embeddings
│   ├── run_pre_analysis.py           # ★ Pre-analysis pipeline CLI
│   └── run_reliability.py            # ★ Reliability ensurance pipeline CLI
│
└── data/
    ├── raw/ctd/                        # 1 file (CTD_Onagawa.tsv, 1.2MB)
    ├── raw/meta/                       # 11 files (Kraken, MetaEuk, QC)
    ├── normalized/                     # 16 parquet files
    ├── canonical/                      # anchor_events, cross_source_links
    ├── serving/                        # retrieval docs, embeddings, registry
    ├── analysis/                       # ★ 6 pre-analysis outputs
    ├── reliability/                    # ★ 5 reliability ensurance outputs
    └── provenance/                     # provenance.jsonl (1,849 records)
```

---

## Data Statistics

### Raw Input

| Source | Files | Size | Time range |
| --- | --- | --- | --- |
| CTD (Onagawa) | 1 TSV | 1.2 MB | Jan 2024 – Mar 2026 |
| Metagenome | 11 TSV/TXT | 34 MB | Apr 2024 – Feb 2026 |
| Satellite SST | 1,848 NetCDF | ~3.7 GB | Dec 2025 – Feb 2026 |
| **Total registered** | **1,849 files** | — | — |

### Processed Output

| Dataset | Records | Format |
| --- | --- | --- |
| CTD profiles (standardized) | 10,955 depth points | Parquet |
| CTD cast summaries | 162 casts | Parquet |
| Kraken genus abundance | 58,712 (716 genera × 82 samples) | Parquet |
| MetaEuk genus abundance | 67,240 (820 genera × 82 samples) | Parquet |
| Upper taxonomic groups | 82 samples | Parquet |
| SST hourly observations | 1,848 points | Parquet |
| SST daily summaries | 79 days | Parquet |
| Anchor events | 286 (207 sample + 79 SST) | Parquet |
| Retrieval documents | 323 (162 CTD + 82 meta + 79 SST) | JSONL |
| Cross-source links | 496 temporal matches | Parquet |
| Embeddings | 323 × 768-dim | NumPy / pgvector |

### Pre-Analysis Outputs

| File | Content |
| --- | --- |
| `ctd_monthly_trends.parquet` | 27 monthly aggregates (temp, salinity, DO, Chl-a) per bay |
| `taxa_env_correlations.parquet` | 100 Spearman pairs (20 genera × 5 env variables), **21 significant** |
| `diversity_indices.parquet` | 164 samples (82 Kraken + 82 MetaEuk): Shannon, Simpson, Richness, Evenness |
| `bay_comparison.parquet` | Per-bay CTD aggregates |
| `taxa_cooccurrence.parquet` | 30×30 Jaccard similarity matrix (variable-prevalence genera) |
| `analysis_documents.jsonl` | 5 text summaries injected into RAG prompts |

### Reliability Ensurance Outputs

| File | Content |
| --- | --- |
| `sst_ctd_validation.parquet` | 24 paired SST↔CTD observations, **100% agreement** (mean ΔT=0.92°C) |
| `gap_interpolation.parquet` | 79 SST days with interpolated CTD-equivalent surface temperatures (mean confidence=0.916) |
| `diversity_prediction.parquet` | 37 samples with predicted vs actual Shannon diversity, **1 anomaly** detected (2024-07-O-s1, −2.3σ) |
| `corroboration.parquet` | 207 observations scored: **37 verified**, 20 supported, 150 standalone |
| `reliability_documents.jsonl` | 4 text summaries injected into RAG prompts |

### PostgreSQL Database (9 tables, 14,231 total rows)

| Table | Rows | Purpose |
| --- | --- | --- |
| `anchor_event` | 286 | Spatiotemporal linking across modalities |
| `ctd_profile` | 10,955 | Full depth-resolved CTD measurements |
| `ctd_summary` | 162 | Per-cast aggregated statistics |
| `metagenome_sample` | 82 | Per-sample sequencing + top taxa |
| `sst_point_observation` | 1,848 | Hourly satellite SST at monitoring point |
| `sst_daily_summary` | 79 | Daily regional SST aggregates |
| `retrieval_document` | 323 | Text chunks + 768-dim embeddings + tsvector |
| `cross_source_link` | 496 | Temporal links between CTD/meta ↔ SST |
| `provenance_record` | 0 | (tracked via JSONL, not yet loaded to PG) |

---

## Retrieval System

### How Hybrid Search Works

1. **User query** → embedded via `nomic-embed-text` (768-dim)
2. **Vector search**: pgvector cosine similarity over 323 document embeddings
3. **Full-text search**: PostgreSQL tsvector with `ts_rank_cd` scoring
4. **SQL filters**: bay, source_type, time range (applied to both)
5. **Reciprocal Rank Fusion (RRF)**: merges vector + FTS rankings

```
RRF_score(doc) = 1/(k + rank_vector) + 1/(k + rank_fts)   where k=60
```

### Pre-Analysis Context Injection

For complex ecosystem queries (detected via keywords: *correlation*, *diversity*, *seasonal*, *trend*, *ecosystem*, etc.), the system automatically injects precomputed analysis summaries as supplementary context alongside retrieved evidence. This gives the LLM:

- Correlation coefficients (e.g., "Gyrodinium × temperature: ρ=−0.60, p=0.0001")
- Diversity statistics (e.g., "Mean Shannon H'=3.884, richness=394 genera")
- Temporal trends (e.g., "Onagawa temperature: 7.5°C Jan → 23°C Aug")
- Bay comparisons

### Reliability Ensurance Context Injection

For queries involving data confidence, validation, or cross-source consistency (detected via keywords: *reliable*, *confidence*, *validate*, *anomaly*, *gap*, *temperature*, *SST*, *CTD*, etc.), reliability documents are injected alongside evidence. This gives the LLM:

- **SST↔CTD validation**: 24/24 paired observations agree within ±2.0°C (mean |ΔT|=0.92°C)
- **Gap interpolation**: 79 SST-derived surface temperature estimates during CTD sampling gaps (mean confidence=0.916)
- **Diversity prediction**: Environment-based diversity estimates with anomaly detection (1 anomaly: 2024-07-O-s1 at −2.3σ)
- **Corroboration tiers**: 37 verified (multi-source), 20 supported (partial), 150 standalone observations

### Provenance-Aware Prompting

Every LLM query includes:

- A system prompt enforcing citation rules (`[doc_id]`, `[analysis_*]`, and `[reliability_*]` notation)
- Retrieved evidence with source type, time, and provenance metadata
- Pre-computed analyses when the query warrants it
- Cross-source reliability data when the query warrants it
- Rules: only use provided evidence, state gaps, report units

---

## Reliability Ensurance System

The reliability ensurance layer runs **cross-source validation** to reinforce data confidence. It uses overlapping data between CTD, metagenome, and SST sources to predict, interpolate, and validate observations.

### Four Validation Engines

| Engine | Method | Result |
| --- | --- | --- |
| **SST ↔ CTD Validation** | Compare satellite SST with CTD surface temperature on matching dates | 24/24 agree (100%), mean ΔT=0.92°C |
| **Gap Interpolation** | Use continuous SST daily data to fill temporal gaps between CTD sampling dates | 79 days interpolated, confidence 0.916 |
| **Diversity Prediction** | Predict expected Shannon diversity from CTD environmental conditions using known taxa-environment correlations | 37 samples evaluated, 1 anomaly detected |
| **Corroboration Scoring** | Count independent sources confirming each observation and assign reliability tiers | 37 verified, 20 supported, 150 standalone |

### Reliability Tiers

- **Verified** — Multi-source agreement (e.g., SST confirms CTD surface T + diversity matches environment-based prediction)
- **Supported** — Partial corroboration (one cross-check passed)
- **Standalone** — Single source only, no cross-validation available

### Detected Anomaly

Sample `2024-07-O-s1` (Onagawa Bay, July 2024) was flagged with **−2.3σ deviation**: actual Shannon H'=1.601 vs predicted H'=3.453 given environmental conditions. This indicates a possible bloom event or dominance shift worth investigating.

---

## Key Ecological Findings

### Taxa–Environment Correlations (21/100 significant, p<0.05)

| Genus | Variable | Spearman ρ | Direction |
| --- | --- | --- | --- |
| Gyrodinium | temperature | −0.60 | Dinoflagellate declines with warming |
| Oncaea | temperature | +0.59 | Copepod increases with warming |
| Levanderina | salinity | −0.50 | Declines with salinity |
| Seminavis | temperature | −0.52 | Diatom declines with warming |
| Clausocalanus | temperature | −0.42 | Copepod declines with warming |
| Asterionellopsis | temperature | −0.43 | Diatom declines with warming |

### Community Diversity (Kraken, 82 samples)

- **Shannon H'**: mean=3.884, range [0.77, 5.10]
- **Simpson 1-D**: mean=0.908 (high evenness)
- **Richness**: mean=394 genera, range [52, 671]

### Co-occurrence Patterns

- Jaccard similarity range: 0.162 – 0.881
- Strongest: Pedosvirus × Anaposvirus (J=0.88)
- Weakest: Aggregatimonas × Anaposvirus (J=0.16)

---

## Application (Streamlit)

The app has **7 tabs**:

| Tab | Description |
| --- | --- |
| **Overview** | System pipeline architecture diagram (Graphviz) with reliability ensurance node and live metrics across all 9 pipeline stages. |
| **Chat** | Streaming LLM chat with provenance-aware RAG, source citations, expandable evidence, and automatic analysis + reliability context injection for complex queries. |
| **Evidence Explorer** | Search 323 documents by keyword + source type + bay, powered by pgvector hybrid search. |
| **Data** | 3 sub-tabs: CTD Profiles (interactive depth profiles for 162 casts), Taxa (Kraken/MetaEuk bar charts for 82 samples), SST (point time series + daily summary). |
| **Pre-Analysis** | 5 sub-tabs: CTD Trends (monthly plots), Correlations (Spearman heatmap), Diversity (Shannon/Simpson scatter), Co-occurrence (Jaccard heatmap), Reliability (SST↔CTD validation, gap interpolation, diversity prediction, corroboration summary). |
| **Database** | 4 sub-tabs: Table Browser, SQL Console (read-only), Schema inspector, Embedding statistics + Similarity Probe. |
| **Stats** | Corpus metrics, sample coverage matrix, bay distribution, provenance file count. |

### Sidebar Configuration Center

- **Model Settings**: Controls for `chat_model`, `temperature`, `top_p` (nucleus sampling), `repeat_penalty`, and `context_window`.
- **Retrieval Settings**: Granular control over hybrid search weights (`vector_weight`, `fts_weight`, `rrf_k` constant), `top_k_sources`, toggles for pre-analysis context injection and reliability context injection.
- **Filters**: Source type and Bay filters, plus date-range filtering (`time_from`, `time_to`).
- **Status**: Backend connection indicator (e.g., green = pgvector active, 323 embedded docs).

---

## How to Run

### Prerequisites

- Python 3.12 with packages: `streamlit pandas sqlalchemy psycopg2-binary pgvector xarray netcdf4 requests numpy matplotlib scipy`
- Podman (or Docker)
- Ollama

### Commands

```bash
# 1. Start infrastructure
podman machine start
podman compose up -d              # PostgreSQL + pgvector on port 5433
ollama pull nomic-embed-text      # embedding model (one-time)

# 2. Run data pipeline (if starting fresh)
python scripts/ingest.py                # ingestion + preprocessing
python scripts/build_retrieval_docs.py  # anchor events + docs + links
python scripts/run_pre_analysis.py      # ★ precompute ecological analyses
python scripts/run_reliability.py       # ★ cross-source reliability validation
python scripts/load_db.py --reset --embed  # populate DB + embed 323 docs

# 3. Launch application
streamlit run app.py
```

### Key Config ([config.py](file:///Users/jaronchai/Desktop/source_chat_agt/config.py))

| Setting | Value |
| --- | --- |
| `DATABASE_URL` | `postgresql://onagawa:onagawa@localhost:5433/onagawa_rag` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `EMBEDDING_MODEL` | `nomic-embed-text` (768-dim) |
| `CHAT_MODEL` | `qwen2.5:14b-instruct` |
| `ANALYSIS_DIR` | `data/analysis/` |
| `RELIABILITY_DIR` | `data/reliability/` |
| `SST_CTD_AGREEMENT_THRESHOLD` | `2.0` (°C, configurable via `SST_CTD_THRESHOLD` env var) |
| `DIVERSITY_ANOMALY_SIGMA` | `2.0` (configurable via `DIVERSITY_ANOMALY_SIGMA` env var) |

---

## Key Design Decisions

1. **Parquet as intermediate format** — columnar storage enables fast analytical queries during development; PostgreSQL for production serving
2. **Anchor events** — spatiotemporal linking layer that connects CTD, metagenome, and SST data from the same place/time
3. **Narrative text chunks** — each retrieval document is a self-contained paragraph with statistics, not raw CSV rows, making it LLM-friendly
4. **Dual retrieval backends** — auto-detects PostgreSQL; falls back to local BM25 + numpy for environments without a database
5. **Pre-analysis injection** — keyword-triggered: only injects analysis context for complex ecosystem queries (correlations, trends, diversity), keeping simple queries lightweight
6. **Reliability ensurance** — modular cross-source validation layer that uses SST↔CTD agreement, environment-based diversity prediction, and corroboration scoring to reinforce data confidence; results are injected into LLM prompts alongside analysis context
7. **Variable-prevalence co-occurrence** — selects genera present in 10–90% of samples, excluding ubiquitous ones that would trivially co-occur everywhere
8. **Read-only SQL console** — blocks destructive queries (DROP, DELETE, etc.) while allowing full analytical exploration
9. **Port 5433** — avoids conflict with any local PostgreSQL installation on the default 5432
10. **Modular pipeline stages** — each layer (preprocessing, pre-analysis, reliability, retrieval) is independently runnable via CLI scripts, enabling incremental updates without full pipeline re-runs
