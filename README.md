# provenance-eco-rag

**Onagawa Source Chat** — a provenance-aware Retrieval-Augmented Generation (RAG) system for marine environmental monitoring in Miyagi Prefecture, Japan.

Transforms fragmented field data — CTD water profiles, metagenome sequencing, and satellite SST — into a citation-grounded question-answering system where every answer traces back to its original source.

---

## Study Sites

| Bay | Code | Coordinates | Data |
| --- | --- | --- | --- |
| Onagawa Bay | O | ~38.44°N, 141.45°E | CTD + Metagenome + SST |
| Ishinomaki Bay | I | ~38.41°N, 141.30°E | CTD + Metagenome |
| Matsushima Bay | M | ~38.35°N, 141.06°E | CTD + Metagenome |

---

## Architecture

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        CTD["CTD\n1 TSV, 10,955 profiles"]
        META["Metagenome\n11 TSV files"]
        SST["Satellite SST\n1,848 NetCDF"]
    end

    PROV["Provenance Registry\nSHA-256, 1,849 records"]

    subgraph Preprocess["Preprocessing"]
        CTD_PP["CTD Pipeline\nstandardize, summaries"]
        META_PP["Metagenome Pipeline\nKraken, MetaEuk, groups"]
        SST_PP["SST Pipeline\npoint extraction, daily agg"]
    end

    NORM["Normalized Parquets\n16 files"]
    ANCHOR["Anchor Events\n286 anchors, 496 links"]

    subgraph Analysis["Ecological Analysis"]
        PREANA["Pre-Analysis\n5 ecological analyses\n5 RAG documents"]
        RELIAB["Reliability Ensurance\n4 validation outputs\n4 RAG documents"]
    end

    RETDOCS["Retrieval Documents\n323 narrative chunks\n162 CTD + 82 meta + 79 SST"]

    subgraph Storage["PostgreSQL + pgvector"]
        EMB["Vector Embeddings\n323 x 768-dim\nnomic-embed-text"]
        FTS["Full-Text Index\ntsvector + ts_rank_cd"]
        DB["9 Relational Tables\nprofiles, samples, links"]
    end

    RET["Hybrid Retrieval\nVector + FTS + RRF\n+ Analysis + Reliability injection"]
    LLM["LLM\nProvenance-aware prompting\nCitation-grounded answers"]

    Sources --> PROV --> Preprocess --> NORM
    NORM --> ANCHOR
    NORM --> Analysis
    PREANA -.->|correlations| RELIAB
    ANCHOR --> RETDOCS
    NORM --> RETDOCS
    RETDOCS --> EMB & FTS & DB
    PREANA --> DB
    EMB -->|cosine| RET
    FTS -->|rank| RET
    PREANA -.->|inject| RET
    RELIAB -.->|inject| RET
    RET -->|top-K + context| LLM
```

---

## Technology Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.12 (31 files, ~7,000 lines) |
| Database | PostgreSQL 16 + pgvector (cosine similarity) |
| Container | Podman / Docker |
| LLM | Ollama (local) — qwen2.5:14b-instruct |
| Embeddings | nomic-embed-text (768-dim) |
| Data | Pandas, Parquet, xarray, netCDF4, SciPy |
| ORM | SQLAlchemy 2.x |
| UI | Streamlit |
| Search | pgvector cosine + tsvector FTS + Reciprocal Rank Fusion |

---

## Quick Start

### Prerequisites

- Python 3.12
- Podman or Docker
- Ollama

### Setup

```bash
# Install dependencies
pip install streamlit pandas sqlalchemy psycopg2-binary pgvector \
    xarray netcdf4 requests numpy matplotlib scipy

# Start database
podman machine start
podman compose up -d              # PostgreSQL + pgvector on port 5433

# Pull models
ollama pull nomic-embed-text
ollama pull qwen2.5:14b-instruct
```

### Data Pipeline

```bash
python scripts/ingest.py                # 1. Ingestion + preprocessing
python scripts/build_retrieval_docs.py  # 2. Anchor events + documents + links
python scripts/run_pre_analysis.py      # 3. Ecological analyses
python scripts/run_reliability.py       # 4. Cross-source reliability validation
python scripts/load_db.py --reset --embed  # 5. Populate DB + embed 323 docs
```

### Launch

```bash
streamlit run app.py
```

---

## Application

The Streamlit interface has **7 tabs**:

| Tab | Description |
| --- | --- |
| **Overview** | Pipeline architecture diagram (Graphviz) with live metrics across all stages |
| **Chat** | Streaming LLM chat with provenance-aware RAG, source citations, and automatic context injection. Shows all sources (retrieved + analysis + reliability) feeding the LLM. |
| **Evidence Explorer** | Search 323 documents by keyword, source type, and bay |
| **Data** | CTD depth profiles, metagenome composition (Kraken/MetaEuk), SST time series |
| **Pre-Analysis** | 5 sub-tabs: CTD Trends, Correlations, Diversity, Co-occurrence, Reliability |
| **Database** | Table browser, SQL console (read-only), schema inspector, embedding statistics |
| **Stats** | Corpus metrics, sample coverage, provenance tracking |

### Sidebar

- **Model**: chat model, temperature, top_p, repeat_penalty, context_window
- **Retrieval**: vector/FTS weights, RRF-k, top-K, pre-analysis toggle, reliability toggle
- **Filters**: source type, bay, date range
- **Status**: backend connection indicator

### Screenshots

![Overview Tab](docs/screenshots/overview_tab.png)
*System Overview with pipeline architecture and live metrics.*

![Data (CTD) Tab](docs/screenshots/data_ctd_tab.png)
*Interactive depth profiles for CTD measurements.*

![Pre-Analysis Tab](docs/screenshots/pre_analysis_tab.png)
*Ecological correlations and diversity indices.*

![Database Explorer Tab](docs/screenshots/database_tab.png)
*Read-only SQL console and table inspector.*

![Stats Tab](docs/screenshots/stats_tab.png)
*Corpus statistics and data coverage.*

---

## Project Structure

```
source_chat_agt/
├── app.py                              # Streamlit application (7 tabs, ~1,820 lines)
├── config.py                           # Paths, DB, models, thresholds
├── docker-compose.yml                  # PostgreSQL + pgvector container
│
├── preprocessing/
│   ├── common.py                       # Sample ID parsing, TSV I/O
│   ├── ctd.py                          # CTD load → standardize → summaries
│   ├── metagenome.py                   # Kraken/MetaEuk abundance, QC, groups
│   ├── remote_sensing.py               # NetCDF SST extraction
│   ├── pre_analysis.py                 # Ecological pre-analysis (5 analyses)
│   └── reliability_ensurance.py        # Cross-source validation (4 engines)
│
├── ingestion/
│   ├── provenance.py                   # SHA-256 file registration (JSONL)
│   └── file_inventory.py              # Directory scanner
│
├── schema/
│   └── anchor_event.py                 # Spatiotemporal linking
│
├── retrieval/
│   ├── document_builder.py             # Raw data → narrative text chunks
│   ├── cross_source_linker.py          # same_sample + time_match links
│   ├── hybrid_retriever.py             # pgvector + FTS + RRF (primary)
│   └── local_retriever.py              # BM25 + numpy fallback
│
├── db/
│   ├── models.py                       # 9 SQLAlchemy ORM tables
│   ├── connection.py                   # Engine, sessions, init_db
│   └── vector_store.py                 # Ollama embedding + cosine search
│
├── orchestration/
│   ├── query_orchestrator.py           # Cross-source evidence expansion
│   └── unified.py                      # Prompt builder + context injection
│
├── scripts/
│   ├── ingest.py                       # Ingestion pipeline
│   ├── build_retrieval_docs.py         # Documents + links
│   ├── load_db.py                      # Populate PostgreSQL + embeddings
│   ├── run_pre_analysis.py             # Pre-analysis pipeline
│   └── run_reliability.py              # Reliability pipeline
│
└── data/
    ├── raw/ctd/                        # 1 file (CTD_Onagawa.tsv)
    ├── raw/meta/                       # 11 files (Kraken, MetaEuk, QC)
    ├── normalized/                     # 16 parquet files
    ├── canonical/                      # anchor_events, cross_source_links
    ├── serving/                        # retrieval docs, embeddings, registry
    ├── analysis/                       # 6 pre-analysis outputs
    ├── reliability/                    # 5 reliability outputs
    └── provenance/                     # provenance.jsonl
```

---

## Data

### Raw Input

| Source | Files | Size | Period |
| --- | --- | --- | --- |
| CTD (Onagawa) | 1 TSV | 1.2 MB | Jan 2024 – Mar 2026 |
| Metagenome | 11 TSV/TXT | 34 MB | Apr 2024 – Feb 2026 |
| Satellite SST | 1,848 NetCDF | ~3.7 GB | Dec 2025 – Feb 2026 |

### Processed Output

| Dataset | Records |
| --- | --- |
| CTD profiles (standardized) | 10,955 depth points |
| CTD cast summaries | 162 casts |
| Kraken genus abundance | 58,712 (716 genera × 82 samples) |
| MetaEuk genus abundance | 67,240 (820 genera × 82 samples) |
| SST hourly observations | 1,848 points |
| SST daily summaries | 79 days |
| Anchor events | 286 (207 sample + 79 SST) |
| Retrieval documents | 323 (162 CTD + 82 meta + 79 SST) |
| Cross-source links | 496 temporal matches |
| Embeddings | 323 × 768-dim |

### Pre-Analysis

| Output | Content |
| --- | --- |
| CTD monthly trends | 27 monthly aggregates per bay |
| Taxa-env correlations | 100 Spearman pairs, **21 significant** (p<0.05) |
| Diversity indices | 164 samples: Shannon, Simpson, Richness, Evenness |
| Bay comparison | Per-bay CTD aggregates |
| Co-occurrence | 30×30 Jaccard similarity matrix |
| Analysis documents | 5 text summaries for RAG injection |

### Reliability Ensurance

| Output | Result |
| --- | --- |
| SST ↔ CTD validation | 24 paired obs, **100% agreement**, mean ΔT = 0.92°C |
| Gap interpolation | 79 SST days, interpolated surface T, confidence 0.916 |
| Diversity prediction | 37 samples, **1 anomaly** (2024-07-O-s1, −2.3σ) |
| Corroboration scoring | **37 verified**, 20 supported, 150 standalone |
| Reliability documents | 4 text summaries for RAG injection |

### PostgreSQL Database (9 tables)

| Table | Rows | Purpose |
| --- | --- | --- |
| `anchor_event` | 286 | Spatiotemporal linking |
| `ctd_profile` | 10,955 | Depth-resolved measurements |
| `ctd_summary` | 162 | Per-cast statistics |
| `metagenome_sample` | 82 | Sequencing + top taxa |
| `sst_point_observation` | 1,848 | Hourly satellite SST |
| `sst_daily_summary` | 79 | Daily regional SST |
| `retrieval_document` | 323 | Text + embeddings + tsvector |
| `cross_source_link` | 496 | CTD/meta ↔ SST links |
| `provenance_record` | 0 | (tracked via JSONL) |

---

## Retrieval System

### Hybrid Search

1. **Query** → embedded via nomic-embed-text (768-dim)
2. **Vector search** — pgvector cosine similarity over 323 embeddings
3. **Full-text search** — PostgreSQL tsvector with ts_rank_cd
4. **SQL filters** — bay, source_type, time range
5. **RRF fusion** — merges vector + FTS rankings: `score = w_v/(k+r_v) + w_f/(k+r_f)` where k=60

### Context Injection

| Context | Trigger keywords | Citations |
| --- | --- | --- |
| **Pre-Analysis** | correlation, diversity, trend, seasonal, ecosystem, ... | `[analysis_*]` |
| **Reliability** | reliable, confidence, validate, anomaly, gap, temperature, SST, CTD, ... | `[reliability_*]` |

Both are toggleable via sidebar checkboxes.

### Provenance-Aware Prompting

Every prompt includes:
- System rules enforcing `[doc_id]`, `[analysis_*]`, and `[reliability_*]` citations
- Retrieved evidence with source type, time, and provenance metadata
- Pre-analysis context (when keyword-triggered)
- Reliability context (when keyword-triggered)

---

## Reliability Ensurance

Cross-source validation layer that uses overlapping data to reinforce system confidence.

| Engine | Method | Result |
| --- | --- | --- |
| **SST ↔ CTD** | Compare satellite SST with CTD surface T on matching dates | 24/24 agree, mean ΔT = 0.92°C |
| **Gap Interpolation** | Continuous SST fills temporal gaps between CTD dates | 79 days, confidence 0.916 |
| **Diversity Prediction** | Predict Shannon H' from CTD conditions via known correlations | 1 anomaly: 2024-07-O-s1 (−2.3σ) |
| **Corroboration** | Multi-source agreement scoring per observation | 37 verified / 207 total |

**Reliability tiers**: verified (multi-source) → supported (partial) → standalone (single source)

---

## Key Ecological Findings

### Taxa–Environment Correlations (21/100 significant, p<0.05)

| Genus | Variable | ρ | Direction |
| --- | --- | --- | --- |
| Gyrodinium | temperature | −0.60 | Dinoflagellate declines with warming |
| Oncaea | temperature | +0.59 | Copepod increases with warming |
| Levanderina | salinity | −0.50 | Declines with salinity |
| Seminavis | temperature | −0.52 | Diatom declines with warming |

### Community Diversity (Kraken, 82 samples)

- **Shannon H'**: mean = 3.884, range [0.77, 5.10]
- **Simpson 1-D**: mean = 0.908
- **Richness**: mean = 394 genera, range [52, 671]

### Detected Anomaly

Sample **2024-07-O-s1** (Onagawa Bay, July 2024): Shannon H' = 1.601 vs predicted 3.453 (−2.3σ). Indicates possible bloom event or dominance shift.

---

## Configuration

Key settings in [config.py](config.py):

| Setting | Default |
| --- | --- |
| `DATABASE_URL` | `postgresql://onagawa:onagawa@localhost:5433/onagawa_rag` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `EMBEDDING_MODEL` | `nomic-embed-text` (768-dim) |
| `CHAT_MODEL` | `qwen2.5:14b-instruct` |
| `SST_CTD_AGREEMENT_THRESHOLD` | 2.0°C (env: `SST_CTD_THRESHOLD`) |
| `DIVERSITY_ANOMALY_SIGMA` | 2.0 (env: `DIVERSITY_ANOMALY_SIGMA`) |

---

## Design Decisions

1. **Parquet as intermediate format** — columnar storage for fast analytical queries; PostgreSQL for production serving
2. **Anchor events** — spatiotemporal linking layer connecting CTD, metagenome, and SST from the same place/time
3. **Narrative text chunks** — each document is a self-contained paragraph with statistics, not raw CSV rows
4. **Dual retrieval backends** — auto-detects PostgreSQL; falls back to local BM25 + numpy without a database
5. **Pre-analysis injection** — keyword-triggered: only injects for complex ecosystem queries
6. **Reliability ensurance** — modular cross-source validation with SST↔CTD agreement, diversity prediction, and corroboration scoring
7. **Variable-prevalence co-occurrence** — selects genera in 10–90% of samples to avoid trivial co-occurrence
8. **Read-only SQL console** — blocks destructive queries while allowing analytical exploration
9. **Port 5433** — avoids conflict with default PostgreSQL on 5432
10. **Modular pipeline** — each stage is independently runnable via CLI scripts
