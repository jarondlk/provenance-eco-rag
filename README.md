# provenance-monitoring-data-integration

Onagawa Source Chat is a provenance-aware Retrieval-Augmented Generation (RAG) application for marine environmental monitoring in Miyagi Prefecture, Japan. It transforms fragmented field data (CTD water profiles, metagenome sequencing results, and satellite sea surface temperature observations) into a searchable, citation-grounded question-answering system.

Every LLM-generated answer links back to its original data sources with traceable provenance. For complex ecosystem questions, the system also draws on precomputed ecological analyses (correlations, diversity indices, temporal trends) that go beyond what any single retrieved document can provide.

## Study Sites

* **Onagawa Bay** (~38.44 N, 141.45 E): CTD, Metagenome, SST
* **Ishinomaki Bay** (~38.41 N, 141.30 E): CTD, Metagenome
* **Matsushima Bay** (~38.35 N, 141.06 E): CTD, Metagenome

## Architecture

The system follows an 8-layer pipeline:
1. **Ingestion**: Provenance registry via SHA-256 hashes.
2. **Preprocessing**: Standardization of CTD profiles, metagenome abundance (Kraken/MetaEuk), and SST netCDFs.
3. **Canonical Schema**: Anchor events for spatiotemporal linking.
4. **Pre-Analysis**: Precomputed ecological relationships (trends, correlations, diversity, co-occurrence).
5. **Retrieval Documents**: Generation of narrative text chunks from raw data.
6. **Database**: PostgreSQL 16 + pgvector storing embeddings and metadata.
7. **Retrieval**: Hybrid search (vector + Full-Text Search + Reciprocal Rank Fusion) and analysis context injection.
8. **Application**: Streamlit interface with chat, data exploration, and analytics.

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

# 4. Populate database and generate embeddings
python scripts/load_db.py --reset --embed
```

## Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The application provides a multi-tab interface:
* **Chat**: Streaming LLM chat with provenance-aware RAG and source citations.
* **Evidence Explorer**: Search documents by keyword, source type, and bay.
* **CTD Profiles**: Interactive depth profiles and summary metrics.
* **Taxa**: Metagenome composition charts.
* **SST**: Satellite temperature time series.
* **Pre-Analysis**: Ecological trends, correlations, diversity, and co-occurrence matrices.
* **Database**: Read-only SQL console and schema inspector.
* **Stats**: Corpus metrics and provenance tracking.

## Retrieval System Details

The hybrid retrieval system merges vector search (cosine similarity via pgvector) and full-text search (tsvector via PostgreSQL) using Reciprocal Rank Fusion (RRF). 

For complex queries containing keywords like "correlation", "diversity", or "trend", the query orchestrator automatically injects precomputed analysis summaries as supplementary context alongside retrieved evidence, allowing the LLM to provide statistically grounded answers.
