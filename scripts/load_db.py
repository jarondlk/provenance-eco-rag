#!/usr/bin/env python3
"""
scripts/load_db.py – Load normalized data into PostgreSQL + pgvector.

1. Creates all tables (init_db)
2. Loads anchor events, CTD, metagenome, SST, and retrieval documents
3. Populates tsvector column for FTS
4. Optionally embeds documents via Ollama

Usage:
    python scripts/load_db.py                # load all data
    python scripts/load_db.py --embed        # also compute embeddings
    python scripts/load_db.py --reset        # drop and recreate tables
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import text

import config
from db.connection import init_db, drop_all, get_session, get_engine
from db.models import (
    AnchorEvent,
    CrossSourceLink,
    CtdProfile,
    CtdSummary,
    MetagenomeSample,
    ProvenanceRecord,
    RetrievalDocument,
    SstDailySummary,
    SstPointObservation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("load_db")


def load_parquet_to_table(
    parquet_path: Path,
    table_name: str,
    column_map: dict | None = None,
    if_exists: str = "append",
) -> int:
    """Load a parquet file into a DB table using pandas + SQLAlchemy."""
    if not parquet_path.exists():
        logger.warning("  Skipping %s – file not found: %s", table_name, parquet_path)
        return 0

    df = pd.read_parquet(parquet_path)
    if column_map:
        df = df.rename(columns=column_map)

    engine = get_engine()

    # Only keep columns that exist in the target table
    from sqlalchemy import inspect as sa_inspect
    inspector = sa_inspect(engine)
    db_cols = {c["name"] for c in inspector.get_columns(table_name)}
    keep_cols = [c for c in df.columns if c in db_cols]
    df = df[keep_cols]

    df.to_sql(table_name, engine, if_exists=if_exists, index=False, method="multi")
    logger.info("  Loaded %d rows → %s", len(df), table_name)
    return len(df)


def load_retrieval_documents() -> int:
    """Load retrieval documents from parquet."""
    path = config.SERVING_DIR / "retrieval_documents.parquet"
    if not path.exists():
        logger.warning("  retrieval_documents.parquet not found")
        return 0

    df = pd.read_parquet(path)
    engine = get_engine()
    df.to_sql("retrieval_document", engine, if_exists="append", index=False, method="multi")
    logger.info("  Loaded %d retrieval documents", len(df))
    return len(df)


def update_fts_vectors() -> None:
    """Populate the text_tsv column for full-text search."""
    with get_session() as session:
        session.execute(text("""
            UPDATE retrieval_document
            SET text_tsv = to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))
            WHERE text_tsv IS NULL
        """))
    logger.info("  Updated FTS vectors")


def load_cross_source_links() -> int:
    """Load cross-source links from parquet."""
    path = config.CANONICAL_DIR / "cross_source_links.parquet"
    return load_parquet_to_table(path, "cross_source_link")


def load_anchor_events() -> int:
    """Load anchor events from parquet."""
    path = config.CANONICAL_DIR / "anchor_events.parquet"
    return load_parquet_to_table(path, "anchor_event")


def load_metagenome_samples() -> int:
    """Load metagenome sample records from the multisource context."""
    path = config.SERVING_DIR / "sample_multisource_context.parquet"
    if not path.exists():
        return 0

    df = pd.read_parquet(path)

    # Map columns to DB model
    records = []
    for _, row in df.iterrows():
        sid = row.get("sample_id")
        if pd.isna(sid):
            continue
        has_kr = row.get("has_kraken", False)
        has_me = row.get("has_metaeuk", False)
        if not has_kr and not has_me:
            continue

        records.append({
            "sample_id": sid,
            "bay": row.get("bay"),
            "station_code": row.get("station_code"),
            "sample_year_month": row.get("sample_year_month"),
            "n_runs": int(row["n_runs"]) if pd.notna(row.get("n_runs")) else None,
            "first_run_date": row.get("first_run_date"),
            "last_run_date": row.get("last_run_date"),
            "sum_reads_gt1kb": row.get("sum_reads_gt1kb"),
            "sum_bases_gt1kb": row.get("sum_bases_gt1kb"),
            "has_kraken": bool(has_kr),
            "has_metaeuk": bool(has_me),
            "has_ctd": bool(row.get("has_ctd", False)),
            "top_kraken_genera_json": row.get("top_genus_10_json_x"),
            "top_metaeuk_genera_json": row.get("top_genus_10_json_y"),
            "top_upper_groups_json": row.get("top_upper_group_10_json"),
        })

    if records:
        engine = get_engine()
        pd.DataFrame(records).to_sql(
            "metagenome_sample", engine, if_exists="append", index=False, method="multi"
        )
    logger.info("  Loaded %d metagenome samples", len(records))
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load data into PostgreSQL")
    parser.add_argument("--embed", action="store_true", help="Compute embeddings via Ollama")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Loading data into PostgreSQL")
    logger.info("=" * 60)

    if args.reset:
        logger.warning("Dropping all tables...")
        drop_all()

    init_db()

    # Load data
    load_anchor_events()

    load_parquet_to_table(
        config.NORMALIZED_DIR / "ctd_profile_standardized.parquet",
        "ctd_profile",
    )
    load_parquet_to_table(
        config.NORMALIZED_DIR / "ctd_summary.parquet",
        "ctd_summary",
    )

    load_metagenome_samples()

    load_parquet_to_table(
        config.NORMALIZED_DIR / "sst_point_timeseries.parquet",
        "sst_point_observation",
    )
    load_parquet_to_table(
        config.NORMALIZED_DIR / "sst_daily_summary.parquet",
        "sst_daily_summary",
    )

    load_retrieval_documents()
    update_fts_vectors()

    load_cross_source_links()

    if args.embed:
        logger.info("Computing embeddings...")
        from db.vector_store import update_document_embeddings
        n = update_document_embeddings()
        logger.info("Embedded %d documents", n)

    logger.info("=" * 60)
    logger.info("Database load complete!")


if __name__ == "__main__":
    main()
