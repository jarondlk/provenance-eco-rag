#!/usr/bin/env python3
"""
scripts/build_retrieval_docs.py – Build retrieval documents from normalized data.

Reads the parquet files produced by ingest.py and creates:
  - anchor_events.parquet   (canonical anchors)
  - retrieval_documents.parquet + retrieval_documents.jsonl
  - cross_source_links.parquet

Usage:
    python scripts/build_retrieval_docs.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import config
from schema.anchor_event import build_anchor_events
from retrieval.document_builder import (
    build_all_documents,
    documents_to_dataframe,
    documents_to_jsonl,
)
from retrieval.cross_source_linker import build_cross_source_links

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_docs")


def main() -> None:
    config.ensure_dirs()

    logger.info("=" * 60)
    logger.info("Building retrieval documents")
    logger.info("=" * 60)

    # Load normalized data
    logger.info("Loading normalized data...")

    ctd_summary = pd.read_parquet(config.NORMALIZED_DIR / "ctd_summary.parquet")
    logger.info("  CTD summary: %d rows", len(ctd_summary))

    sample_context = pd.read_parquet(config.SERVING_DIR / "sample_multisource_context.parquet")
    logger.info("  Sample context: %d rows", len(sample_context))

    sample_registry = pd.read_parquet(config.SERVING_DIR / "sample_registry.parquet")
    logger.info("  Sample registry: %d rows", len(sample_registry))

    # SST (optional)
    sst_daily = None
    sst_point = None
    sst_daily_path = config.NORMALIZED_DIR / "sst_daily_summary.parquet"
    sst_point_path = config.NORMALIZED_DIR / "sst_point_timeseries.parquet"

    if sst_daily_path.exists():
        sst_daily = pd.read_parquet(sst_daily_path)
        logger.info("  SST daily: %d rows", len(sst_daily))
    else:
        logger.warning("  SST daily summary not found – skipping SST docs")

    if sst_point_path.exists():
        sst_point = pd.read_parquet(sst_point_path)
        logger.info("  SST point: %d rows", len(sst_point))

    # Build anchor events
    logger.info("Building anchor events...")
    anchors = build_anchor_events(sample_registry, ctd_summary, sst_daily)
    anchors.to_parquet(config.CANONICAL_DIR / "anchor_events.parquet", index=False)
    logger.info("  Saved: anchor_events.parquet (%d events)", len(anchors))

    # Build retrieval documents
    logger.info("Building retrieval documents...")
    docs = build_all_documents(ctd_summary, sample_context, sst_daily, sst_point)

    docs_df = documents_to_dataframe(docs)
    docs_df.to_parquet(config.SERVING_DIR / "retrieval_documents.parquet", index=False)
    logger.info("  Saved: retrieval_documents.parquet (%d docs)", len(docs_df))

    # Also write JSONL for backwards compatibility with the existing RAG pipeline
    documents_to_jsonl(docs, config.SERVING_DIR / "retrieval_documents.jsonl")

    # Build cross-source links
    logger.info("Building cross-source links...")
    links = build_cross_source_links(anchors)
    links.to_parquet(config.CANONICAL_DIR / "cross_source_links.parquet", index=False)
    logger.info("  Saved: cross_source_links.parquet (%d links)", len(links))

    # Summary
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("  Anchor events:         %d", len(anchors))
    logger.info("  Retrieval documents:   %d", len(docs_df))
    logger.info("    CTD:                 %d", len(docs_df[docs_df["source_type"] == "ctd"]))
    logger.info("    Metagenome:          %d", len(docs_df[docs_df["source_type"] == "metagenome"]))
    logger.info("    Remote sensing:      %d", len(docs_df[docs_df["source_type"] == "remote_sensing"]))
    logger.info("  Cross-source links:    %d", len(links))


if __name__ == "__main__":
    main()
