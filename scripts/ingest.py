#!/usr/bin/env python3
"""
scripts/ingest.py – End-to-end ingestion pipeline.

Runs: raw → preprocessing → normalized parquet output.

Usage:
    python scripts/ingest.py                   # full pipeline
    python scripts/ingest.py --validate-only   # dry-run: check raw files exist
    python scripts/ingest.py --skip-sst        # skip SST (needs xarray/netcdf4)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from ingestion.provenance import ProvenanceRegistry
from ingestion.file_inventory import inventory_dir, inventory_recursive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def validate_raw_files() -> bool:
    """Check that all expected raw files exist."""
    ok = True
    for key, path in config.RAW_FILES.items():
        if path.exists():
            logger.info("  ✓  %-30s  %s", key, path.name)
        else:
            logger.error("  ✗  %-30s  MISSING: %s", key, path)
            ok = False

    # SST files
    sst_files = list(config.SST_NETCDF_DIR.rglob("*.nc")) if config.SST_NETCDF_DIR.exists() else []
    logger.info("  SST NetCDF files: %d under %s", len(sst_files), config.SST_NETCDF_DIR)

    return ok


def run_provenance_registration(registry: ProvenanceRegistry, run_id: str) -> None:
    """Register all raw input files in the provenance registry."""
    logger.info("--- Provenance registration ---")

    for key, path in config.RAW_FILES.items():
        if path.exists():
            rec = registry.register(path, source_dataset=key, processing_run=run_id)
            logger.info("  Registered: %s  sha256=%s…", path.name, rec.sha256[:12])

    # SST NetCDFs
    if config.SST_NETCDF_DIR.exists():
        for fp in sorted(config.SST_NETCDF_DIR.rglob("*.nc")):
            rec = registry.register(fp, source_dataset="sst_netcdf", processing_run=run_id)
        logger.info("  Registered %d SST files", len(list(config.SST_NETCDF_DIR.rglob("*.nc"))))


def run_ctd_pipeline() -> None:
    """Run the CTD preprocessing pipeline."""
    from preprocessing.ctd import load_ctd_raw, standardize_ctd_columns, summarize_ctd_profiles

    logger.info("--- CTD pipeline ---")
    ctd_raw = load_ctd_raw(config.RAW_FILES["ctd"])
    ctd_std = standardize_ctd_columns(ctd_raw)
    ctd_summary = summarize_ctd_profiles(ctd_std)

    # Save outputs
    ctd_raw.to_parquet(config.NORMALIZED_DIR / "ctd_profile.parquet", index=False)
    ctd_std.to_parquet(config.NORMALIZED_DIR / "ctd_profile_standardized.parquet", index=False)
    ctd_summary.to_parquet(config.NORMALIZED_DIR / "ctd_summary.parquet", index=False)

    logger.info("  Saved: ctd_profile.parquet (%d rows)", len(ctd_raw))
    logger.info("  Saved: ctd_profile_standardized.parquet (%d rows)", len(ctd_std))
    logger.info("  Saved: ctd_summary.parquet (%d rows)", len(ctd_summary))


def run_metagenome_pipeline() -> None:
    """Run the metagenome preprocessing pipeline."""
    from preprocessing.metagenome import (
        load_run_mapping,
        load_read_summary,
        build_run_qc,
        build_sample_qc,
        load_abundance_wide,
        wide_to_long,
        load_group_mapping,
        load_upper_group_abundance,
        load_gn_consistency,
        load_km_consistency,
        enrich_abundance,
        top_n_taxa_as_json,
        build_sample_registry,
        build_sample_multisource_context,
    )
    from preprocessing.ctd import load_ctd_raw, standardize_ctd_columns, summarize_ctd_profiles

    logger.info("--- Metagenome pipeline ---")

    # 1. Run QC
    runid = load_run_mapping(config.RAW_FILES["runid"])
    read_summary = load_read_summary(config.RAW_FILES["read_summary"])
    run_qc = build_run_qc(runid, read_summary)
    sample_qc = build_sample_qc(run_qc)

    run_qc.to_parquet(config.NORMALIZED_DIR / "run_qc.parquet", index=False)
    sample_qc.to_parquet(config.NORMALIZED_DIR / "sample_qc.parquet", index=False)
    logger.info("  Saved: run_qc.parquet, sample_qc.parquet")

    # 2. Abundance matrices
    kraken_wide = load_abundance_wide(config.RAW_FILES["kraken_genus_sample_tsv"])
    metaeuk_wide = load_abundance_wide(config.RAW_FILES["metaeuk_genus_sample"])

    kraken_long = wide_to_long(kraken_wide, method="kraken")
    metaeuk_long = wide_to_long(metaeuk_wide, method="metaeuk")

    kraken_long.to_parquet(config.NORMALIZED_DIR / "kraken_genus_abundance.parquet", index=False)
    metaeuk_long.to_parquet(config.NORMALIZED_DIR / "metaeuk_genus_abundance.parquet", index=False)
    logger.info("  Saved: kraken_genus_abundance.parquet, metaeuk_genus_abundance.parquet")

    # 3. Group mappings
    kraken_group_map = load_group_mapping(config.RAW_FILES["kraken_genus_group"])
    global_group_map = load_group_mapping(config.RAW_FILES["genus_group"])

    kraken_group_map.to_parquet(config.NORMALIZED_DIR / "kraken_group_map.parquet", index=False)
    global_group_map.to_parquet(config.NORMALIZED_DIR / "global_group_map.parquet", index=False)
    logger.info("  Saved: kraken_group_map.parquet, global_group_map.parquet")

    # 4. Upper-group abundance
    kraken_upper_long = load_upper_group_abundance(
        config.RAW_FILES["kraken_upper_group_sample"]
    )
    kraken_upper_long.to_parquet(
        config.NORMALIZED_DIR / "kraken_upper_group_abundance.parquet", index=False
    )
    logger.info("  Saved: kraken_upper_group_abundance.parquet")

    # 5. Consistency tables
    gn_con = load_gn_consistency(config.RAW_FILES["gn_consistency"])
    km_con = load_km_consistency(config.RAW_FILES["km_consistency"])

    gn_con.to_parquet(config.NORMALIZED_DIR / "genus_consistency.parquet", index=False)
    km_con.to_parquet(config.NORMALIZED_DIR / "contig_consistency.parquet", index=False)
    logger.info("  Saved: genus_consistency.parquet, contig_consistency.parquet")

    # 6. Enrich abundance with group + consistency
    kraken_enriched = enrich_abundance(kraken_long, kraken_group_map, gn_con)
    metaeuk_enriched = enrich_abundance(metaeuk_long, global_group_map, gn_con)

    kraken_enriched.to_parquet(config.NORMALIZED_DIR / "kraken_genus_enriched.parquet", index=False)
    metaeuk_enriched.to_parquet(config.NORMALIZED_DIR / "metaeuk_genus_enriched.parquet", index=False)
    logger.info("  Saved: kraken_genus_enriched.parquet, metaeuk_genus_enriched.parquet")

    # 7. CTD summary (reload for registry building)
    ctd_summary = None
    ctd_summary_path = config.NORMALIZED_DIR / "ctd_summary.parquet"
    if ctd_summary_path.exists():
        import pandas as pd
        ctd_summary = pd.read_parquet(ctd_summary_path)
    else:
        ctd_raw = load_ctd_raw(config.RAW_FILES["ctd"])
        ctd_std = standardize_ctd_columns(ctd_raw)
        ctd_summary = summarize_ctd_profiles(ctd_std)

    # 8. Top-N taxa
    kraken_top = top_n_taxa_as_json(kraken_enriched, "genus", "abundance_value", n=10)
    metaeuk_top = top_n_taxa_as_json(metaeuk_enriched, "genus", "abundance_value", n=10)
    kraken_upper_top = top_n_taxa_as_json(kraken_upper_long, "upper_group", "abundance_value", n=10)
    logger.info("  Computed top-N taxa JSON columns")

    # 9. Sample registry
    sample_reg = build_sample_registry(
        sample_qc, ctd_summary, kraken_long, metaeuk_long, kraken_upper_long
    )
    sample_reg.to_parquet(config.SERVING_DIR / "sample_registry.parquet", index=False)
    logger.info("  Saved: sample_registry.parquet (%d samples)", len(sample_reg))

    # 10. Multisource context
    ctx = build_sample_multisource_context(
        sample_reg, ctd_summary, kraken_top, metaeuk_top, kraken_upper_top
    )
    ctx.to_parquet(config.SERVING_DIR / "sample_multisource_context.parquet", index=False)
    logger.info("  Saved: sample_multisource_context.parquet (%d rows)", len(ctx))


def run_sst_pipeline() -> None:
    """Run the satellite SST preprocessing pipeline."""
    from preprocessing.remote_sensing import extract_point_timeseries, compute_daily_summary

    logger.info("--- SST pipeline ---")

    if not config.SST_NETCDF_DIR.exists():
        logger.warning("SST directory not found: %s – skipping", config.SST_NETCDF_DIR)
        return

    # Point time series
    ts = extract_point_timeseries(
        config.SST_NETCDF_DIR, config.ONAGAWA_LAT, config.ONAGAWA_LON
    )
    ts.to_parquet(config.NORMALIZED_DIR / "sst_point_timeseries.parquet", index=False)
    logger.info("  Saved: sst_point_timeseries.parquet (%d records)", len(ts))

    # Daily regional summary
    daily = compute_daily_summary(
        config.SST_NETCDF_DIR,
        lat_min=config.SST_LAT_MIN,
        lat_max=config.SST_LAT_MAX,
        lon_min=config.SST_LON_MIN,
        lon_max=config.SST_LON_MAX,
    )
    daily.to_parquet(config.NORMALIZED_DIR / "sst_daily_summary.parquet", index=False)
    logger.info("  Saved: sst_daily_summary.parquet (%d days)", len(daily))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument("--validate-only", action="store_true", help="Check raw files only")
    parser.add_argument("--skip-sst", action="store_true", help="Skip SST pipeline")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Onagawa Source Chat – Ingestion Pipeline")
    logger.info("=" * 60)

    # Ensure output dirs
    config.ensure_dirs()

    # Validate
    logger.info("Validating raw files...")
    if not validate_raw_files():
        logger.error("Some raw files are missing. Aborting.")
        sys.exit(1)

    if args.validate_only:
        logger.info("Validation passed. Exiting (--validate-only).")
        return

    # Provenance
    run_id = f"ingest_{int(time.time())}"
    registry = ProvenanceRegistry(config.PROVENANCE_DIR / "provenance.jsonl")
    run_provenance_registration(registry, run_id)
    logger.info("Provenance registry: %d records", len(registry))

    # Run pipelines
    run_ctd_pipeline()
    run_metagenome_pipeline()

    if not args.skip_sst:
        run_sst_pipeline()
    else:
        logger.info("--- SST pipeline SKIPPED (--skip-sst) ---")

    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info("Normalized outputs:")
    for p in sorted(config.NORMALIZED_DIR.glob("*.parquet")):
        logger.info("  %s  (%.1f KB)", p.name, p.stat().st_size / 1024)
    logger.info("Serving outputs:")
    for p in sorted(config.SERVING_DIR.glob("*.parquet")):
        logger.info("  %s  (%.1f KB)", p.name, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()
