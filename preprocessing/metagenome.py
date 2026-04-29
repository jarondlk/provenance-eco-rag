"""
Metagenome preprocessing pipeline.

Extracted from notebook cells 7, 9, 15-16, 18, 26-27, 29, 31-32, 34-35, 37
of 01_phase1_ingestion.ipynb.

Pipeline overview:
    load_run_mapping + load_read_summary → build_run_qc → build_sample_qc
    load_abundance_wide → wide_to_long
    load_group_mapping, load_gn/km_consistency
    enrich_abundance (join group + consistency)
    top_n_taxa_as_json
    build_sample_registry → build_sample_multisource_context
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .common import (
    add_sample_parsed_columns,
    derive_sample_dims,
    normalize_genus_key,
    read_tsv_no_header,
    read_tsv_with_header,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Run mapping and QC  (notebook cells 7, 9)
# =====================================================================
def load_run_mapping(path: Path) -> pd.DataFrame:
    """Load runid.tsv: run_id ↔ sample_replicate ↔ run_date."""
    runid = read_tsv_no_header(path, columns=["run_id", "sample_replicate", "run_date"])
    runid["run_date"] = pd.to_datetime(runid["run_date"], errors="coerce")
    runid = add_sample_parsed_columns(runid, "sample_replicate")
    logger.info("Loaded run mapping: %d rows", len(runid))
    return runid


def load_read_summary(path: Path) -> pd.DataFrame:
    """Load 01.read_summary_gt1kb.tsv: per-replicate QC stats."""
    read_summary = read_tsv_no_header(
        path,
        columns=[
            "sample_replicate",
            "n_reads_gt1kb",
            "bases_gt1kb",
            "n_reads_gt10kb",
            "bases_gt10kb",
        ],
    )
    for c in ["n_reads_gt1kb", "bases_gt1kb", "n_reads_gt10kb", "bases_gt10kb"]:
        read_summary[c] = pd.to_numeric(read_summary[c], errors="coerce")
    read_summary = add_sample_parsed_columns(read_summary, "sample_replicate")
    logger.info("Loaded read summary: %d rows", len(read_summary))
    return read_summary


def build_run_qc(runid: pd.DataFrame, read_summary: pd.DataFrame) -> pd.DataFrame:
    """Join run mapping with read summary for per-run QC."""
    drop_cols = [
        c
        for c in ["sample_year_month", "bay", "station_code", "replicate_no", "sample_id"]
        if c in read_summary.columns
    ]
    run_qc = runid.merge(
        read_summary.drop(columns=drop_cols),
        on="sample_replicate",
        how="left",
        validate="one_to_one",
    )
    logger.info("Run QC: %d rows", len(run_qc))
    return run_qc


def build_sample_qc(run_qc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level QC to sample-level summaries."""
    sample_qc = (
        run_qc.groupby("sample_id", dropna=False)
        .agg(
            n_runs=("run_id", "nunique"),
            first_run_date=("run_date", "min"),
            last_run_date=("run_date", "max"),
            sum_reads_gt1kb=("n_reads_gt1kb", "sum"),
            sum_bases_gt1kb=("bases_gt1kb", "sum"),
            sum_reads_gt10kb=("n_reads_gt10kb", "sum"),
            sum_bases_gt10kb=("bases_gt10kb", "sum"),
        )
        .reset_index()
    )
    logger.info("Sample QC: %d samples", len(sample_qc))
    return sample_qc


# =====================================================================
# 2. Abundance matrices  (notebook cells 15, 16)
# =====================================================================
def load_abundance_wide(path: Path) -> pd.DataFrame:
    """
    Load a genus × sample abundance matrix (TSV).

    These files have a header row with N sample IDs but each data row
    has N+1 fields: the genus name (unlabelled) followed by N abundance
    values.  We read with ``index_col=0`` so pandas treats the first
    field of each data row as the index, then reset it as the 'genus'
    column.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = "genus"
    df = df.reset_index()
    df["genus"] = df["genus"].astype("string").str.strip()
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    logger.info("Loaded abundance matrix: %d genera × %d samples from %s",
                df.shape[0], df.shape[1] - 1, path.name)
    return df


def wide_to_long(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Melt a wide genus × sample matrix to long format."""
    long_df = df.melt(
        id_vars=["genus"],
        var_name="sample_id",
        value_name="abundance_value",
    )
    long_df["method"] = method
    long_df["abundance_value"] = pd.to_numeric(long_df["abundance_value"], errors="coerce")
    return long_df


# =====================================================================
# 3. Group mappings and consistency  (notebook cells 26, 27, 29)
# =====================================================================
def load_group_mapping(path: Path) -> pd.DataFrame:
    """Load genus → upper_group mapping (Kraken or global)."""
    df = pd.read_csv(path, sep="\t", header=None)

    if df.shape[1] == 4:
        df.columns = ["genus_taxid", "genus", "upper_taxid", "upper_group_label"]
    elif df.shape[1] == 3:
        df.columns = ["genus", "upper_taxid", "upper_group_label"]
        df["genus_taxid"] = pd.NA
    elif df.shape[1] == 2:
        df.columns = ["genus", "upper_group_label"]
        df["genus_taxid"] = pd.NA
        df["upper_taxid"] = pd.NA
    else:
        raise ValueError(f"{path.name}: unsupported number of columns: {df.shape[1]}")

    df["genus"] = df["genus"].astype("string").str.strip()
    df["upper_group_label"] = df["upper_group_label"].astype("string").str.strip()
    df["upper_domain"] = df["upper_group_label"].str.split(":", n=1).str[0]
    df["upper_group_name"] = (
        df["upper_group_label"].str.split(":", n=1).str[1].fillna(df["upper_group_label"])
    )
    return df[["genus_taxid", "genus", "upper_taxid", "upper_group_label", "upper_domain", "upper_group_name"]]


def load_upper_group_abundance(path: Path) -> pd.DataFrame:
    """
    Load Kraken upper-group × sample abundance and melt to long.

    File format: header row has (empty)\t sample_id_1 \t sample_id_2 …
    Data rows: upper_group \t upper_group_label \t val_1 \t val_2 …
    So the first data column becomes the index (upper_group), the
    first header-named column is the upper_group_label, and the rest
    are sample abundance values.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = "upper_group"
    df = df.reset_index()
    df["upper_group"] = df["upper_group"].astype("string").str.strip()

    # Second column is upper_group_label
    label_col = df.columns[1]
    df = df.rename(columns={label_col: "upper_group_label"})
    df["upper_group_label"] = df["upper_group_label"].astype("string").str.strip()

    sample_ids = list(df.columns[2:])
    for c in sample_ids:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["upper_domain"] = df["upper_group_label"].str.split(":", n=1).str[0]
    df["upper_group_name"] = (
        df["upper_group_label"].str.split(":", n=1).str[1].fillna(
            df["upper_group_label"]
        )
    )

    long = df.melt(
        id_vars=["upper_group", "upper_group_label", "upper_domain", "upper_group_name"],
        var_name="sample_id",
        value_name="abundance_value",
    )
    logger.info("Loaded upper-group abundance: %d groups × %d samples from %s",
                df.shape[0], len(sample_ids), path.name)
    return long


def load_gn_consistency(path: Path) -> pd.DataFrame:
    """Load genus-level consistency table."""
    df = pd.read_csv(path, sep="\t", header=None)

    if df.shape[1] == 3:
        df.columns = ["genus_taxid", "genus", "consistency_level"]
    elif df.shape[1] == 2:
        df.columns = ["genus", "consistency_level"]
        df["genus_taxid"] = pd.NA
    else:
        raise ValueError(f"{path.name}: unsupported number of columns: {df.shape[1]}")

    df["genus"] = df["genus"].astype("string").str.strip()
    df["consistency_level"] = pd.to_numeric(df["consistency_level"], errors="coerce")
    return df[["genus_taxid", "genus", "consistency_level"]]


def load_km_consistency(path: Path) -> pd.DataFrame:
    """Load contig-level (km) consistency table."""
    df = pd.read_csv(path, sep="\t", header=None, dtype="string")

    if df.shape[1] == 7:
        df.columns = [
            "contig_id",
            "method_code",
            "consistency_level",
            "consistent_taxid",
            "consistent_taxname",
            "kraken_taxonomy",
            "metaeuk_taxonomy",
        ]
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        df = df.rename(
            columns={"col_0": "contig_id", "col_1": "method_code", "col_2": "consistency_level"}
        )

    df["consistency_level"] = pd.to_numeric(df["consistency_level"], errors="coerce")
    return df


# =====================================================================
# 4. Enrich abundance  (notebook cells 31, 32)
# =====================================================================
def enrich_abundance(
    genus_long: pd.DataFrame,
    group_map: pd.DataFrame,
    gn_consistency: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join genus-level abundance with group mapping and consistency levels.

    Both inputs are deduped on genus before joining.
    """
    genus_long = genus_long.copy()
    genus_long["genus"] = normalize_genus_key(genus_long["genus"])

    group_dedup = (
        group_map[["genus", "upper_group_label", "upper_domain", "upper_group_name"]]
        .dropna(subset=["genus"])
        .drop_duplicates(subset=["genus"])
    )
    group_dedup["genus"] = normalize_genus_key(group_dedup["genus"])

    consistency_dedup = (
        gn_consistency[["genus", "consistency_level"]]
        .dropna(subset=["genus"])
        .drop_duplicates(subset=["genus"])
    )
    consistency_dedup["genus"] = normalize_genus_key(consistency_dedup["genus"])

    enriched = genus_long.merge(group_dedup, on="genus", how="left").merge(
        consistency_dedup, on="genus", how="left"
    )
    return enriched


# =====================================================================
# 5. Top-N taxa per sample  (notebook cell 35)
# =====================================================================
def top_n_taxa_as_json(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    For each sample_id, collect the top-N taxa by abundance as a JSON string.

    Returns a DataFrame: sample_id, top_{group_col}_{n}_json
    """
    out_rows = []

    for sample_id, g in df.groupby("sample_id", dropna=False):
        g2 = (
            g.dropna(subset=[group_col, value_col])
            .sort_values(value_col, ascending=False)
            .head(n)
        )

        records = []
        for _, row in g2.iterrows():
            item = {
                group_col: row[group_col],
                "abundance_value": float(row[value_col]) if pd.notna(row[value_col]) else None,
            }
            if "upper_group_label" in row.index:
                item["upper_group_label"] = (
                    row["upper_group_label"] if pd.notna(row["upper_group_label"]) else None
                )
            if "consistency_level" in row.index:
                item["consistency_level"] = (
                    int(row["consistency_level"]) if pd.notna(row["consistency_level"]) else None
                )
            records.append(item)

        out_rows.append(
            {
                "sample_id": sample_id,
                f"top_{group_col}_{n}_json": json.dumps(records, ensure_ascii=False),
            }
        )

    return pd.DataFrame(out_rows)


# =====================================================================
# 6. Sample registry & multi-source context  (notebook cells 34, 37)
# =====================================================================
def build_sample_registry(
    sample_qc: pd.DataFrame,
    ctd_summary: pd.DataFrame,
    kraken_genus_long: pd.DataFrame,
    metaeuk_genus_long: pd.DataFrame,
    kraken_upper_group_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a unified sample registry with coverage flags.
    """
    all_sample_ids = sorted(
        set(sample_qc["sample_id"].dropna())
        | set(ctd_summary["sample_id"].dropna())
        | set(kraken_genus_long["sample_id"].dropna())
        | set(metaeuk_genus_long["sample_id"].dropna())
        | set(kraken_upper_group_long["sample_id"].dropna())
    )

    registry = derive_sample_dims(pd.Series(all_sample_ids))

    registry = registry.merge(sample_qc, on="sample_id", how="left")
    registry = registry.merge(
        ctd_summary[["sample_id", "ctd_date", "n_depth_points", "min_depth_m", "max_depth_m"]],
        on="sample_id",
        how="left",
    )

    registry["has_run_qc"] = registry["n_runs"].notna()
    registry["has_kraken"] = registry["sample_id"].isin(
        kraken_genus_long["sample_id"].unique()
    )
    registry["has_metaeuk"] = registry["sample_id"].isin(
        metaeuk_genus_long["sample_id"].unique()
    )
    registry["has_ctd"] = registry["sample_id"].isin(
        ctd_summary["sample_id"].unique()
    )
    registry["has_kraken_upper_group"] = registry["sample_id"].isin(
        kraken_upper_group_long["sample_id"].unique()
    )

    registry["ctd_year_month"] = registry["ctd_date"].dt.strftime("%Y-%m")
    registry["run_first_year_month"] = pd.to_datetime(
        registry["first_run_date"], errors="coerce"
    ).dt.strftime("%Y-%m")
    registry["run_last_year_month"] = pd.to_datetime(
        registry["last_run_date"], errors="coerce"
    ).dt.strftime("%Y-%m")

    registry["ctd_month_match"] = np.where(
        registry["ctd_year_month"].isna(),
        pd.NA,
        registry["sample_year_month"] == registry["ctd_year_month"],
    )
    registry["run_first_month_match"] = np.where(
        registry["run_first_year_month"].isna(),
        pd.NA,
        registry["sample_year_month"] == registry["run_first_year_month"],
    )
    registry["run_last_month_match"] = np.where(
        registry["run_last_year_month"].isna(),
        pd.NA,
        registry["sample_year_month"] == registry["run_last_year_month"],
    )

    logger.info("Sample registry: %d samples", len(registry))
    return registry


def build_sample_multisource_context(
    sample_registry: pd.DataFrame,
    ctd_summary: pd.DataFrame,
    kraken_top_genera: pd.DataFrame,
    metaeuk_top_genera: pd.DataFrame,
    kraken_top_upper_groups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join the sample registry with CTD summary and top-taxa JSON columns
    to produce the serving-layer multisource context table.
    """
    ctx = (
        sample_registry.merge(ctd_summary, on="sample_id", how="left", suffixes=("", "_ctd"))
        .merge(kraken_top_genera, on="sample_id", how="left")
        .merge(metaeuk_top_genera, on="sample_id", how="left")
        .merge(kraken_top_upper_groups, on="sample_id", how="left")
    )
    logger.info("Multisource context: %d rows × %d cols", *ctx.shape)
    return ctx
