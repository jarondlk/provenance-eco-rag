"""
Reliability Ensurance layer – cross-source validation and corroboration.

Uses overlapping data between CTD, metagenome, and SST sources to
predict, interpolate, and validate observations. Produces reliability
scores and corroboration documents injected into RAG prompts.

Outputs are saved to data/reliability/ as parquet + JSONL text summaries.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. SST ↔ CTD Surface Temperature Validation
# ─────────────────────────────────────────────
def validate_sst_ctd_surface_temp() -> pd.DataFrame:
    """
    Compare satellite SST with CTD surface temperature on matching dates.

    For each CTD cast date, finds the corresponding satellite SST daily
    observation and computes agreement metrics. Satellite SST measures a
    thin skin layer (~10µm) while CTD measures at ~0.5m depth, so small
    discrepancies are expected.

    Returns DataFrame with columns:
        sample_id, ctd_date, bay, ctd_surface_t, sst_daily_mean,
        delta_t, abs_delta_t, agrees, reliability_score
    """
    ctd_path = config.NORMALIZED_DIR / "ctd_summary.parquet"
    sst_path = config.NORMALIZED_DIR / "sst_daily_summary.parquet"

    if not ctd_path.exists() or not sst_path.exists():
        logger.warning("Missing CTD or SST data for SST-CTD validation")
        return pd.DataFrame()

    ctd = pd.read_parquet(ctd_path)
    sst = pd.read_parquet(sst_path)

    if ctd.empty or sst.empty:
        return pd.DataFrame()

    # Ensure date columns
    ctd["ctd_date"] = pd.to_datetime(ctd["ctd_date"], errors="coerce")
    ctd["date_str"] = ctd["ctd_date"].dt.strftime("%Y-%m-%d")
    sst["date_str"] = sst["date_jst"].astype(str)

    # Parse bay from sample_id
    ctd["bay"] = ctd["sample_id"].str.extract(r"\d{4}-\d{2}-([A-Z])-")

    # Filter to CTD rows with valid surface temperature
    ctd_valid = ctd[ctd["surface_temperature"].notna()].copy()

    if ctd_valid.empty:
        logger.warning("No CTD surface temperature data available")
        return pd.DataFrame()

    # Merge on date
    merged = ctd_valid.merge(
        sst[["date_str", "mean_sst", "min_sst", "max_sst"]],
        on="date_str",
        how="inner",
    )

    if merged.empty:
        logger.warning("No date overlap between CTD and SST observations")
        return pd.DataFrame()

    threshold = config.SST_CTD_AGREEMENT_THRESHOLD

    # Compute agreement metrics
    merged["delta_t"] = merged["surface_temperature"] - merged["mean_sst"]
    merged["abs_delta_t"] = merged["delta_t"].abs()
    merged["agrees"] = merged["abs_delta_t"] <= threshold

    # Reliability score: 1.0 when delta=0, decays toward 0 at threshold
    merged["reliability_score"] = np.clip(
        1.0 - (merged["abs_delta_t"] / (threshold * 2)), 0.0, 1.0
    ).round(4)

    result = merged[[
        "sample_id", "ctd_date", "bay", "surface_temperature",
        "mean_sst", "delta_t", "abs_delta_t", "agrees", "reliability_score",
    ]].rename(columns={
        "surface_temperature": "ctd_surface_t",
        "mean_sst": "sst_daily_mean",
    }).copy()

    n_agree = result["agrees"].sum()
    logger.info(
        "SST-CTD validation: %d paired observations, %d agree (%.0f%%), "
        "mean |ΔT|=%.2f°C",
        len(result), n_agree,
        n_agree / len(result) * 100 if len(result) > 0 else 0,
        result["abs_delta_t"].mean(),
    )
    return result


# ─────────────────────────────────────────────
# 2. Temporal Gap Interpolation via SST
# ─────────────────────────────────────────────
def interpolate_sst_for_gaps() -> pd.DataFrame:
    """
    Use continuous SST daily coverage to interpolate expected surface
    temperatures during gaps between CTD sampling dates.

    First establishes the SST↔CTD relationship from paired observations,
    then applies it to SST dates without CTD coverage to produce
    predicted CTD-equivalent surface temperatures.

    Returns DataFrame with columns:
        date, sst_daily_mean, interpolated_surface_t, confidence,
        method, nearest_ctd_days, in_ctd_gap
    """
    ctd_path = config.NORMALIZED_DIR / "ctd_summary.parquet"
    sst_path = config.NORMALIZED_DIR / "sst_daily_summary.parquet"

    if not ctd_path.exists() or not sst_path.exists():
        logger.warning("Missing data for gap interpolation")
        return pd.DataFrame()

    ctd = pd.read_parquet(ctd_path)
    sst = pd.read_parquet(sst_path)

    if ctd.empty or sst.empty:
        return pd.DataFrame()

    ctd["ctd_date"] = pd.to_datetime(ctd["ctd_date"], errors="coerce")
    ctd_dates = ctd["ctd_date"].dropna().dt.normalize().unique()
    ctd_dates = pd.DatetimeIndex(ctd_dates).sort_values()

    sst["date"] = pd.to_datetime(sst["date_jst"], errors="coerce")
    sst = sst.dropna(subset=["date", "mean_sst"]).copy()

    if len(ctd_dates) < 2:
        logger.warning("Need at least 2 CTD dates for gap interpolation")
        return pd.DataFrame()

    # Compute SST→CTD offset from paired observations
    ctd_for_offset = ctd[ctd["surface_temperature"].notna()].copy()
    ctd_for_offset["date_key"] = ctd_for_offset["ctd_date"].dt.strftime("%Y-%m-%d")
    sst["date_key"] = sst["date"].dt.strftime("%Y-%m-%d")

    paired = ctd_for_offset.merge(
        sst[["date_key", "mean_sst"]], on="date_key", how="inner"
    )

    if paired.empty:
        # No overlap — use SST directly without offset correction
        offset = 0.0
        offset_std = 1.0
        logger.info("No SST-CTD overlap for offset; using raw SST values")
    else:
        offsets = paired["surface_temperature"] - paired["mean_sst"]
        offset = float(offsets.mean())
        offset_std = float(offsets.std()) if len(offsets) > 1 else 1.0
        logger.info(
            "SST→CTD offset: %.2f ± %.2f°C (from %d paired obs)",
            offset, offset_std, len(paired),
        )

    # For each SST date, compute interpolated value and confidence
    rows = []
    for _, sr in sst.iterrows():
        d = sr["date"]
        sst_val = sr["mean_sst"]

        # Distance to nearest CTD date
        deltas = np.abs((ctd_dates - d).total_seconds()) / 86400
        nearest_days = float(deltas.min()) if len(deltas) > 0 else 999

        # Check if this date is in a CTD gap (between two CTD dates)
        in_gap = bool((d > ctd_dates.min()) and (d < ctd_dates.max()))

        # Interpolated surface T = SST + learned offset
        interp_t = sst_val + offset

        # Confidence decays with distance from nearest CTD observation
        # Full confidence near CTD dates, lower in long gaps
        confidence = float(np.clip(
            1.0 - (nearest_days / 90.0),  # 90 days → confidence 0
            0.1, 1.0,
        ))

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "sst_daily_mean": round(sst_val, 3),
            "interpolated_surface_t": round(interp_t, 3),
            "confidence": round(confidence, 4),
            "method": "sst_offset_correction",
            "nearest_ctd_days": round(nearest_days, 1),
            "in_ctd_gap": in_gap,
        })

    result = pd.DataFrame(rows)
    n_gaps = result["in_ctd_gap"].sum()
    logger.info(
        "Gap interpolation: %d SST days processed, %d in CTD gaps, "
        "offset=%.2f°C",
        len(result), n_gaps, offset,
    )
    return result


# ─────────────────────────────────────────────
# 3. Diversity Prediction from Environment
# ─────────────────────────────────────────────
def predict_diversity_from_env() -> pd.DataFrame:
    """
    Use taxa-environment correlations and CTD conditions to predict
    expected Shannon diversity, then compare with actual values.

    Uses a simple linear model: for each significantly correlated
    genus-environment pair, predicts genus contribution direction.
    The ensemble of these predictions yields an expected diversity range.

    Returns DataFrame with columns:
        sample_id, bay, predicted_shannon, actual_shannon,
        deviation, deviation_sigma, is_anomaly,
        n_supporting_vars, supporting_env_summary
    """
    div_path = config.ANALYSIS_DIR / "diversity_indices.parquet"
    corr_path = config.ANALYSIS_DIR / "taxa_env_correlations.parquet"
    ctx_path = config.SERVING_DIR / "sample_multisource_context.parquet"

    if not all(p.exists() for p in [div_path, corr_path, ctx_path]):
        logger.warning("Missing data for diversity prediction")
        return pd.DataFrame()

    diversity = pd.read_parquet(div_path)
    correlations = pd.read_parquet(corr_path)
    ctx = pd.read_parquet(ctx_path)

    # Use Kraken diversity only
    div_kr = diversity[diversity["source"] == "kraken"].copy()
    if div_kr.empty:
        logger.warning("No Kraken diversity data for prediction")
        return pd.DataFrame()

    # Get significant correlations
    sig_corr = correlations[correlations["significant"]].copy()
    if sig_corr.empty:
        logger.warning("No significant correlations for diversity prediction")
        return pd.DataFrame()

    # Get CTD-linked samples
    paired = ctx[(ctx["has_ctd"]) & (ctx["has_kraken"])].copy()
    if paired.empty:
        return pd.DataFrame()

    # Environmental variables used in correlations
    env_vars = sig_corr["env_variable"].unique().tolist()
    available_env = [v for v in env_vars if v in paired.columns]

    if not available_env:
        return pd.DataFrame()

    # Merge diversity with environmental data
    div_env = div_kr.merge(
        paired[["sample_id"] + available_env],
        on="sample_id",
        how="inner",
    )

    if len(div_env) < 3:
        logger.warning("Too few paired samples (%d) for prediction", len(div_env))
        return pd.DataFrame()

    # Simple prediction: use mean Shannon as baseline, adjust by
    # how many env variables deviate from their means in a direction
    # consistent with positive/negative genus correlations
    mean_shannon = div_env["shannon_h"].mean()
    std_shannon = div_env["shannon_h"].std()
    if std_shannon == 0:
        std_shannon = 0.1

    # Compute z-scores for environmental variables
    env_means = {v: div_env[v].mean() for v in available_env if div_env[v].notna().sum() > 0}
    env_stds = {v: div_env[v].std() for v in available_env if div_env[v].notna().sum() > 0}

    results = []
    anomaly_sigma = config.DIVERSITY_ANOMALY_SIGMA

    for _, row in div_env.iterrows():
        sid = row["sample_id"]
        actual = row["shannon_h"]

        # Count how many env vars suggest higher/lower diversity
        env_signals = []
        supporting_parts = []

        for ev in available_env:
            val = row.get(ev)
            if pd.isna(val) or ev not in env_means or env_stds.get(ev, 0) == 0:
                continue

            z = (val - env_means[ev]) / env_stds[ev]

            # Direction from correlations: positive rho + high env → high diversity
            ev_corrs = sig_corr[sig_corr["env_variable"] == ev]
            mean_rho = ev_corrs["spearman_rho"].mean()

            signal = z * mean_rho  # positive = expect higher diversity
            env_signals.append(signal)
            direction = "↑" if signal > 0 else "↓"
            supporting_parts.append(
                f"{ev.replace('mean_', '')}={val:.1f}(z={z:+.1f}{direction})"
            )

        if not env_signals:
            continue

        # Predicted diversity: baseline ± adjustment
        adjustment = np.mean(env_signals) * std_shannon * 0.5
        predicted = mean_shannon + adjustment

        deviation = actual - predicted
        deviation_sigma = deviation / std_shannon if std_shannon > 0 else 0

        results.append({
            "sample_id": sid,
            "bay": row.get("bay", None),
            "predicted_shannon": round(predicted, 4),
            "actual_shannon": round(actual, 4),
            "deviation": round(deviation, 4),
            "deviation_sigma": round(deviation_sigma, 4),
            "is_anomaly": abs(deviation_sigma) > anomaly_sigma,
            "n_supporting_vars": len(env_signals),
            "supporting_env_summary": "; ".join(supporting_parts),
        })

    result = pd.DataFrame(results)
    n_anomalies = result["is_anomaly"].sum() if not result.empty else 0
    logger.info(
        "Diversity prediction: %d samples, %d anomalies (>%.1fσ)",
        len(result), n_anomalies, anomaly_sigma,
    )
    return result


# ─────────────────────────────────────────────
# 4. Cross-Source Corroboration Summary
# ─────────────────────────────────────────────
def corroborate_cross_source(
    sst_ctd_validation: pd.DataFrame,
    diversity_prediction: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign reliability tiers to observations based on how many
    independent sources corroborate them.

    Tiers:
        verified   — multi-source agreement (SST+CTD match, diversity predicted correctly)
        supported  — partial corroboration (one cross-check passed)
        standalone — single source only, no cross-validation available

    Returns DataFrame with columns:
        event_id, sample_id, source_type, reliability_tier,
        corroboration_sources, reliability_score, detail
    """
    ctx_path = config.SERVING_DIR / "sample_multisource_context.parquet"
    if not ctx_path.exists():
        return pd.DataFrame()

    ctx = pd.read_parquet(ctx_path)
    if ctx.empty:
        return pd.DataFrame()

    # Build SST validation lookup
    sst_valid_map: Dict[str, dict] = {}
    if not sst_ctd_validation.empty:
        for _, r in sst_ctd_validation.iterrows():
            sst_valid_map[r["sample_id"]] = {
                "agrees": bool(r["agrees"]),
                "delta_t": float(r["delta_t"]),
                "score": float(r["reliability_score"]),
            }

    # Build diversity prediction lookup
    div_pred_map: Dict[str, dict] = {}
    if not diversity_prediction.empty:
        for _, r in diversity_prediction.iterrows():
            div_pred_map[r["sample_id"]] = {
                "is_anomaly": bool(r["is_anomaly"]),
                "deviation_sigma": float(r["deviation_sigma"]),
            }

    rows = []
    for _, sr in ctx.iterrows():
        sid = sr.get("sample_id")
        if pd.isna(sid):
            continue

        has_ctd = bool(sr.get("has_ctd", False))
        has_kraken = bool(sr.get("has_kraken", False))
        has_metaeuk = bool(sr.get("has_metaeuk", False))

        # Count available source types
        source_types = []
        if has_ctd:
            source_types.append("ctd")
        if has_kraken or has_metaeuk:
            source_types.append("metagenome")

        # Check corroborations
        corroboration_sources = []
        scores = []
        details = []

        # SST ↔ CTD check
        if sid in sst_valid_map:
            sv = sst_valid_map[sid]
            corroboration_sources.append("sst_validation")
            scores.append(sv["score"])
            if sv["agrees"]:
                details.append(f"SST agrees (ΔT={sv['delta_t']:+.1f}°C)")
            else:
                details.append(f"SST disagrees (ΔT={sv['delta_t']:+.1f}°C)")

        # Diversity prediction check
        if sid in div_pred_map:
            dp = div_pred_map[sid]
            corroboration_sources.append("diversity_prediction")
            div_score = max(0.0, 1.0 - abs(dp["deviation_sigma"]) / 4.0)
            scores.append(div_score)
            if dp["is_anomaly"]:
                details.append(f"Diversity anomaly ({dp['deviation_sigma']:+.1f}σ)")
            else:
                details.append(f"Diversity matches prediction ({dp['deviation_sigma']:+.1f}σ)")

        # Multi-source bonus: having both CTD + metagenome is itself corroboration
        if has_ctd and (has_kraken or has_metaeuk):
            corroboration_sources.append("multi_modal")
            scores.append(0.8)
            details.append("Multi-modal sample (CTD + metagenome)")

        # Determine tier
        n_checks = len(corroboration_sources)
        if n_checks >= 2:
            tier = "verified"
        elif n_checks == 1:
            tier = "supported"
        else:
            tier = "standalone"

        avg_score = float(np.mean(scores)) if scores else 0.3

        rows.append({
            "event_id": f"sample_{sid}",
            "sample_id": sid,
            "source_type": ",".join(source_types),
            "reliability_tier": tier,
            "corroboration_sources": ",".join(corroboration_sources),
            "reliability_score": round(avg_score, 4),
            "n_checks": n_checks,
            "detail": " | ".join(details) if details else "No cross-validation",
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        tier_counts = result["reliability_tier"].value_counts().to_dict()
        logger.info(
            "Corroboration: %d observations — verified=%d, supported=%d, standalone=%d",
            len(result),
            tier_counts.get("verified", 0),
            tier_counts.get("supported", 0),
            tier_counts.get("standalone", 0),
        )
    return result


# ─────────────────────────────────────────────
# 5. Build Reliability Text Documents (for RAG)
# ─────────────────────────────────────────────
def build_reliability_documents(
    sst_ctd_validation: pd.DataFrame,
    gap_interpolation: pd.DataFrame,
    diversity_prediction: pd.DataFrame,
    corroboration: pd.DataFrame,
) -> List[dict]:
    """
    Convert reliability outputs into text documents for RAG injection.
    Same pattern as pre_analysis.build_analysis_documents().
    """
    docs = []
    bay_names = {"O": "Onagawa Bay", "I": "Ishinomaki Bay", "M": "Matsushima Bay"}

    # ── SST-CTD validation summary ──
    if not sst_ctd_validation.empty:
        n_total = len(sst_ctd_validation)
        n_agree = int(sst_ctd_validation["agrees"].sum())
        mean_delta = sst_ctd_validation["abs_delta_t"].mean()
        mean_score = sst_ctd_validation["reliability_score"].mean()

        text = "Cross-source validation: Satellite SST vs CTD surface temperature.\n"
        text += f"  Paired observations: {n_total}\n"
        text += f"  Agreement (within ±{config.SST_CTD_AGREEMENT_THRESHOLD}°C): "
        text += f"{n_agree}/{n_total} ({n_agree/n_total*100:.0f}%)\n"
        text += f"  Mean |ΔT|: {mean_delta:.2f}°C\n"
        text += f"  Mean reliability score: {mean_score:.3f}\n"

        # Per-bay breakdown
        for bay in sorted(sst_ctd_validation["bay"].dropna().unique()):
            bd = sst_ctd_validation[sst_ctd_validation["bay"] == bay]
            bn = bay_names.get(bay, bay)
            ba = int(bd["agrees"].sum())
            text += f"  {bn}: {ba}/{len(bd)} agree, mean |ΔT|={bd['abs_delta_t'].mean():.2f}°C\n"

        # Individual disagreements
        disagree = sst_ctd_validation[~sst_ctd_validation["agrees"]]
        if not disagree.empty:
            text += "  Notable disagreements:\n"
            for _, r in disagree.iterrows():
                text += (
                    f"    {r['sample_id']}: CTD={r['ctd_surface_t']:.1f}°C vs "
                    f"SST={r['sst_daily_mean']:.1f}°C (Δ={r['delta_t']:+.1f}°C)\n"
                )

        docs.append({
            "id": "reliability_sst_ctd_validation",
            "source_type": "reliability",
            "title": "SST ↔ CTD surface temperature cross-validation",
            "text": text,
            "analysis_type": "cross_source_validation",
        })

    # ── Gap interpolation summary ──
    if not gap_interpolation.empty:
        gaps = gap_interpolation[gap_interpolation["in_ctd_gap"]]
        text = "Temporal gap filling: SST-based surface temperature interpolation.\n"
        text += f"  Total SST days available: {len(gap_interpolation)}\n"
        text += f"  Days in CTD gaps: {len(gaps)}\n"

        if not gaps.empty:
            text += f"  Interpolated T range: {gaps['interpolated_surface_t'].min():.1f}"
            text += f"–{gaps['interpolated_surface_t'].max():.1f}°C\n"
            text += f"  Mean confidence: {gaps['confidence'].mean():.3f}\n"
            text += f"  High confidence (>0.7): {(gaps['confidence'] > 0.7).sum()} days\n"

        docs.append({
            "id": "reliability_gap_interpolation",
            "source_type": "reliability",
            "title": "Temporal gap interpolation via satellite SST",
            "text": text,
            "analysis_type": "gap_interpolation",
        })

    # ── Diversity prediction summary ──
    if not diversity_prediction.empty:
        n_total = len(diversity_prediction)
        n_anomaly = int(diversity_prediction["is_anomaly"].sum())
        mean_dev = diversity_prediction["deviation_sigma"].abs().mean()

        text = "Diversity prediction: CTD environment → expected Shannon diversity.\n"
        text += f"  Samples evaluated: {n_total}\n"
        text += f"  Anomalies (>{config.DIVERSITY_ANOMALY_SIGMA}σ deviation): {n_anomaly}\n"
        text += f"  Mean |deviation|: {mean_dev:.2f}σ\n"

        # Anomaly details
        anomalies = diversity_prediction[diversity_prediction["is_anomaly"]]
        if not anomalies.empty:
            text += "  Anomalous samples:\n"
            for _, r in anomalies.iterrows():
                bn = bay_names.get(r.get("bay"), r.get("bay", ""))
                text += (
                    f"    {r['sample_id']} ({bn}): predicted H'={r['predicted_shannon']:.3f}, "
                    f"actual={r['actual_shannon']:.3f} ({r['deviation_sigma']:+.1f}σ)\n"
                )

        docs.append({
            "id": "reliability_diversity_prediction",
            "source_type": "reliability",
            "title": "Environment-based diversity prediction and anomalies",
            "text": text,
            "analysis_type": "diversity_prediction",
        })

    # ── Corroboration summary ──
    if not corroboration.empty:
        tier_counts = corroboration["reliability_tier"].value_counts().to_dict()
        mean_score = corroboration["reliability_score"].mean()

        text = "Cross-source corroboration summary.\n"
        text += f"  Total observations: {len(corroboration)}\n"
        text += f"  Verified (multi-source agreement): {tier_counts.get('verified', 0)}\n"
        text += f"  Supported (partial corroboration): {tier_counts.get('supported', 0)}\n"
        text += f"  Standalone (single source): {tier_counts.get('standalone', 0)}\n"
        text += f"  Mean reliability score: {mean_score:.3f}\n"

        # List verified observations
        verified = corroboration[corroboration["reliability_tier"] == "verified"]
        if not verified.empty:
            text += "  Verified observations:\n"
            for _, r in verified.head(10).iterrows():
                text += f"    {r['sample_id']}: {r['detail']}\n"

        docs.append({
            "id": "reliability_corroboration_summary",
            "source_type": "reliability",
            "title": "Cross-source corroboration summary",
            "text": text,
            "analysis_type": "corroboration",
        })

    logger.info("Built %d reliability documents", len(docs))
    return docs


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
def run_all() -> Dict[str, Any]:
    """Run all reliability checks and save outputs."""
    config.RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # 1. SST ↔ CTD validation
    logger.info("Running SST ↔ CTD surface temperature validation...")
    sst_ctd = validate_sst_ctd_surface_temp()
    sst_ctd.to_parquet(
        config.RELIABILITY_DIR / "sst_ctd_validation.parquet", index=False
    )
    results["sst_ctd_validation"] = sst_ctd

    # 2. Gap interpolation
    logger.info("Running temporal gap interpolation via SST...")
    gaps = interpolate_sst_for_gaps()
    gaps.to_parquet(
        config.RELIABILITY_DIR / "gap_interpolation.parquet", index=False
    )
    results["gap_interpolation"] = gaps

    # 3. Diversity prediction
    logger.info("Running diversity prediction from environment...")
    div_pred = predict_diversity_from_env()
    div_pred.to_parquet(
        config.RELIABILITY_DIR / "diversity_prediction.parquet", index=False
    )
    results["diversity_prediction"] = div_pred

    # 4. Cross-source corroboration
    logger.info("Computing cross-source corroboration...")
    corrob = corroborate_cross_source(sst_ctd, div_pred)
    corrob.to_parquet(
        config.RELIABILITY_DIR / "corroboration.parquet", index=False
    )
    results["corroboration"] = corrob

    # 5. Build reliability documents
    logger.info("Building reliability documents for RAG...")
    rel_docs = build_reliability_documents(sst_ctd, gaps, div_pred, corrob)
    doc_path = config.RELIABILITY_DIR / "reliability_documents.jsonl"
    with open(doc_path, "w", encoding="utf-8") as f:
        for d in rel_docs:
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    results["documents"] = rel_docs

    logger.info("Reliability ensurance complete: %d outputs", len(results))
    return results
