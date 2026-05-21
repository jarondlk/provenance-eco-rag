"""
Evaluation report generation and multi-run comparison.

Generates markdown summary reports from benchmark results and
compares performance across multiple evaluation runs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.benchmark import compute_summary_metrics

logger = logging.getLogger(__name__)

# Metric display names and format strings
METRIC_DISPLAY = {
    "retrieval_precision": ("Retrieval Precision", ".1%"),
    "source_coverage": ("Source Coverage", ".1%"),
    "citation_count": ("Avg Citations", ".1f"),
    "citation_accuracy": ("Citation Accuracy", ".1%"),
    "context_utilization": ("Context Utilization", ".1%"),
    "latency_seconds": ("Avg Latency (s)", ".1f"),
}

METRIC_COLS = list(METRIC_DISPLAY.keys())


def _fmt(val: float, fmt: str) -> str:
    """Format a metric value with the given format spec."""
    if fmt.endswith("%"):
        return f"{val:{fmt}}"
    return f"{val:{fmt}}"


# ─────────────────────────────────────────────
# Terminal Summary
# ─────────────────────────────────────────────
def print_summary(results_df: pd.DataFrame) -> None:
    """Print a quick terminal-friendly summary of evaluation results."""
    n_total = len(results_df)
    n_errors = (results_df["error"] != "").sum()
    modes = results_df["mode"].unique()
    categories = results_df["category"].unique()

    print(f"\n{'=' * 70}")
    print(f"  Evaluation Summary: {n_total} evaluations")
    print(f"  Modes: {', '.join(modes)}")
    print(f"  Categories: {len(categories)}")
    if n_errors > 0:
        print(f"  ⚠ Errors: {n_errors}")
    print(f"{'=' * 70}")

    # By mode
    summaries = compute_summary_metrics(results_df)
    by_mode = summaries["by_mode"]

    # Header
    header = f"  {'Mode':<16}"
    for col in METRIC_COLS:
        name = METRIC_DISPLAY[col][0]
        header += f"  {name:>12}"
    print(f"\n{header}")
    print(f"  {'-' * (16 + 14 * len(METRIC_COLS))}")

    # Rows
    for mode_name, row in by_mode.iterrows():
        line = f"  {mode_name:<16}"
        for col in METRIC_COLS:
            fmt = METRIC_DISPLAY[col][1]
            line += f"  {_fmt(row[col], fmt):>12}"
        print(line)

    # Best mode highlight
    if len(by_mode) > 1:
        best_prec = by_mode["retrieval_precision"].idxmax()
        best_cite = by_mode["citation_accuracy"].idxmax()
        print(f"\n  Best retrieval precision: {best_prec}")
        print(f"  Best citation accuracy:  {best_cite}")

    print()


# ─────────────────────────────────────────────
# Markdown Report Generation
# ─────────────────────────────────────────────
def generate_report(
    results_df: pd.DataFrame,
    run_meta: Dict[str, Any],
) -> str:
    """
    Generate a formatted markdown summary report for one evaluation run.

    Args:
        results_df: DataFrame with EvalResult columns.
        run_meta: Dict with run metadata (model, timestamp, config, etc.).

    Returns:
        Markdown-formatted report string.
    """
    lines: List[str] = []
    lines.append(f"# Evaluation Report — {run_meta.get('model', 'unknown')}")
    lines.append("")
    lines.append(f"**Run ID:** {run_meta.get('run_id', 'N/A')}")
    lines.append(f"**Timestamp:** {run_meta.get('timestamp', 'N/A')}")
    lines.append(f"**Model:** `{run_meta.get('model', 'N/A')}`")
    if run_meta.get("tag"):
        lines.append(f"**Tag:** {run_meta['tag']}")
    lines.append(f"**Backend:** {run_meta.get('backend', 'N/A')}")
    lines.append(f"**Config:** top_k={run_meta.get('top_k', '?')}, "
                 f"temperature={run_meta.get('temperature', '?')}, "
                 f"num_ctx={run_meta.get('num_ctx', '?')}")
    lines.append(f"**Evaluations:** {run_meta.get('n_evaluations', len(results_df))} "
                 f"({run_meta.get('n_questions', '?')} questions × "
                 f"{run_meta.get('n_modes', '?')} modes)")
    duration = run_meta.get("duration_seconds", 0)
    if duration:
        lines.append(f"**Duration:** {duration:.1f}s ({duration / 60:.1f} min)")
    lines.append("")

    # Errors
    n_errors = (results_df["error"] != "").sum()
    if n_errors > 0:
        lines.append(f"> [!WARNING]")
        lines.append(f"> {n_errors} evaluations encountered errors.")
        lines.append("")
        error_rows = results_df[results_df["error"] != ""]
        for _, row in error_rows.iterrows():
            lines.append(f"- **{row['question_id']}** / {row['mode']}: "
                         f"`{row['error'][:120]}`")
        lines.append("")

    # Summary metrics
    summaries = compute_summary_metrics(results_df)

    # By Mode
    lines.append("---")
    lines.append("")
    lines.append("## Performance by Mode")
    lines.append("")
    _append_metric_table(lines, summaries["by_mode"], "Mode")

    # Mode comparison insight
    by_mode = summaries["by_mode"]
    if "Baseline" in by_mode.index and "Full" in by_mode.index:
        lines.append("")
        lines.append("### Baseline → Full Delta")
        lines.append("")
        baseline = by_mode.loc["Baseline"]
        full = by_mode.loc["Full"]
        lines.append("| Metric | Baseline | Full | Delta |")
        lines.append("|---|---|---|---|")
        for col in METRIC_COLS:
            fmt = METRIC_DISPLAY[col][1]
            name = METRIC_DISPLAY[col][0]
            b_val = baseline[col]
            f_val = full[col]
            delta = f_val - b_val
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {name} | {_fmt(b_val, fmt)} | {_fmt(f_val, fmt)} | "
                f"{sign}{_fmt(delta, fmt)} |"
            )
        lines.append("")

    # By Category
    lines.append("---")
    lines.append("")
    lines.append("## Performance by Category")
    lines.append("")
    _append_metric_table(lines, summaries["by_category"], "Category")

    # Key findings
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    _append_findings(lines, results_df, summaries)

    # Per-question summary (compact)
    lines.append("---")
    lines.append("")
    lines.append("## Per-Question Summary (Full Mode)")
    lines.append("")
    full_rows = results_df[results_df["mode"] == "Full"] if "Full" in results_df["mode"].values else results_df
    if not full_rows.empty:
        lines.append("| Question | Category | Precision | Coverage | Citations | Cit. Acc. | Latency |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in full_rows.iterrows():
            lines.append(
                f"| {row['question_id']} | {row['category']} | "
                f"{row['retrieval_precision']:.0%} | {row['source_coverage']:.0%} | "
                f"{row['citation_count']} | {row['citation_accuracy']:.0%} | "
                f"{row['latency_seconds']:.1f}s |"
            )
    lines.append("")

    return "\n".join(lines)


def _append_metric_table(
    lines: List[str],
    df: pd.DataFrame,
    index_label: str,
) -> None:
    """Append a markdown metric table to lines."""
    header = f"| {index_label} |"
    sep = "|---|"
    for col in METRIC_COLS:
        header += f" {METRIC_DISPLAY[col][0]} |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for idx, row in df.iterrows():
        line = f"| {idx} |"
        for col in METRIC_COLS:
            fmt = METRIC_DISPLAY[col][1]
            line += f" {_fmt(row[col], fmt)} |"
        lines.append(line)


def _append_findings(
    lines: List[str],
    results_df: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
) -> None:
    """Append key findings based on the evaluation results."""
    by_mode = summaries["by_mode"]

    # Best overall mode
    if len(by_mode) > 1:
        best_mode = by_mode["citation_accuracy"].idxmax()
        lines.append(f"- **Best citation accuracy:** {best_mode} mode "
                     f"({by_mode.loc[best_mode, 'citation_accuracy']:.0%})")

    # Worst performing questions
    if "Full" in results_df["mode"].values:
        full = results_df[results_df["mode"] == "Full"]
        worst = full.nsmallest(3, "retrieval_precision")
        if not worst.empty:
            lines.append("- **Lowest retrieval precision (Full mode):**")
            for _, row in worst.iterrows():
                lines.append(f"  - {row['question_id']}: {row['retrieval_precision']:.0%}")

    # Context utilization
    for mode_name in ["+Analysis", "+Reliability", "Full"]:
        if mode_name in by_mode.index:
            util = by_mode.loc[mode_name, "context_utilization"]
            lines.append(f"- **Context utilization ({mode_name}):** {util:.0%}")

    # Average latency
    mean_latency = results_df["latency_seconds"].mean()
    lines.append(f"- **Average latency:** {mean_latency:.1f}s per evaluation")

    # Error rate
    n_errors = (results_df["error"] != "").sum()
    if n_errors > 0:
        lines.append(f"- **Error rate:** {n_errors}/{len(results_df)} "
                     f"({n_errors / len(results_df):.0%})")
    else:
        lines.append("- **Error rate:** 0 — all evaluations completed successfully")


# ─────────────────────────────────────────────
# Multi-Run Comparison
# ─────────────────────────────────────────────
def load_run(csv_path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a run's results CSV and its accompanying meta JSON."""
    results_df = pd.read_csv(csv_path)
    meta_path = csv_path.with_name(
        csv_path.stem.replace("_full", "_full_meta").replace(".csv", "") + ".json"
    )
    # Try common meta file naming patterns
    meta = {}
    for candidate in [
        csv_path.with_suffix(".json"),
        csv_path.parent / (csv_path.stem + "_meta.json"),
        meta_path,
    ]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                meta = json.load(f)
            break

    if not meta:
        # Infer basic metadata from filename
        meta = {
            "model": csv_path.stem,
            "run_id": csv_path.stem,
            "n_evaluations": len(results_df),
        }

    return results_df, meta


def compare_runs(run_paths: List[Path]) -> str:
    """
    Load multiple evaluation CSVs and produce a comparison report.

    Args:
        run_paths: List of paths to evaluation result CSV files.

    Returns:
        Markdown-formatted comparison report.
    """
    if len(run_paths) < 2:
        return "Need at least 2 runs to compare."

    runs: List[tuple[pd.DataFrame, Dict[str, Any]]] = []
    for p in run_paths:
        try:
            runs.append(load_run(Path(p)))
        except Exception as e:
            logger.warning("Could not load %s: %s", p, e)

    if len(runs) < 2:
        return "Could not load enough runs for comparison."

    lines: List[str] = []
    lines.append("# Evaluation Comparison Report")
    lines.append("")
    lines.append(f"**Runs compared:** {len(runs)}")
    lines.append(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # Run overview
    lines.append("## Runs")
    lines.append("")
    lines.append("| # | Model | Tag | Evaluations | Date |")
    lines.append("|---|---|---|---|---|")
    labels = []
    for i, (df, meta) in enumerate(runs):
        model = meta.get("model", "unknown")
        tag = meta.get("tag", "—")
        ts = meta.get("timestamp", "—")
        label = f"{model}" + (f" ({tag})" if tag != "—" else "")
        labels.append(label)
        lines.append(f"| {i + 1} | `{model}` | {tag} | {len(df)} | {ts} |")
    lines.append("")

    # Side-by-side mode performance
    lines.append("---")
    lines.append("")
    lines.append("## Performance by Mode (Side-by-Side)")
    lines.append("")

    for mode_name in ["Baseline", "+Analysis", "+Reliability", "Full"]:
        mode_data_available = False
        for df, _ in runs:
            if mode_name in df["mode"].values:
                mode_data_available = True
                break

        if not mode_data_available:
            continue

        lines.append(f"### {mode_name}")
        lines.append("")

        header = "| Metric |"
        sep = "|---|"
        for label in labels:
            header += f" {label} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)

        for col in METRIC_COLS:
            name = METRIC_DISPLAY[col][0]
            fmt = METRIC_DISPLAY[col][1]
            row_line = f"| {name} |"
            for df, _ in runs:
                mode_df = df[df["mode"] == mode_name]
                if mode_df.empty:
                    row_line += " — |"
                else:
                    val = mode_df[col].mean()
                    row_line += f" {_fmt(val, fmt)} |"
            lines.append(row_line)

        lines.append("")

    # Best model per category
    lines.append("---")
    lines.append("")
    lines.append("## Best Model per Category (Full Mode)")
    lines.append("")

    full_dfs = []
    for i, (df, meta) in enumerate(runs):
        full = df[df["mode"] == "Full"].copy()
        if not full.empty:
            full["_run_label"] = labels[i]
            full_dfs.append(full)

    if full_dfs:
        combined = pd.concat(full_dfs, ignore_index=True)
        categories = combined["category"].unique()

        lines.append("| Category | Best Model | Precision | Cit. Accuracy |")
        lines.append("|---|---|---|---|")

        for cat in sorted(categories):
            cat_df = combined[combined["category"] == cat]
            # Score by average of precision + citation accuracy
            scores = cat_df.groupby("_run_label")[
                ["retrieval_precision", "citation_accuracy"]
            ].mean()
            scores["combined"] = scores.mean(axis=1)
            best = scores["combined"].idxmax()
            prec = scores.loc[best, "retrieval_precision"]
            acc = scores.loc[best, "citation_accuracy"]
            lines.append(f"| {cat} | {best} | {prec:.0%} | {acc:.0%} |")

        lines.append("")

    return "\n".join(lines)


def print_comparison(run_paths: List[Path]) -> None:
    """Print comparison to terminal."""
    report = compare_runs(run_paths)
    # Convert markdown to simple terminal output
    for line in report.split("\n"):
        # Strip markdown formatting for terminal
        line = line.replace("**", "").replace("`", "").replace("###", " ").replace("##", "").replace("#", "")
        print(line)
