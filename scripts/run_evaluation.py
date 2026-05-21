#!/usr/bin/env python3
"""
Run the evaluation benchmark from the command line.

Executes benchmark questions through the RAG pipeline, measures
6 quantitative metrics, and saves timestamped results with a
markdown summary report.

Usage:
    # Full benchmark (15 questions × 4 modes = 60 evaluations)
    python scripts/run_evaluation.py

    # Quick run (5 questions × 4 modes = 20 evaluations)
    python scripts/run_evaluation.py --quick

    # Specific model with a tag
    python scripts/run_evaluation.py --model qwen2.5:7b-instruct --tag "small-model"

    # Multi-model comparison
    python scripts/run_evaluation.py --models qwen2.5:14b-instruct,qwen2.5:7b-instruct

    # Filter by category or question IDs
    python scripts/run_evaluation.py --categories "Reliability-dependent"
    python scripts/run_evaluation.py --questions ctd_01,dual_02,reliability_01

    # Filter by mode
    python scripts/run_evaluation.py --modes Baseline,Full
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evaluation")

import config
from evaluation.benchmark import (
    BENCHMARK_QUESTIONS,
    EVAL_MODES,
    BenchmarkQuestion,
    EvalMode,
    run_full_benchmark,
)
from evaluation.report import generate_report, print_summary


def _check_ollama(url: str) -> bool:
    """Check if Ollama is reachable."""
    try:
        import requests
        resp = requests.get(f"{url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _check_backend() -> str:
    """Check which retrieval backend is available."""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            n = conn.execute(
                text("SELECT count(*) FROM retrieval_document WHERE embedding IS NOT NULL")
            ).scalar()
        if n and n > 0:
            return f"pgvector ({n} embedded docs)"
    except Exception:
        pass
    return "local (BM25 fallback)"


def _select_questions(
    question_ids: str | None,
    categories: str | None,
    quick: bool,
) -> list[BenchmarkQuestion]:
    """Filter benchmark questions based on CLI arguments."""
    questions = list(BENCHMARK_QUESTIONS)

    if question_ids:
        ids = {q.strip() for q in question_ids.split(",")}
        questions = [q for q in questions if q.id in ids]
        if not questions:
            logger.error("No matching question IDs found: %s", question_ids)
            sys.exit(1)

    if categories:
        cats = {c.strip().lower() for c in categories.split(",")}
        questions = [q for q in questions if q.category.lower() in cats]
        if not questions:
            logger.error("No matching categories found: %s", categories)
            sys.exit(1)

    if quick:
        # 1 per category
        seen = set()
        filtered = []
        for q in questions:
            if q.category not in seen:
                filtered.append(q)
                seen.add(q.category)
        questions = filtered

    return questions


def _select_modes(mode_filter: str | None) -> list[EvalMode]:
    """Filter evaluation modes based on CLI arguments."""
    if not mode_filter:
        return list(EVAL_MODES)

    names = {m.strip() for m in mode_filter.split(",")}
    modes = [m for m in EVAL_MODES if m.name in names]
    if not modes:
        logger.error("No matching modes found: %s (available: %s)",
                     mode_filter, ", ".join(m.name for m in EVAL_MODES))
        sys.exit(1)
    return modes


def _make_run_id(model: str) -> str:
    """Create a timestamped run ID."""
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
    model_slug = model.replace(":", "-").replace("/", "-")
    return f"eval_{ts}_{model_slug}"


def _save_results(
    results_df,
    run_meta: dict,
    output_dir: Path,
    run_id: str,
) -> tuple[Path, Path, Path]:
    """Save CSV, meta JSON, and markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{run_id}.csv"
    meta_path = output_dir / f"{run_id}_meta.json"
    report_path = output_dir / f"{run_id}_report.md"

    # CSV
    results_df.to_csv(csv_path, index=False)

    # Meta JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)

    # Report
    report_md = generate_report(results_df, run_meta)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return csv_path, meta_path, report_path


def run_single_model(
    model: str,
    questions: list[BenchmarkQuestion],
    modes: list[EvalMode],
    *,
    ollama_url: str,
    top_k: int,
    num_ctx: int,
    output_dir: Path,
    tag: str | None,
) -> Path:
    """Run evaluation for a single model and save results."""
    run_id = _make_run_id(model)
    total = len(questions) * len(modes)
    backend = _check_backend()

    print(f"\n{'=' * 70}")
    print(f"  Run ID:     {run_id}")
    print(f"  Model:      {model}")
    print(f"  Questions:  {len(questions)} × {len(modes)} modes = {total} evaluations")
    print(f"  Backend:    {backend}")
    print(f"  Top-K:      {top_k}  |  Context: {num_ctx}")
    if tag:
        print(f"  Tag:        {tag}")
    print(f"{'=' * 70}\n")

    t0 = time.time()

    def _progress(current: int, total: int) -> None:
        if current < total:
            q_idx = current // len(modes)
            m_idx = current % len(modes)
            if q_idx < len(questions):
                q = questions[q_idx]
                m = modes[m_idx]
                elapsed = time.time() - t0
                avg = elapsed / max(current, 1)
                remaining = avg * (total - current)
                print(
                    f"  [{current + 1:>3}/{total}] {q.id:<20} / {m.name:<15} "
                    f"(~{remaining:.0f}s remaining)",
                    flush=True,
                )

    results_df = run_full_benchmark(
        model=model,
        ollama_url=ollama_url,
        top_k=top_k,
        temperature=0.0,
        num_ctx=num_ctx,
        questions=questions,
        modes=modes,
        progress_callback=_progress,
    )

    duration = time.time() - t0
    n_errors = (results_df["error"] != "").sum()

    # Build metadata
    run_meta = {
        "run_id": run_id,
        "model": model,
        "tag": tag,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "n_modes": len(modes),
        "n_evaluations": len(results_df),
        "top_k": top_k,
        "temperature": 0.0,
        "num_ctx": num_ctx,
        "backend": backend,
        "duration_seconds": round(duration, 1),
        "n_errors": int(n_errors),
        "question_ids": [q.id for q in questions],
        "mode_names": [m.name for m in modes],
    }

    # Save
    csv_path, meta_path, report_path = _save_results(
        results_df, run_meta, output_dir, run_id
    )

    # Terminal summary
    print_summary(results_df)

    print(f"  Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    if n_errors > 0:
        print(f"  ⚠ {n_errors} evaluations had errors")
    print(f"\n  Saved to:")
    print(f"    CSV:    {csv_path}")
    print(f"    Meta:   {meta_path}")
    print(f"    Report: {report_path}")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Run the Onagawa Source Chat evaluation benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_evaluation.py                           # Full benchmark
  python scripts/run_evaluation.py --quick                   # 5 questions × 4 modes
  python scripts/run_evaluation.py --model qwen2.5:7b-instruct --tag v2
  python scripts/run_evaluation.py --models qwen2.5:14b-instruct,qwen2.5:7b-instruct
  python scripts/run_evaluation.py --categories "Reliability-dependent" --modes Baseline,Full
  python scripts/run_evaluation.py --questions ctd_01,dual_02
        """,
    )
    parser.add_argument(
        "--model", default=config.CHAT_MODEL,
        help=f"Ollama model name (default: {config.CHAT_MODEL})",
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model list for multi-model comparison",
    )
    parser.add_argument(
        "--questions", default=None,
        help="Comma-separated question IDs to run (e.g., ctd_01,dual_02)",
    )
    parser.add_argument(
        "--categories", default=None,
        help="Comma-separated category names to filter",
    )
    parser.add_argument(
        "--modes", default=None,
        help="Comma-separated mode names (e.g., Baseline,Full)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick evaluation: 1 question per category (5 × 4 = 20 evals)",
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Number of documents to retrieve (default: 8)",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=8192,
        help="LLM context window size (default: 8192)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Output directory (default: {config.EVALUATION_DIR})",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional tag for this run (e.g., 'thesis-final')",
    )
    parser.add_argument(
        "--ollama-url", default=config.OLLAMA_BASE_URL,
        help=f"Ollama API URL (default: {config.OLLAMA_BASE_URL})",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or config.EVALUATION_DIR

    # Preflight checks
    print("\n  Preflight checks...")

    if not _check_ollama(args.ollama_url):
        print(f"  ✗ Ollama not reachable at {args.ollama_url}")
        print("    Start Ollama first: ollama serve")
        sys.exit(1)
    print(f"  ✓ Ollama reachable at {args.ollama_url}")

    backend = _check_backend()
    print(f"  ✓ Retrieval backend: {backend}")

    # Select questions and modes
    questions = _select_questions(args.questions, args.categories, args.quick)
    modes = _select_modes(args.modes)

    print(f"  ✓ Questions: {len(questions)} selected")
    print(f"  ✓ Modes: {', '.join(m.name for m in modes)}")

    # Determine models to run
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
    else:
        model_list = [args.model]

    print(f"  ✓ Models: {', '.join(model_list)}")

    # Run evaluations
    csv_paths = []
    for model in model_list:
        csv_path = run_single_model(
            model=model,
            questions=questions,
            modes=modes,
            ollama_url=args.ollama_url,
            top_k=args.top_k,
            num_ctx=args.num_ctx,
            output_dir=output_dir,
            tag=args.tag,
        )
        csv_paths.append(csv_path)

    # Multi-model comparison
    if len(csv_paths) > 1:
        print(f"\n{'=' * 70}")
        print("  Multi-Model Comparison")
        print(f"{'=' * 70}")
        from evaluation.report import print_comparison
        print_comparison(csv_paths)

    print(f"\n  All results saved to: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
