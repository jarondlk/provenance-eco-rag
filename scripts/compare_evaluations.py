#!/usr/bin/env python3
"""
Compare multiple evaluation runs.

Loads evaluation result CSVs from data/evaluation/ and produces
a side-by-side comparison showing which model/configuration
performs best across different metrics and categories.

Usage:
    # Compare all runs in the evaluation directory
    python scripts/compare_evaluations.py --all

    # Compare specific CSV files
    python scripts/compare_evaluations.py data/evaluation/eval_*_qwen2.5*.csv data/evaluation/eval_*_llama3*.csv

    # Save comparison report to file
    python scripts/compare_evaluations.py --all --save
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluation.report import compare_runs, print_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_evaluations.py --all
  python scripts/compare_evaluations.py data/evaluation/run_a.csv data/evaluation/run_b.csv
  python scripts/compare_evaluations.py --all --save
        """,
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="Paths to evaluation result CSV files to compare",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Compare all evaluation CSVs in data/evaluation/",
    )
    parser.add_argument(
        "--dir", type=Path, default=None,
        help=f"Directory to scan for CSVs (default: {config.EVALUATION_DIR})",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save comparison report as markdown file",
    )

    args = parser.parse_args()
    eval_dir = args.dir or config.EVALUATION_DIR

    # Collect CSV paths
    csv_paths: list[Path] = []

    if args.all:
        if not eval_dir.exists():
            print(f"  Evaluation directory not found: {eval_dir}")
            sys.exit(1)
        csv_paths = sorted(
            p for p in eval_dir.glob("*.csv")
            if not p.name.endswith("_comparison.csv")
        )
    elif args.csv_files:
        csv_paths = [Path(f) for f in args.csv_files]
    else:
        parser.print_help()
        sys.exit(1)

    # Validate
    csv_paths = [p for p in csv_paths if p.exists()]
    if len(csv_paths) < 2:
        print(f"  Need at least 2 evaluation CSVs to compare (found {len(csv_paths)}).")
        if not csv_paths:
            print(f"  No CSVs found in {eval_dir}")
            print(f"  Run evaluations first: python scripts/run_evaluation.py")
        sys.exit(1)

    print(f"\n  Comparing {len(csv_paths)} evaluation runs:")
    for p in csv_paths:
        print(f"    {p.name}")
    print()

    # Generate comparison
    report = compare_runs(csv_paths)

    # Print to terminal
    print_comparison(csv_paths)

    # Save if requested
    if args.save:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
        save_path = eval_dir / f"comparison_{ts}.md"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  Comparison report saved to: {save_path}")

    print()


if __name__ == "__main__":
    main()
