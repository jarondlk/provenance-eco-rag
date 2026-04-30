#!/usr/bin/env python3
"""
Run the pre-analysis pipeline.

Computes ecological relationships between CTD, metagenome, and SST data:
  - CTD monthly trends per bay
  - Taxa–environment correlations (Spearman)
  - Community diversity indices (Shannon, Simpson)
  - Bay comparisons
  - Taxa co-occurrence matrix

Usage:
    python scripts/run_pre_analysis.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from preprocessing.pre_analysis import run_all
import config


def main():
    print("=" * 60)
    print("Pre-Analysis Pipeline")
    print("=" * 60)

    results = run_all()

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for name, data in results.items():
        if name == "documents":
            print(f"  {name}: {len(data)} analysis documents")
        elif hasattr(data, "shape"):
            print(f"  {name}: {data.shape}")
        else:
            print(f"  {name}: {type(data).__name__}")

    print(f"\nOutputs saved to: {config.ANALYSIS_DIR}")
    for f in sorted(config.ANALYSIS_DIR.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
