#!/usr/bin/env python3
"""
Run the reliability ensurance pipeline.

Cross-source validation and corroboration:
  - SST ↔ CTD surface temperature validation
  - Temporal gap interpolation via SST
  - Environment-based diversity prediction
  - Cross-source corroboration scoring

Usage:
    python scripts/run_reliability.py
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

from preprocessing.reliability_ensurance import run_all
import config


def main():
    print("=" * 60)
    print("Reliability Ensurance Pipeline")
    print("=" * 60)

    results = run_all()

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for name, data in results.items():
        if name == "documents":
            print(f"  {name}: {len(data)} reliability documents")
        elif hasattr(data, "shape"):
            print(f"  {name}: {data.shape}")
        else:
            print(f"  {name}: {type(data).__name__}")

    print(f"\nOutputs saved to: {config.RELIABILITY_DIR}")
    if config.RELIABILITY_DIR.exists():
        for f in sorted(config.RELIABILITY_DIR.glob("*")):
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
