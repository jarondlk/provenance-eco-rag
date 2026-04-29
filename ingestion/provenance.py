"""
Provenance registry – tracks every raw file that enters the system.

Each file is registered with a SHA-256 hash, source dataset label,
and processing run metadata so that downstream outputs can be traced
back to their exact inputs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ProvenanceRecord:
    """One record per ingested raw file."""

    source_dataset: str          # e.g. "ctd", "metagenome", "remote_sensing"
    source_file: str             # relative path within data/raw/
    sha256: str                  # hex digest of file content
    file_size_bytes: int
    ingested_at: str             # ISO-8601 UTC
    processing_run: str          # identifier for the pipeline run
    notes: Optional[str] = None


class ProvenanceRegistry:
    """
    Append-only registry of provenance records.

    Persisted as a JSONL file for simplicity, convertible to parquet.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: List[ProvenanceRecord] = []
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._records.append(ProvenanceRecord(**obj))

    def _append(self, rec: ProvenanceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(
        self,
        file_path: Path,
        source_dataset: str,
        processing_run: str,
        *,
        notes: Optional[str] = None,
    ) -> ProvenanceRecord:
        """Register a file.  Skips if the sha256 is already known."""
        sha = _sha256(file_path)
        if self.lookup_sha(sha):
            return self.lookup_sha(sha)  # type: ignore[return-value]

        rec = ProvenanceRecord(
            source_dataset=source_dataset,
            source_file=str(file_path),
            sha256=sha,
            file_size_bytes=file_path.stat().st_size,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            processing_run=processing_run,
            notes=notes,
        )
        self._records.append(rec)
        self._append(rec)
        return rec

    def lookup_sha(self, sha: str) -> Optional[ProvenanceRecord]:
        for r in self._records:
            if r.sha256 == sha:
                return r
        return None

    @property
    def records(self) -> List[ProvenanceRecord]:
        return list(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self._records])

    def to_parquet(self, path: Path) -> None:
        self.to_dataframe().to_parquet(path, index=False)

    def __len__(self) -> int:
        return len(self._records)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _sha256(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
