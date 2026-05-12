"""Tests for ingestion/provenance.py — file registration and SHA-256 hashing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ingestion.provenance import ProvenanceRegistry, ProvenanceRecord


class TestProvenanceRegistry:
    """Validate provenance registration, dedup, and persistence."""

    def _make_file(self, tmp_path: Path, name: str, content: str) -> Path:
        """Create a temp file with known content."""
        p = tmp_path / name
        p.write_text(content)
        return p

    def test_register_file(self, tmp_path: Path):
        """Registering a file creates a JSONL record with correct SHA-256."""
        registry_path = tmp_path / "provenance.jsonl"
        reg = ProvenanceRegistry(registry_path)

        f = self._make_file(tmp_path, "test.tsv", "col1\tcol2\n1\t2\n")
        rec = reg.register(f, source_dataset="ctd", processing_run="test_run")

        assert isinstance(rec, ProvenanceRecord)
        assert rec.source_dataset == "ctd"
        assert rec.processing_run == "test_run"
        assert len(rec.sha256) == 64  # SHA-256 hex digest
        assert rec.file_size_bytes > 0
        assert registry_path.exists()
        assert len(reg) == 1

    def test_duplicate_skipped(self, tmp_path: Path):
        """Registering the same file twice returns the original record."""
        registry_path = tmp_path / "provenance.jsonl"
        reg = ProvenanceRegistry(registry_path)

        f = self._make_file(tmp_path, "data.tsv", "same content")
        rec1 = reg.register(f, source_dataset="ctd", processing_run="run1")
        rec2 = reg.register(f, source_dataset="ctd", processing_run="run2")

        assert rec1.sha256 == rec2.sha256
        assert len(reg) == 1  # Only one record stored

    def test_different_files_registered(self, tmp_path: Path):
        """Different files get separate records."""
        registry_path = tmp_path / "provenance.jsonl"
        reg = ProvenanceRegistry(registry_path)

        f1 = self._make_file(tmp_path, "a.tsv", "content A")
        f2 = self._make_file(tmp_path, "b.tsv", "content B")
        reg.register(f1, source_dataset="ctd", processing_run="run1")
        reg.register(f2, source_dataset="metagenome", processing_run="run1")

        assert len(reg) == 2
        assert reg.records[0].sha256 != reg.records[1].sha256

    def test_lookup_sha(self, tmp_path: Path):
        """Can find a registered record by its SHA-256."""
        registry_path = tmp_path / "provenance.jsonl"
        reg = ProvenanceRegistry(registry_path)

        f = self._make_file(tmp_path, "test.tsv", "lookup test")
        rec = reg.register(f, source_dataset="ctd", processing_run="run1")

        found = reg.lookup_sha(rec.sha256)
        assert found is not None
        assert found.source_dataset == "ctd"

    def test_lookup_unknown_sha(self, tmp_path: Path):
        """Looking up an unknown SHA returns None."""
        registry_path = tmp_path / "provenance.jsonl"
        reg = ProvenanceRegistry(registry_path)
        assert reg.lookup_sha("0" * 64) is None

    def test_to_dataframe(self, tmp_path: Path):
        """Registry converts to a pandas DataFrame."""
        registry_path = tmp_path / "provenance.jsonl"
        reg = ProvenanceRegistry(registry_path)

        f = self._make_file(tmp_path, "test.tsv", "df test")
        reg.register(f, source_dataset="ctd", processing_run="run1")

        df = reg.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "sha256" in df.columns
        assert "source_dataset" in df.columns

    def test_persistence_across_instances(self, tmp_path: Path):
        """A new ProvenanceRegistry instance loads records from disk."""
        registry_path = tmp_path / "provenance.jsonl"

        # First instance: register a file
        reg1 = ProvenanceRegistry(registry_path)
        f = self._make_file(tmp_path, "test.tsv", "persist test")
        reg1.register(f, source_dataset="sst", processing_run="run1")

        # Second instance: should load from JSONL
        reg2 = ProvenanceRegistry(registry_path)
        assert len(reg2) == 1
        assert reg2.records[0].source_dataset == "sst"
