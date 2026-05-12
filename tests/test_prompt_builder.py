"""Tests for orchestration/unified.py — prompt building and context injection."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestration.unified import build_prompt, _load_analysis_context, _load_reliability_context


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    """Validate prompt structure and content."""

    def _sample_results(self) -> list:
        """Minimal retrieved documents."""
        return [
            {
                "doc_id": "ctd_2024-04-O-s1",
                "source_type": "ctd",
                "title": "CTD cast at Onagawa Bay",
                "text": "Temperature 10.5°C at 0m depth.",
                "time": "2024-04-15",
                "score": 0.85,
            },
            {
                "doc_id": "meta_2024-04-O-s1",
                "source_type": "metagenome",
                "title": "Metagenome sample at Onagawa",
                "text": "Dominant genus: Gyrodinium (15%).",
                "time": "2024-04",
                "score": 0.72,
            },
        ]

    def test_prompt_contains_evidence(self):
        """Retrieved documents appear in the prompt."""
        prompt = build_prompt("What is the temperature?", self._sample_results(),
                              inject_analysis=False, inject_reliability=False)
        assert "ctd_2024-04-O-s1" in prompt
        assert "10.5°C" in prompt
        assert "Gyrodinium" in prompt

    def test_prompt_contains_system_rules(self):
        """System prompt includes citation rules."""
        prompt = build_prompt("test", self._sample_results(),
                              inject_analysis=False, inject_reliability=False)
        assert "[doc_id]" in prompt
        assert "ONLY use the evidence" in prompt

    def test_prompt_contains_study_sites(self):
        """System prompt includes study site coordinates."""
        prompt = build_prompt("test", self._sample_results(),
                              inject_analysis=False, inject_reliability=False)
        assert "Onagawa Bay" in prompt
        assert "38.44" in prompt

    def test_prompt_with_empty_results(self):
        """Empty results still produce a valid prompt."""
        prompt = build_prompt("test", [], inject_analysis=False, inject_reliability=False)
        assert "RULES" in prompt
        assert len(prompt) > 100

    def test_prompt_source_type_in_evidence(self):
        """Evidence section includes source type labels."""
        prompt = build_prompt("test", self._sample_results(),
                              inject_analysis=False, inject_reliability=False)
        assert "ctd" in prompt.lower()
        assert "metagenome" in prompt.lower()


# ---------------------------------------------------------------------------
# _load_analysis_context
# ---------------------------------------------------------------------------
class TestLoadAnalysisContext:
    """Validate keyword-triggered analysis context injection."""

    def test_keyword_triggers_injection(self, sample_analysis_docs):
        """Query with 'correlation' triggers analysis context."""
        with patch("config.ANALYSIS_DIR", sample_analysis_docs.parent):
            result = _load_analysis_context("What are the correlation patterns?")
        assert "PRE-COMPUTED ANALYSES" in result
        assert "analysis_trends" in result

    def test_no_keyword_skips_injection(self, sample_analysis_docs):
        """Query without eco keywords returns empty string."""
        with patch("config.ANALYSIS_DIR", sample_analysis_docs.parent):
            result = _load_analysis_context("Hello, how are you?")
        assert result == ""

    def test_diversity_keyword(self, sample_analysis_docs):
        """'diversity' keyword triggers injection."""
        with patch("config.ANALYSIS_DIR", sample_analysis_docs.parent):
            result = _load_analysis_context("What is the diversity index?")
        assert "PRE-COMPUTED ANALYSES" in result

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing JSONL file returns empty string gracefully."""
        with patch("config.ANALYSIS_DIR", tmp_path):
            result = _load_analysis_context("correlation trends")
        assert result == ""


# ---------------------------------------------------------------------------
# _load_reliability_context
# ---------------------------------------------------------------------------
class TestLoadReliabilityContext:
    """Validate keyword-triggered reliability context injection."""

    def test_reliability_keyword(self, sample_reliability_docs):
        """Query with 'reliable' triggers reliability context."""
        with patch("config.RELIABILITY_DIR", sample_reliability_docs.parent):
            result = _load_reliability_context("Is the temperature data reliable?")
        assert "RELIABILITY ENSURANCE" in result
        assert "reliability_sst_ctd_validation" in result

    def test_temperature_keyword(self, sample_reliability_docs):
        """'temperature' keyword triggers reliability context."""
        with patch("config.RELIABILITY_DIR", sample_reliability_docs.parent):
            result = _load_reliability_context("What is the temperature trend?")
        assert "RELIABILITY ENSURANCE" in result

    def test_no_keyword_skips(self, sample_reliability_docs):
        """Generic query without keywords returns empty."""
        with patch("config.RELIABILITY_DIR", sample_reliability_docs.parent):
            result = _load_reliability_context("Hello world")
        assert result == ""

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing JSONL file returns empty string gracefully."""
        with patch("config.RELIABILITY_DIR", tmp_path):
            result = _load_reliability_context("reliable SST data")
        assert result == ""


# ---------------------------------------------------------------------------
# Integration: build_prompt with injection
# ---------------------------------------------------------------------------
class TestBuildPromptWithInjection:
    """Test that build_prompt integrates context injection."""

    def test_analysis_injected_when_enabled(self, sample_analysis_docs):
        """Analysis context appears in prompt when inject_analysis=True and keywords match."""
        results = [{"doc_id": "test", "source_type": "ctd", "title": "Test",
                     "text": "Test data", "time": "2024-01", "score": 0.9}]
        with patch("config.ANALYSIS_DIR", sample_analysis_docs.parent):
            prompt = build_prompt("What are the correlation patterns?", results,
                                  inject_analysis=True, inject_reliability=False)
        assert "PRE-COMPUTED ANALYSES" in prompt

    def test_reliability_injected_when_enabled(self, sample_reliability_docs):
        """Reliability context appears when inject_reliability=True and keywords match."""
        results = [{"doc_id": "test", "source_type": "ctd", "title": "Test",
                     "text": "Test data", "time": "2024-01", "score": 0.9}]
        with patch("config.RELIABILITY_DIR", sample_reliability_docs.parent):
            prompt = build_prompt("Is temperature data reliable?", results,
                                  inject_analysis=False, inject_reliability=True)
        assert "RELIABILITY ENSURANCE" in prompt

    def test_both_disabled(self):
        """Neither context appears when both are disabled."""
        results = [{"doc_id": "test", "source_type": "ctd", "title": "Test",
                     "text": "Test data", "time": "2024-01", "score": 0.9}]
        prompt = build_prompt("correlation reliable temperature", results,
                              inject_analysis=False, inject_reliability=False)
        assert "PRE-COMPUTED ANALYSES" not in prompt
        assert "RELIABILITY ENSURANCE" not in prompt
