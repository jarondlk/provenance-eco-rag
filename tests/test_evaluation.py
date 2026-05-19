"""Tests for evaluation/benchmark.py — citation extraction, metrics, and question definitions."""
from __future__ import annotations

import pandas as pd
import pytest

from evaluation.benchmark import (
    BENCHMARK_QUESTIONS,
    EVAL_MODES,
    EvalResult,
    extract_citations,
    compute_citation_accuracy,
    compute_summary_metrics,
)


# ---------------------------------------------------------------------------
# Benchmark question definitions
# ---------------------------------------------------------------------------
class TestBenchmarkQuestions:
    """Validate the benchmark question set."""

    def test_has_15_questions(self):
        """Benchmark has exactly 15 questions."""
        assert len(BENCHMARK_QUESTIONS) == 15

    def test_five_categories(self):
        """Questions span exactly 5 categories."""
        categories = {q.category for q in BENCHMARK_QUESTIONS}
        assert len(categories) == 5

    def test_three_per_category(self):
        """Each category has exactly 3 questions."""
        from collections import Counter
        counts = Counter(q.category for q in BENCHMARK_QUESTIONS)
        for cat, n in counts.items():
            assert n == 3, f"{cat} has {n} questions, expected 3"

    def test_unique_ids(self):
        """All question IDs are unique."""
        ids = [q.id for q in BENCHMARK_QUESTIONS]
        assert len(ids) == len(set(ids))

    def test_expected_source_types_not_empty(self):
        """Every question has at least one expected source type."""
        for q in BENCHMARK_QUESTIONS:
            assert len(q.expected_source_types) >= 1, f"{q.id} has no expected sources"

    def test_analysis_questions_flagged(self):
        """Analysis-dependent questions have requires_analysis=True."""
        analysis_qs = [q for q in BENCHMARK_QUESTIONS if q.category == "Analysis-dependent"]
        assert len(analysis_qs) == 3
        assert all(q.requires_analysis for q in analysis_qs)

    def test_reliability_questions_flagged(self):
        """Reliability-dependent questions have requires_reliability=True."""
        rel_qs = [q for q in BENCHMARK_QUESTIONS if q.category == "Reliability-dependent"]
        assert len(rel_qs) == 3
        assert all(q.requires_reliability for q in rel_qs)


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------
class TestEvalModes:
    """Validate evaluation mode definitions."""

    def test_has_4_modes(self):
        assert len(EVAL_MODES) == 4

    def test_baseline_mode(self):
        baseline = EVAL_MODES[0]
        assert baseline.name == "Baseline"
        assert baseline.inject_analysis is False
        assert baseline.inject_reliability is False

    def test_full_mode(self):
        full = EVAL_MODES[3]
        assert full.name == "Full"
        assert full.inject_analysis is True
        assert full.inject_reliability is True


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------
class TestExtractCitations:
    """Validate citation regex extraction from LLM responses."""

    def test_doc_citations(self):
        text = "According to [ctd_2024-04-O-s1] and [meta_2024-05-O-s1], the temperature..."
        cited = extract_citations(text)
        assert "ctd_2024-04-O-s1" in cited
        assert "meta_2024-05-O-s1" in cited
        assert len(cited) == 2

    def test_analysis_citations(self):
        text = "Based on [analysis_trends] and [analysis_correlations]..."
        cited = extract_citations(text)
        assert "analysis_trends" in cited
        assert "analysis_correlations" in cited

    def test_reliability_citations(self):
        text = "The [reliability_sst_ctd_validation] shows agreement."
        cited = extract_citations(text)
        assert "reliability_sst_ctd_validation" in cited

    def test_mixed_citations(self):
        text = (
            "Data from [ctd_2024-04-O-s1] shows temperature. "
            "Cross-validation [reliability_corroboration_summary] confirms. "
            "Trends from [analysis_trends] support this."
        )
        cited = extract_citations(text)
        assert len(cited) == 3

    def test_no_citations(self):
        text = "This response has no source citations."
        cited = extract_citations(text)
        assert len(cited) == 0

    def test_sst_citations(self):
        text = "Satellite data [sst_2024-04-10] shows surface temperature."
        cited = extract_citations(text)
        assert "sst_2024-04-10" in cited

    def test_duplicate_citations(self):
        text = "See [ctd_2024-04-O-s1]. Again [ctd_2024-04-O-s1]."
        cited = extract_citations(text)
        assert len(cited) == 2  # Both instances extracted


# ---------------------------------------------------------------------------
# Citation accuracy
# ---------------------------------------------------------------------------
class TestComputeCitationAccuracy:
    """Validate citation accuracy computation."""

    def test_all_valid(self):
        accuracy = compute_citation_accuracy(
            cited_ids=["ctd_a", "ctd_b"],
            retrieved_ids=["ctd_a", "ctd_b", "ctd_c"],
            analysis_ids=[],
            reliability_ids=[],
        )
        assert accuracy == 1.0

    def test_none_valid(self):
        accuracy = compute_citation_accuracy(
            cited_ids=["hallucinated_doc"],
            retrieved_ids=["ctd_a"],
            analysis_ids=[],
            reliability_ids=[],
        )
        assert accuracy == 0.0

    def test_partial_valid(self):
        accuracy = compute_citation_accuracy(
            cited_ids=["ctd_a", "fake_doc"],
            retrieved_ids=["ctd_a"],
            analysis_ids=[],
            reliability_ids=[],
        )
        assert accuracy == 0.5

    def test_analysis_ids_counted(self):
        accuracy = compute_citation_accuracy(
            cited_ids=["analysis_trends"],
            retrieved_ids=[],
            analysis_ids=["analysis_trends"],
            reliability_ids=[],
        )
        assert accuracy == 1.0

    def test_reliability_ids_counted(self):
        accuracy = compute_citation_accuracy(
            cited_ids=["reliability_sst_ctd_validation"],
            retrieved_ids=[],
            analysis_ids=[],
            reliability_ids=["reliability_sst_ctd_validation"],
        )
        assert accuracy == 1.0

    def test_empty_citations(self):
        accuracy = compute_citation_accuracy([], ["ctd_a"], [], [])
        assert accuracy == 0.0


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
class TestComputeSummaryMetrics:
    """Validate metric aggregation from results DataFrame."""

    def _make_results(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"question_id": "q1", "category": "Cat A", "mode": "Baseline",
             "retrieval_precision": 0.8, "source_coverage": 1.0,
             "citation_count": 3, "citation_accuracy": 1.0,
             "context_utilization": 0.0, "latency_seconds": 5.0},
            {"question_id": "q1", "category": "Cat A", "mode": "Full",
             "retrieval_precision": 0.9, "source_coverage": 1.0,
             "citation_count": 5, "citation_accuracy": 0.8,
             "context_utilization": 1.0, "latency_seconds": 7.0},
            {"question_id": "q2", "category": "Cat B", "mode": "Baseline",
             "retrieval_precision": 0.6, "source_coverage": 0.5,
             "citation_count": 2, "citation_accuracy": 1.0,
             "context_utilization": 0.0, "latency_seconds": 4.0},
            {"question_id": "q2", "category": "Cat B", "mode": "Full",
             "retrieval_precision": 0.7, "source_coverage": 1.0,
             "citation_count": 4, "citation_accuracy": 0.75,
             "context_utilization": 0.5, "latency_seconds": 6.0},
        ])

    def test_by_mode_keys(self):
        summaries = compute_summary_metrics(self._make_results())
        assert "by_mode" in summaries
        assert "by_category" in summaries
        assert "by_mode_category" in summaries

    def test_by_mode_values(self):
        summaries = compute_summary_metrics(self._make_results())
        by_mode = summaries["by_mode"]
        assert "Baseline" in by_mode.index
        assert "Full" in by_mode.index
        # Full mode should have higher context utilization than Baseline
        assert by_mode.loc["Full", "context_utilization"] > by_mode.loc["Baseline", "context_utilization"]

    def test_by_category_values(self):
        summaries = compute_summary_metrics(self._make_results())
        by_cat = summaries["by_category"]
        assert "Cat A" in by_cat.index
        assert "Cat B" in by_cat.index
