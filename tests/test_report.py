"""
Tests for evaluation/report.py — report generation and run comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
def _make_results_df(n_questions: int = 3, modes: list | None = None) -> pd.DataFrame:
    """Create a synthetic evaluation results DataFrame."""
    if modes is None:
        modes = ["Baseline", "+Analysis", "+Reliability", "Full"]

    rows = []
    categories = ["Single-source (CTD)", "Dual-source (CTD+SST)", "Reliability-dependent"]
    question_ids = [f"q_{i}" for i in range(n_questions)]

    for i, qid in enumerate(question_ids):
        cat = categories[i % len(categories)]
        for mode in modes:
            rows.append({
                "question_id": qid,
                "category": cat,
                "question": f"Test question {i}?",
                "mode": mode,
                "n_retrieved": 8,
                "retrieved_source_types": "ctd,remote_sensing",
                "retrieval_precision": round(0.5 + np.random.random() * 0.5, 4),
                "source_coverage": round(0.6 + np.random.random() * 0.4, 4),
                "citation_count": np.random.randint(1, 8),
                "citation_accuracy": round(0.4 + np.random.random() * 0.6, 4),
                "analysis_cited": mode in ["+Analysis", "Full"],
                "reliability_cited": mode in ["+Reliability", "Full"],
                "context_utilization": 1.0 if mode == "Full" else (0.5 if mode in ["+Analysis", "+Reliability"] else 0.0),
                "latency_seconds": round(2.0 + np.random.random() * 5, 2),
                "response": f"This is the response for {qid} in {mode} mode.",
                "cited_ids": "ctd_2024-04-O-s1,meta_2024-05-O-s1",
                "error": "",
            })
    return pd.DataFrame(rows)


def _make_run_meta(model: str = "test-model:7b", tag: str | None = None) -> dict:
    """Create a synthetic run metadata dict."""
    return {
        "run_id": "eval_2026-05-21T12-00_test-model",
        "model": model,
        "tag": tag,
        "timestamp": "2026-05-21T12:00:00+09:00",
        "n_questions": 3,
        "n_modes": 4,
        "n_evaluations": 12,
        "top_k": 8,
        "temperature": 0.0,
        "num_ctx": 8192,
        "backend": "pgvector (323 embedded docs)",
        "duration_seconds": 45.3,
        "n_errors": 0,
    }


# ─────────────────────────────────────────────
# Tests: print_summary
# ─────────────────────────────────────────────
class TestPrintSummary:
    def test_runs_without_error(self, capsys):
        from evaluation.report import print_summary
        df = _make_results_df()
        print_summary(df)
        out = capsys.readouterr().out
        assert "Evaluation Summary" in out
        assert "Baseline" in out
        assert "Full" in out

    def test_shows_error_count(self, capsys):
        from evaluation.report import print_summary
        df = _make_results_df()
        df.loc[0, "error"] = "Connection timeout"
        print_summary(df)
        out = capsys.readouterr().out
        assert "Errors" in out


# ─────────────────────────────────────────────
# Tests: generate_report
# ─────────────────────────────────────────────
class TestGenerateReport:
    def test_report_contains_model_name(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta(model="qwen2.5:14b-instruct")
        report = generate_report(df, meta)
        assert "qwen2.5:14b-instruct" in report

    def test_report_has_mode_table(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta()
        report = generate_report(df, meta)
        assert "Performance by Mode" in report
        assert "Baseline" in report
        assert "Full" in report

    def test_report_has_category_table(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta()
        report = generate_report(df, meta)
        assert "Performance by Category" in report

    def test_report_has_baseline_full_delta(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta()
        report = generate_report(df, meta)
        assert "Baseline → Full Delta" in report
        assert "Delta" in report

    def test_report_has_key_findings(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta()
        report = generate_report(df, meta)
        assert "Key Findings" in report
        assert "Average latency" in report

    def test_report_has_per_question_summary(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta()
        report = generate_report(df, meta)
        assert "Per-Question Summary" in report

    def test_report_with_tag(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta(tag="thesis-final")
        report = generate_report(df, meta)
        assert "thesis-final" in report

    def test_report_with_errors(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        df.loc[0, "error"] = "LLM timeout after 180s"
        meta = _make_run_meta()
        meta["n_errors"] = 1
        report = generate_report(df, meta)
        assert "WARNING" in report
        assert "LLM timeout" in report

    def test_report_with_duration(self):
        from evaluation.report import generate_report
        df = _make_results_df()
        meta = _make_run_meta()
        meta["duration_seconds"] = 342.5
        report = generate_report(df, meta)
        assert "342.5s" in report

    def test_report_no_full_mode(self):
        """Report should still work if Full mode is not present."""
        from evaluation.report import generate_report
        df = _make_results_df(modes=["Baseline", "+Analysis"])
        meta = _make_run_meta()
        report = generate_report(df, meta)
        assert "Performance by Mode" in report
        # Should not crash on missing Baseline→Full delta


# ─────────────────────────────────────────────
# Tests: compare_runs
# ─────────────────────────────────────────────
class TestCompareRuns:
    def test_compare_two_runs(self, tmp_path):
        from evaluation.report import compare_runs

        # Create two synthetic run CSVs
        df1 = _make_results_df()
        df2 = _make_results_df()
        # Make df2 slightly different
        df2["retrieval_precision"] = df2["retrieval_precision"] * 0.9

        csv1 = tmp_path / "run_a.csv"
        csv2 = tmp_path / "run_b.csv"
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)

        # Create meta files
        meta1 = _make_run_meta(model="model-a:7b")
        meta2 = _make_run_meta(model="model-b:14b")
        with open(tmp_path / "run_a_meta.json", "w") as f:
            json.dump(meta1, f)
        with open(tmp_path / "run_b_meta.json", "w") as f:
            json.dump(meta2, f)

        report = compare_runs([csv1, csv2])
        assert "Comparison Report" in report
        assert "model-a" in report
        assert "model-b" in report
        assert "Side-by-Side" in report

    def test_compare_needs_two_runs(self):
        from evaluation.report import compare_runs
        report = compare_runs([Path("single.csv")])
        assert "Need at least 2" in report

    def test_compare_without_meta_files(self, tmp_path):
        from evaluation.report import compare_runs

        df1 = _make_results_df()
        df2 = _make_results_df()

        csv1 = tmp_path / "run_x.csv"
        csv2 = tmp_path / "run_y.csv"
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)

        # No meta JSONs — should still work with inferred metadata
        report = compare_runs([csv1, csv2])
        assert "Comparison Report" in report

    def test_compare_has_best_model_per_category(self, tmp_path):
        from evaluation.report import compare_runs

        df1 = _make_results_df()
        df2 = _make_results_df()

        csv1 = tmp_path / "run_1.csv"
        csv2 = tmp_path / "run_2.csv"
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)

        meta1 = _make_run_meta(model="fast-model")
        meta2 = _make_run_meta(model="big-model")
        with open(tmp_path / "run_1_meta.json", "w") as f:
            json.dump(meta1, f)
        with open(tmp_path / "run_2_meta.json", "w") as f:
            json.dump(meta2, f)

        report = compare_runs([csv1, csv2])
        assert "Best Model per Category" in report


# ─────────────────────────────────────────────
# Tests: load_run
# ─────────────────────────────────────────────
class TestLoadRun:
    def test_loads_csv_and_meta(self, tmp_path):
        from evaluation.report import load_run

        df = _make_results_df(n_questions=2, modes=["Baseline"])
        csv_path = tmp_path / "test_run.csv"
        df.to_csv(csv_path, index=False)

        meta = {"model": "test", "run_id": "test_run"}
        with open(tmp_path / "test_run_meta.json", "w") as f:
            json.dump(meta, f)

        loaded_df, loaded_meta = load_run(csv_path)
        assert len(loaded_df) == len(df)
        assert loaded_meta["model"] == "test"

    def test_loads_csv_without_meta(self, tmp_path):
        from evaluation.report import load_run

        df = _make_results_df(n_questions=1, modes=["Full"])
        csv_path = tmp_path / "orphan.csv"
        df.to_csv(csv_path, index=False)

        loaded_df, loaded_meta = load_run(csv_path)
        assert len(loaded_df) == len(df)
        # Should infer metadata from filename
        assert "orphan" in loaded_meta["run_id"]
