"""
Evaluation benchmark for the Onagawa Source Chat RAG system.

Defines a benchmark of 15 questions across 5 categories, runs each
through the RAG pipeline with configurable modes (with/without
pre-analysis and reliability context injection), and computes
6 quantitative metrics per evaluation.

Metrics:
    1. Retrieval Precision  — fraction of retrieved docs matching expected source types
    2. Source Coverage       — fraction of expected source types present in retrieved docs
    3. Citation Count        — number of [doc_id] / [analysis_*] / [reliability_*] citations
    4. Citation Accuracy     — fraction of cited IDs that exist in the provided context
    5. Context Utilization   — whether injected analysis/reliability context was cited
    6. Response Latency      — wall-clock seconds from query to final token
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config
from orchestration.unified import build_prompt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Benchmark Questions
# ─────────────────────────────────────────────
@dataclass
class BenchmarkQuestion:
    """One evaluation question with ground truth annotations."""
    id: str
    category: str
    question: str
    expected_source_types: List[str]
    expected_min_citations: int
    requires_analysis: bool = False
    requires_reliability: bool = False


BENCHMARK_QUESTIONS: List[BenchmarkQuestion] = [
    # ── Single-source: CTD (3) ──
    BenchmarkQuestion(
        id="ctd_01",
        category="Single-source (CTD)",
        question="What is the temperature and salinity profile at Onagawa Bay in April 2024?",
        expected_source_types=["ctd"],
        expected_min_citations=1,
    ),
    BenchmarkQuestion(
        id="ctd_02",
        category="Single-source (CTD)",
        question="How does dissolved oxygen vary with depth at Onagawa Bay?",
        expected_source_types=["ctd"],
        expected_min_citations=1,
    ),
    BenchmarkQuestion(
        id="ctd_03",
        category="Single-source (CTD)",
        question="What is the chlorophyll-a concentration at the surface in Onagawa Bay during summer?",
        expected_source_types=["ctd"],
        expected_min_citations=1,
    ),

    # ── Single-source: Metagenome (3) ──
    BenchmarkQuestion(
        id="meta_01",
        category="Single-source (Metagenome)",
        question="What are the dominant microbial genera found in Ishinomaki Bay?",
        expected_source_types=["metagenome"],
        expected_min_citations=1,
    ),
    BenchmarkQuestion(
        id="meta_02",
        category="Single-source (Metagenome)",
        question="How does microbial community composition differ between Kraken and MetaEuk classifiers?",
        expected_source_types=["metagenome"],
        expected_min_citations=1,
    ),
    BenchmarkQuestion(
        id="meta_03",
        category="Single-source (Metagenome)",
        question="What metagenome samples are available from Matsushima Bay?",
        expected_source_types=["metagenome"],
        expected_min_citations=1,
    ),

    # ── Dual-source: CTD + SST (3) ──
    BenchmarkQuestion(
        id="dual_01",
        category="Dual-source (CTD+SST)",
        question="How does satellite SST compare to CTD surface temperature measurements at Onagawa Bay?",
        expected_source_types=["ctd", "remote_sensing"],
        expected_min_citations=2,
    ),
    BenchmarkQuestion(
        id="dual_02",
        category="Dual-source (CTD+SST)",
        question="What is the seasonal temperature trend from both CTD profiles and satellite observations?",
        expected_source_types=["ctd", "remote_sensing"],
        expected_min_citations=2,
    ),
    BenchmarkQuestion(
        id="dual_03",
        category="Dual-source (CTD+SST)",
        question="Compare the surface water temperature from in-situ CTD measurements and remote sensing SST data.",
        expected_source_types=["ctd", "remote_sensing"],
        expected_min_citations=2,
    ),

    # ── Analysis-dependent (3) ──
    BenchmarkQuestion(
        id="analysis_01",
        category="Analysis-dependent",
        question="What taxa show significant correlation with temperature changes across all bays?",
        expected_source_types=["metagenome", "ctd"],
        expected_min_citations=2,
        requires_analysis=True,
    ),
    BenchmarkQuestion(
        id="analysis_02",
        category="Analysis-dependent",
        question="How does microbial diversity vary seasonally between Onagawa and Ishinomaki bays?",
        expected_source_types=["metagenome"],
        expected_min_citations=2,
        requires_analysis=True,
    ),
    BenchmarkQuestion(
        id="analysis_03",
        category="Analysis-dependent",
        question="Are there co-occurrence patterns between dinoflagellates and diatoms in the ecosystem?",
        expected_source_types=["metagenome"],
        expected_min_citations=1,
        requires_analysis=True,
    ),

    # ── Reliability-dependent (3) ──
    BenchmarkQuestion(
        id="reliability_01",
        category="Reliability-dependent",
        question="How reliable is our SST data when validated against in-situ CTD measurements?",
        expected_source_types=["ctd", "remote_sensing"],
        expected_min_citations=2,
        requires_reliability=True,
    ),
    BenchmarkQuestion(
        id="reliability_02",
        category="Reliability-dependent",
        question="Are there any anomalous diversity measurements that deviate from environmental predictions?",
        expected_source_types=["metagenome"],
        expected_min_citations=1,
        requires_reliability=True,
    ),
    BenchmarkQuestion(
        id="reliability_03",
        category="Reliability-dependent",
        question="What is the confidence level of cross-source corroboration for our observations?",
        expected_source_types=["ctd", "metagenome"],
        expected_min_citations=1,
        requires_reliability=True,
    ),
]


# ─────────────────────────────────────────────
# Evaluation Modes
# ─────────────────────────────────────────────
@dataclass
class EvalMode:
    """One evaluation configuration."""
    name: str
    inject_analysis: bool
    inject_reliability: bool


EVAL_MODES: List[EvalMode] = [
    EvalMode("Baseline", False, False),
    EvalMode("+Analysis", True, False),
    EvalMode("+Reliability", False, True),
    EvalMode("Full", True, True),
]


# ─────────────────────────────────────────────
# Evaluation Result
# ─────────────────────────────────────────────
@dataclass
class EvalResult:
    """Metrics from one evaluation run."""
    question_id: str
    category: str
    question: str
    mode: str
    # Retrieval metrics
    n_retrieved: int = 0
    retrieved_source_types: str = ""
    retrieval_precision: float = 0.0
    source_coverage: float = 0.0
    # Citation metrics
    citation_count: int = 0
    citation_accuracy: float = 0.0
    analysis_cited: bool = False
    reliability_cited: bool = False
    context_utilization: float = 0.0
    # Performance
    latency_seconds: float = 0.0
    # Raw outputs
    response: str = ""
    cited_ids: str = ""
    error: str = ""


# ─────────────────────────────────────────────
# Citation Extraction
# ─────────────────────────────────────────────
_CITATION_RE = re.compile(
    r"\[("
    r"(?:ctd|meta|sst|doc)_[\w\-]+"             # doc citations
    r"|analysis_[\w]+"                            # analysis citations
    r"|reliability_[\w]+"                         # reliability citations
    r")\]",
    re.IGNORECASE,
)


def extract_citations(text: str) -> List[str]:
    """Extract all [doc_id], [analysis_*], [reliability_*] citations from text."""
    return _CITATION_RE.findall(text)


def compute_citation_accuracy(
    cited_ids: List[str],
    retrieved_ids: List[str],
    analysis_ids: List[str],
    reliability_ids: List[str],
) -> float:
    """Fraction of cited IDs that actually exist in the provided context."""
    if not cited_ids:
        return 0.0
    all_valid = set(retrieved_ids) | set(analysis_ids) | set(reliability_ids)
    valid = sum(1 for c in cited_ids if c in all_valid)
    return valid / len(cited_ids)


# ─────────────────────────────────────────────
# Single Evaluation Runner
# ─────────────────────────────────────────────
def run_single_evaluation(
    question: BenchmarkQuestion,
    mode: EvalMode,
    *,
    model: str = "qwen2.5:14b-instruct",
    ollama_url: str = "http://localhost:11434",
    top_k: int = 8,
    temperature: float = 0.0,
    num_ctx: int = 8192,
) -> EvalResult:
    """
    Run one question through the RAG pipeline and measure metrics.

    Uses orchestration.unified.retrieve() which auto-detects
    PostgreSQL vs local backend. Temperature defaults to 0.0
    for deterministic evaluation.
    """
    from orchestration.unified import retrieve

    result = EvalResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        mode=mode.name,
    )

    try:
        # 1. Retrieve
        retrieved = retrieve(question.question, k=top_k)

        result.n_retrieved = len(retrieved)

        # Source types in retrieved docs
        ret_types = set()
        ret_ids = []
        for r in retrieved:
            st = r.get("source_type", "")
            if st:
                ret_types.add(st)
            doc_id = r.get("doc_id", r.get("id", ""))
            if doc_id:
                ret_ids.append(doc_id)

        result.retrieved_source_types = ",".join(sorted(ret_types))

        # Retrieval precision: fraction matching expected types
        expected = set(question.expected_source_types)
        if retrieved:
            matching = sum(
                1 for r in retrieved
                if r.get("source_type", "") in expected
            )
            result.retrieval_precision = round(matching / len(retrieved), 4)
        else:
            result.retrieval_precision = 0.0

        # Source coverage: fraction of expected types present
        if expected:
            result.source_coverage = round(
                len(ret_types & expected) / len(expected), 4
            )
        else:
            result.source_coverage = 1.0

        # 2. Build prompt with mode-specific injection
        prompt_text = build_prompt(
            question.question,
            retrieved,
            inject_analysis=mode.inject_analysis,
            inject_reliability=mode.inject_reliability,
        )

        # Collect available context IDs for citation accuracy
        analysis_ids = []
        reliability_ids = []
        if mode.inject_analysis:
            adoc_path = config.ANALYSIS_DIR / "analysis_documents.jsonl"
            if adoc_path.exists():
                with open(adoc_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            analysis_ids.append(d.get("id", ""))

        if mode.inject_reliability:
            rdoc_path = config.RELIABILITY_DIR / "reliability_documents.jsonl"
            if rdoc_path.exists():
                with open(rdoc_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            reliability_ids.append(d.get("id", ""))

        # 3. Call LLM
        t0 = time.time()
        full_response = ""
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt_text}],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                },
            },
            stream=True, timeout=180,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                full_response += token

        result.latency_seconds = round(time.time() - t0, 2)
        result.response = full_response

        # 4. Extract citations
        cited = extract_citations(full_response)
        result.citation_count = len(cited)
        result.cited_ids = ",".join(cited)

        # Citation accuracy
        result.citation_accuracy = round(
            compute_citation_accuracy(cited, ret_ids, analysis_ids, reliability_ids),
            4,
        )

        # Context utilization
        result.analysis_cited = any(c.startswith("analysis_") for c in cited)
        result.reliability_cited = any(c.startswith("reliability_") for c in cited)

        # Context utilization score:
        # - If analysis was injected, did the model cite it? (0 or 1)
        # - If reliability was injected, did the model cite it? (0 or 1)
        # - Average of applicable checks
        util_checks = []
        if mode.inject_analysis and analysis_ids:
            util_checks.append(1.0 if result.analysis_cited else 0.0)
        if mode.inject_reliability and reliability_ids:
            util_checks.append(1.0 if result.reliability_cited else 0.0)
        result.context_utilization = round(
            float(np.mean(util_checks)) if util_checks else 0.0, 4
        )

    except Exception as e:
        result.error = str(e)
        logger.error("Evaluation error for %s/%s: %s", question.id, mode.name, e)

    return result


# ─────────────────────────────────────────────
# Full Benchmark Runner
# ─────────────────────────────────────────────
def run_full_benchmark(
    *,
    model: str = "qwen2.5:14b-instruct",
    ollama_url: str = "http://localhost:11434",
    top_k: int = 8,
    temperature: float = 0.0,
    num_ctx: int = 8192,
    questions: Optional[List[BenchmarkQuestion]] = None,
    modes: Optional[List[EvalMode]] = None,
    progress_callback: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Run all benchmark questions through all evaluation modes.

    Args:
        model: Ollama model name.
        ollama_url: Ollama API base URL.
        top_k: Number of documents to retrieve.
        temperature: LLM temperature (0.0 for deterministic).
        num_ctx: Context window size.
        questions: Override default benchmark questions.
        modes: Override default evaluation modes.
        progress_callback: Optional callable(current, total) for progress updates.

    Returns:
        DataFrame with one row per (question, mode) evaluation.
    """
    qs = questions or BENCHMARK_QUESTIONS
    ms = modes or EVAL_MODES
    total = len(qs) * len(ms)

    results = []
    for i, q in enumerate(qs):
        for j, m in enumerate(ms):
            idx = i * len(ms) + j
            if progress_callback:
                progress_callback(idx, total)

            logger.info("Evaluating [%d/%d] %s / %s", idx + 1, total, q.id, m.name)
            r = run_single_evaluation(
                q, m,
                model=model,
                ollama_url=ollama_url,
                top_k=top_k,
                temperature=temperature,
                num_ctx=num_ctx,
            )
            results.append(asdict(r))

    if progress_callback:
        progress_callback(total, total)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Summary Metrics
# ─────────────────────────────────────────────
def compute_summary_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute aggregate metrics from benchmark results.

    Returns dict with keys:
        'by_mode': average metrics per mode
        'by_category': average metrics per category
        'by_mode_category': average metrics per (mode, category)
    """
    metric_cols = [
        "retrieval_precision", "source_coverage",
        "citation_count", "citation_accuracy",
        "context_utilization", "latency_seconds",
    ]

    by_mode = df.groupby("mode")[metric_cols].mean().round(4)
    by_category = df.groupby("category")[metric_cols].mean().round(4)
    by_mode_category = df.groupby(["mode", "category"])[metric_cols].mean().round(4)

    return {
        "by_mode": by_mode,
        "by_category": by_category,
        "by_mode_category": by_mode_category,
    }
