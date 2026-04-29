"""
Unified query orchestrator.

Auto-detects whether PostgreSQL is available and falls back to the
local retriever if not.  Either way, the same provenance-aware prompt
is built for the LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


def _pg_available() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def retrieve(
    query: str,
    *,
    k: int = 8,
    source_type: Optional[str] = None,
    bay: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve relevant documents using the best available backend.
    """
    if _pg_available():
        logger.info("Using PostgreSQL hybrid retriever")
        from retrieval.hybrid_retriever import hybrid_search
        results = hybrid_search(
            query, k=k, source_type=source_type, bay=bay,
            time_from=time_from, time_to=time_to,
        )
        return [
            {
                "doc_id": r.doc_id,
                "source_type": r.source_type,
                "sample_id": r.sample_id,
                "event_id": r.event_id,
                "time": r.time,
                "bay": r.bay,
                "station": r.station,
                "title": r.title,
                "text": r.text,
                "score": r.score,
            }
            for r in results
        ]
    else:
        logger.info("PostgreSQL not available – using local retriever")
        from retrieval.local_retriever import get_local_retriever
        retriever = get_local_retriever()
        return retriever.search(
            query, k=k, source_type=source_type, bay=bay,
            time_from=time_from, time_to=time_to,
        )


def build_prompt(query: str, results: List[dict]) -> str:
    """
    Build the provenance-aware system prompt with evidence.
    """
    system = """You are an expert marine science assistant for the Onagawa Bay monitoring programme (Japan).
You analyze CTD water profiles, metagenome taxonomic data, and satellite SST observations.

RULES:
1. ONLY use the evidence provided below. Do not hallucinate.
2. ALWAYS cite sources using [doc_id] notation.
3. Distinguish data types: CTD measurements, metagenome taxonomy, satellite SST.
4. State data gaps explicitly. Report values with units.
5. When comparing across time/space, note the resolution.

STUDY SITES:
• Onagawa Bay (O) ≈ 38.44°N 141.45°E
• Ishinomaki Bay (I) ≈ 38.41°N 141.30°E
• Matsushima Bay (M) ≈ 38.35°N 141.06°E"""

    evidence_text = "\n=== EVIDENCE ===\n"
    for r in results:
        doc_id = r.get("doc_id") or r.get("id", "unknown")
        src = r.get("source_type", "unknown")
        t = r.get("time") or r.get("date", "")
        text = r.get("text", "")
        evidence_text += f"\n[{doc_id}] ({src}, {t})\n{text}\n"

    return f"{system}\n{evidence_text}\n\nUSER QUESTION: {query}"


def ask(
    query: str,
    *,
    k: int = 8,
    source_type: Optional[str] = None,
    bay: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → build prompt → call LLM → return answer + sources.
    """
    import requests

    # Retrieve
    results = retrieve(
        query, k=k, source_type=source_type, bay=bay,
        time_from=time_from, time_to=time_to,
    )

    # Build prompt
    prompt = build_prompt(query, results)

    # Call Ollama
    model = model or config.CHAT_MODEL
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"]
    except Exception as e:
        answer = f"LLM error: {e}"

    return {
        "query": query,
        "answer": answer,
        "sources": results,
        "model": model,
        "n_sources": len(results),
    }
