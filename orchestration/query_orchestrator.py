"""
Query orchestration – the reasoning layer.

Takes a user query, runs hybrid retrieval, expands evidence via
cross-source links, and constructs a provenance-aware LLM prompt.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

import config
from db.connection import get_session

logger = logging.getLogger(__name__)


@dataclass
class EvidenceBundle:
    """All evidence gathered for a single query."""
    query: str
    primary_results: List[dict]           # from hybrid retrieval
    linked_evidence: List[dict]           # expanded via cross-source links
    source_types_found: List[str]
    time_range: Optional[str] = None
    bays_found: List[str] = field(default_factory=list)


def expand_evidence(primary_results: List[dict], max_links: int = 5) -> List[dict]:
    """
    Given primary retrieval results, find cross-source linked documents
    to provide multi-modal context to the LLM.
    """
    if not primary_results:
        return []

    event_ids = [r.get("event_id") for r in primary_results if r.get("event_id")]
    if not event_ids:
        return []

    linked = []
    with get_session() as session:
        placeholders = ", ".join([f":eid_{i}" for i in range(len(event_ids))])
        params = {f"eid_{i}": eid for i, eid in enumerate(event_ids)}

        sql = text(f"""
            SELECT DISTINCT rd.doc_id, rd.source_type, rd.sample_id,
                   rd.event_id, rd.time, rd.bay, rd.station, rd.title, rd.text,
                   cl.link_type, cl.time_delta_days
            FROM cross_source_link cl
            JOIN retrieval_document rd
              ON rd.event_id = cl.target_event_id
            WHERE cl.source_event_id IN ({placeholders})
              AND rd.doc_id NOT IN (
                SELECT doc_id FROM unnest(ARRAY[{', '.join([f":did_{i}" for i in range(len(primary_results))])}]::text[]) AS doc_id
              )
            ORDER BY cl.time_delta_days ASC NULLS LAST
            LIMIT :max_links
        """)
        params["max_links"] = max_links
        for i, r in enumerate(primary_results):
            params[f"did_{i}"] = r.get("doc_id", "")

        try:
            rows = session.execute(sql, params).fetchall()
            for r in rows:
                linked.append({
                    "doc_id": r.doc_id,
                    "source_type": r.source_type,
                    "sample_id": r.sample_id,
                    "event_id": r.event_id,
                    "time": r.time,
                    "bay": r.bay,
                    "title": r.title,
                    "text": r.text,
                    "link_type": r.link_type,
                })
        except Exception as e:
            logger.warning("Evidence expansion failed: %s", e)

    logger.info("Expanded evidence: %d linked documents", len(linked))
    return linked


def build_provenance_prompt(
    query: str,
    evidence: EvidenceBundle,
) -> str:
    """
    Build a provenance-aware system + user prompt for the LLM.

    The prompt instructs the model to:
    1. Only use the provided evidence
    2. Cite sources using [source_id] notation
    3. Distinguish between data types (CTD, metagenome, SST)
    4. Note data gaps or uncertainties
    """
    system_prompt = """You are an expert marine science assistant specializing in the Onagawa Bay region of Japan. You analyze oceanographic, metagenomic, and satellite SST data to provide evidence-based answers.

STRICT RULES:
1. ONLY use the evidence provided below to answer the question.
2. ALWAYS cite your sources using [doc_id] notation (e.g., [ctd_2024-06-O-s1]).
3. Clearly distinguish between data types: CTD measurements, metagenome taxonomic data, and satellite SST observations.
4. If data is insufficient to fully answer the question, explicitly state what data is missing.
5. Report numeric values with appropriate precision and units.
6. When comparing across time or space, note the temporal and spatial resolution of the data.
7. If evidence from different sources conflicts, note the discrepancy.

DATA SOURCES AVAILABLE:
- CTD: Physical/chemical water column profiles (temperature, salinity, dissolved oxygen, chlorophyll, etc.)
- Metagenome: Taxonomic composition from DNA sequencing (Kraken and MetaEuk classifiers)
- SST: Satellite-derived sea surface temperature from Himawari-9

STUDY AREA:
- Onagawa Bay (O): ~38.44°N, 141.45°E
- Ishinomaki Bay (I): ~38.41°N, 141.30°E  
- Matsushima Bay (M): ~38.35°N, 141.06°E"""

    # Format evidence
    evidence_text = "=== PRIMARY EVIDENCE ===\n"
    for i, r in enumerate(evidence.primary_results, 1):
        evidence_text += f"\n[{r.get('doc_id', f'doc_{i}')}] ({r.get('source_type', 'unknown')}, {r.get('time', 'no date')})\n"
        evidence_text += f"{r.get('text', '')}\n"

    if evidence.linked_evidence:
        evidence_text += "\n=== CROSS-SOURCE EVIDENCE (related observations) ===\n"
        for r in evidence.linked_evidence:
            evidence_text += f"\n[{r.get('doc_id', '')}] ({r.get('source_type', 'unknown')}, linked via {r.get('link_type', 'unknown')})\n"
            evidence_text += f"{r.get('text', '')}\n"

    # Compose
    full_prompt = f"{system_prompt}\n\n{evidence_text}\n\n--- USER QUESTION ---\n{query}"

    return full_prompt


def query_pipeline(
    query: str,
    *,
    k: int = 8,
    source_type: Optional[str] = None,
    bay: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    expand: bool = True,
) -> EvidenceBundle:
    """
    Full query pipeline:
      1. Run hybrid retrieval
      2. Optionally expand via cross-source links
      3. Package into EvidenceBundle
    """
    from retrieval.hybrid_retriever import hybrid_search

    # Hybrid retrieval
    results = hybrid_search(
        query,
        k=k,
        source_type=source_type,
        bay=bay,
        time_from=time_from,
        time_to=time_to,
    )

    primary = [
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

    # Expand evidence
    linked = []
    if expand and primary:
        linked = expand_evidence(primary)

    # Collect metadata
    source_types = list(set(r["source_type"] for r in primary))
    bays = list(set(r["bay"] for r in primary if r.get("bay")))
    times = [r["time"] for r in primary if r.get("time")]
    time_range = f"{min(times)} to {max(times)}" if times else None

    bundle = EvidenceBundle(
        query=query,
        primary_results=primary,
        linked_evidence=linked,
        source_types_found=source_types,
        time_range=time_range,
        bays_found=bays,
    )

    logger.info(
        "Query pipeline: %d primary, %d linked, sources=%s",
        len(primary), len(linked), source_types,
    )
    return bundle
