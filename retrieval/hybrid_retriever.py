"""
Hybrid retriever combining:
  1. pgvector cosine similarity  (semantic)
  2. PostgreSQL tsvector FTS     (keyword)
  3. Structured SQL filters      (bay, time range, source_type)

Results are fused using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import text

import config
from db.connection import get_session
from db.vector_store import embed_text

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """One ranked search result."""
    doc_id: str
    source_type: str
    sample_id: Optional[str]
    event_id: Optional[str]
    time: Optional[str]
    bay: Optional[str]
    station: Optional[str]
    title: str
    text: str
    score: float
    rank_sources: Dict[str, int] = field(default_factory=dict)


def hybrid_search(
    query: str,
    *,
    k: int = 10,
    source_type: Optional[str] = None,
    bay: Optional[str] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    vector_weight: float = 0.6,
    fts_weight: float = 0.4,
    rrf_k: int = 60,
) -> List[RetrievalResult]:
    """
    Run a hybrid search combining vector similarity and full-text search.

    Results are merged using Reciprocal Rank Fusion (RRF).
    """
    # Build filter clause
    filters = []
    params: Dict[str, Any] = {"k": k * 2}  # over-fetch for fusion

    if source_type:
        filters.append("source_type = :source_type")
        params["source_type"] = source_type
    if bay:
        filters.append("bay = :bay")
        params["bay"] = bay
    if time_from:
        filters.append("time >= :time_from")
        params["time_from"] = time_from
    if time_to:
        filters.append("time <= :time_to")
        params["time_to"] = time_to

    where = ""
    if filters:
        where = "WHERE " + " AND ".join(filters)

    vector_results: Dict[str, int] = {}
    fts_results: Dict[str, int] = {}
    doc_map: Dict[str, dict] = {}

    with get_session() as session:
        # --- Vector search ---
        try:
            query_emb = embed_text(query)
            emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"
            params["emb"] = emb_str

            vector_where = where
            if vector_where:
                vector_where += " AND embedding IS NOT NULL"
            else:
                vector_where = "WHERE embedding IS NOT NULL"

            sql = text(f"""
                SELECT doc_id, source_type, sample_id, event_id, time,
                       bay, station, title, text
                FROM retrieval_document
                {vector_where}
                ORDER BY embedding <=> :emb
                LIMIT :k
            """)
            rows = session.execute(sql, params).fetchall()
            for rank, r in enumerate(rows):
                vector_results[r.doc_id] = rank + 1
                doc_map[r.doc_id] = {
                    "doc_id": r.doc_id,
                    "source_type": r.source_type,
                    "sample_id": r.sample_id,
                    "event_id": r.event_id,
                    "time": r.time,
                    "bay": r.bay,
                    "station": r.station,
                    "title": r.title,
                    "text": r.text,
                }
        except Exception as e:
            logger.warning("Vector search failed: %s", e)

        # --- Full-text search ---
        try:
            fts_where = where
            if fts_where:
                fts_where += " AND text_tsv @@ plainto_tsquery('english', :query)"
            else:
                fts_where = "WHERE text_tsv @@ plainto_tsquery('english', :query)"

            fts_params = {**params, "query": query}

            sql = text(f"""
                SELECT doc_id, source_type, sample_id, event_id, time,
                       bay, station, title, text,
                       ts_rank_cd(text_tsv, plainto_tsquery('english', :query)) AS fts_rank
                FROM retrieval_document
                {fts_where}
                ORDER BY fts_rank DESC
                LIMIT :k
            """)
            rows = session.execute(sql, fts_params).fetchall()
            for rank, r in enumerate(rows):
                fts_results[r.doc_id] = rank + 1
                if r.doc_id not in doc_map:
                    doc_map[r.doc_id] = {
                        "doc_id": r.doc_id,
                        "source_type": r.source_type,
                        "sample_id": r.sample_id,
                        "event_id": r.event_id,
                        "time": r.time,
                        "bay": r.bay,
                        "station": r.station,
                        "title": r.title,
                        "text": r.text,
                    }
        except Exception as e:
            logger.warning("FTS search failed: %s", e)

    # --- RRF fusion ---
    all_doc_ids = set(vector_results.keys()) | set(fts_results.keys())
    scored: List[RetrievalResult] = []

    for doc_id in all_doc_ids:
        v_rank = vector_results.get(doc_id, k * 2 + 1)
        f_rank = fts_results.get(doc_id, k * 2 + 1)

        rrf_score = (
            vector_weight * (1.0 / (rrf_k + v_rank))
            + fts_weight * (1.0 / (rrf_k + f_rank))
        )

        info = doc_map[doc_id]
        scored.append(RetrievalResult(
            doc_id=doc_id,
            source_type=info["source_type"],
            sample_id=info["sample_id"],
            event_id=info["event_id"],
            time=info["time"],
            bay=info["bay"],
            station=info["station"],
            title=info["title"],
            text=info["text"],
            score=rrf_score,
            rank_sources={"vector": v_rank, "fts": f_rank},
        ))

    scored.sort(key=lambda r: r.score, reverse=True)
    results = scored[:k]

    logger.info(
        "Hybrid search: query=%r  vector=%d  fts=%d  fused=%d",
        query[:60], len(vector_results), len(fts_results), len(results),
    )
    return results
