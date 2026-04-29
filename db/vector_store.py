"""
Vector store – embed documents via Ollama and search via pgvector.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text

import config
from .connection import get_session

logger = logging.getLogger(__name__)


def embed_text(text_input: str, model: str = None) -> List[float]:
    """
    Get embedding vector from Ollama for a single text.
    """
    model = model or config.EMBEDDING_MODEL
    url = f"{config.OLLAMA_BASE_URL}/api/embed"
    payload = {"model": model, "input": text_input}

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Ollama returns {"embeddings": [[...]]}
    embeddings = data.get("embeddings", [])
    if embeddings:
        return embeddings[0]

    raise ValueError(f"No embedding returned for model {model}")


def embed_batch(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Embed a batch of texts.  Ollama supports batch via the 'input' field.
    Falls back to sequential if batch fails.
    """
    model = model or config.EMBEDDING_MODEL
    url = f"{config.OLLAMA_BASE_URL}/api/embed"

    try:
        payload = {"model": model, "input": texts}
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings", [])
        if len(embeddings) == len(texts):
            return embeddings
    except Exception as e:
        logger.warning("Batch embedding failed, falling back to sequential: %s", e)

    # Sequential fallback
    return [embed_text(t, model) for t in texts]


def update_document_embeddings(batch_size: int = 32) -> int:
    """
    Find retrieval documents without embeddings and compute them.
    Returns the number of documents updated.
    """
    from .models import RetrievalDocument

    count = 0

    with get_session() as session:
        docs = (
            session.query(RetrievalDocument)
            .filter(RetrievalDocument.embedding.is_(None))
            .all()
        )

        if not docs:
            logger.info("All documents already have embeddings")
            return 0

        logger.info("Embedding %d documents...", len(docs))

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.text for d in batch]

            try:
                embeddings = embed_batch(texts)
                for doc, emb in zip(batch, embeddings):
                    doc.embedding = emb
                    count += 1
                session.flush()
                logger.info("  Embedded batch %d-%d", i, i + len(batch))
            except Exception as e:
                logger.error("  Failed batch %d-%d: %s", i, i + len(batch), e)

    logger.info("Updated %d document embeddings", count)
    return count


def search_similar(
    query: str,
    k: int = 5,
    source_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for documents similar to *query* using pgvector cosine distance.
    """
    query_emb = embed_text(query)

    with get_session() as session:
        emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"

        where_clause = ""
        if source_type:
            where_clause = f"AND source_type = '{source_type}'"

        sql = text(f"""
            SELECT doc_id, source_type, sample_id, event_id, time,
                   lat, lon, bay, station, title, text,
                   embedding <=> :emb AS distance
            FROM retrieval_document
            WHERE embedding IS NOT NULL {where_clause}
            ORDER BY embedding <=> :emb
            LIMIT :k
        """)

        rows = session.execute(sql, {"emb": emb_str, "k": k}).fetchall()

    return [
        {
            "doc_id": r.doc_id,
            "source_type": r.source_type,
            "sample_id": r.sample_id,
            "event_id": r.event_id,
            "time": r.time,
            "lat": r.lat,
            "lon": r.lon,
            "bay": r.bay,
            "station": r.station,
            "title": r.title,
            "text": r.text,
            "distance": float(r.distance),
            "score": 1.0 - float(r.distance),  # cosine similarity
        }
        for r in rows
    ]
