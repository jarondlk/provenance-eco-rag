"""
Local retriever (no PostgreSQL required).

Loads retrieval_documents.jsonl and uses:
  - BM25 for keyword search
  - Ollama for vector search (in-memory numpy cosine)
  - RRF fusion

This is the fallback when PostgreSQL is not available.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


# =====================================================================
# BM25 (adapted from existing engines/rag_engine.py)
# =====================================================================
class BM25:
    """Simple in-memory BM25 scorer."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._df: Counter = Counter()
        self._N: int = 0

    def fit(self, documents: List[str]) -> None:
        self._corpus = [self._tokenize(d) for d in documents]
        self._N = len(self._corpus)
        self._doc_len = [len(d) for d in self._corpus]
        self._avgdl = sum(self._doc_len) / max(self._N, 1)
        self._df = Counter()
        for doc in self._corpus:
            for term in set(doc):
                self._df[term] += 1

    def score(self, query: str) -> List[float]:
        q_tokens = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self._corpus):
            s = 0.0
            dl = self._doc_len[i]
            tf_map = Counter(doc)
            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                df = self._df.get(qt, 0)
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += idf * num / den
            scores.append(s)
        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


# =====================================================================
# Local retriever
# =====================================================================
class LocalRetriever:
    """
    In-memory hybrid retriever using BM25 + optional Ollama embeddings.
    """

    def __init__(self) -> None:
        self.documents: List[dict] = []
        self.bm25 = BM25()
        self._embeddings: Optional[np.ndarray] = None
        self._embed_available: bool = False

    def load(self, jsonl_path: Path | None = None) -> None:
        """Load documents from JSONL."""
        if jsonl_path is None:
            jsonl_path = config.SERVING_DIR / "retrieval_documents.jsonl"

        if not jsonl_path.exists():
            logger.warning("Document file not found: %s", jsonl_path)
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.documents = [json.loads(line) for line in f if line.strip()]

        # Fit BM25
        texts = [d.get("text", "") for d in self.documents]
        self.bm25.fit(texts)

        logger.info("Loaded %d documents for local retrieval", len(self.documents))

    def ensure_embeddings(self) -> bool:
        """
        Try to compute embeddings via Ollama.
        Returns True if embeddings are available.
        """
        if self._embeddings is not None:
            return True

        # Try loading cached embeddings
        cache_path = config.SERVING_DIR / "retrieval_embeddings.npy"
        if cache_path.exists():
            self._embeddings = np.load(str(cache_path))
            if len(self._embeddings) == len(self.documents):
                self._embed_available = True
                logger.info("Loaded cached embeddings: %s", self._embeddings.shape)
                return True

        # Try computing via Ollama
        try:
            from db.vector_store import embed_batch

            texts = [d.get("text", "") for d in self.documents]
            batch_size = 32
            all_embs = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embs = embed_batch(batch)
                all_embs.extend(embs)
                logger.info("  Embedded %d/%d", min(i + batch_size, len(texts)), len(texts))

            self._embeddings = np.array(all_embs, dtype="float32")
            np.save(str(cache_path), self._embeddings)
            self._embed_available = True
            logger.info("Computed and cached embeddings: %s", self._embeddings.shape)
            return True

        except Exception as e:
            logger.warning("Could not compute embeddings: %s", e)
            return False

    def search(
        self,
        query: str,
        k: int = 8,
        source_type: Optional[str] = None,
        bay: Optional[str] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
    ) -> List[dict]:
        """
        Hybrid search: BM25 + (optional) vector, fused with RRF.
        """
        if not self.documents:
            return []

        # Apply filters
        valid_indices = []
        for i, doc in enumerate(self.documents):
            if source_type and doc.get("source_type") != source_type:
                continue
            if bay and doc.get("bay") != bay:
                continue
            if time_from and (doc.get("time") or "") < time_from:
                continue
            if time_to and (doc.get("time") or "") > time_to:
                continue
            valid_indices.append(i)

        if not valid_indices:
            return []

        # BM25 scores
        all_bm25 = self.bm25.score(query)
        bm25_scored = [(i, all_bm25[i]) for i in valid_indices]
        bm25_scored.sort(key=lambda x: x[1], reverse=True)
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_scored)}

        # Vector scores
        vector_ranks: Dict[int, int] = {}
        if self._embed_available and self._embeddings is not None:
            try:
                from db.vector_store import embed_text
                q_emb = np.array(embed_text(query), dtype="float32")
                valid_embs = self._embeddings[valid_indices]
                # Cosine similarity
                norms = np.linalg.norm(valid_embs, axis=1) * np.linalg.norm(q_emb)
                norms[norms == 0] = 1e-10
                sims = valid_embs @ q_emb / norms
                sim_order = np.argsort(-sims)
                for rank, pos in enumerate(sim_order):
                    vector_ranks[valid_indices[pos]] = rank + 1
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        # RRF fusion
        rrf_k = 60
        v_weight = 0.6 if vector_ranks else 0.0
        b_weight = 1.0 - v_weight

        scored = []
        for idx in valid_indices:
            br = bm25_ranks.get(idx, len(valid_indices) + 1)
            vr = vector_ranks.get(idx, len(valid_indices) + 1)
            score = b_weight / (rrf_k + br) + v_weight / (rrf_k + vr)
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        results = []
        for idx, score in top_k:
            doc = self.documents[idx].copy()
            doc["score"] = score
            results.append(doc)

        return results


# Global singleton
_retriever: Optional[LocalRetriever] = None


def get_local_retriever() -> LocalRetriever:
    """Get or create the global local retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = LocalRetriever()
        _retriever.load()
        _retriever.ensure_embeddings()
    return _retriever
