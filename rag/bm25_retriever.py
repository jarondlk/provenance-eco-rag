from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from rank_bm25 import BM25Okapi

from .mock_store import SourceDoc

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


@dataclass(frozen=True)
class Retrieved:
    doc: SourceDoc
    score: float
    excerpt: str


class BM25Retriever:
    def __init__(self, docs: List[SourceDoc]):
        self.docs = docs
        self.corpus_tokens = [_tokenize(d.title + " " + d.text) for d in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 5) -> List[Retrieved]:
        q = _tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: List[Retrieved] = []
        for idx, score in ranked:
            d = self.docs[idx]
            excerpt = (d.text[:500] + "…") if len(d.text) > 500 else d.text
            out.append(Retrieved(doc=d, score=float(score), excerpt=excerpt))
        return out
