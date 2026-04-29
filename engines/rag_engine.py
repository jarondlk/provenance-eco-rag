from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from .base import ChatEngine, ChatParams, Message
from rag.bm25_retriever import BM25Retriever, Retrieved


_RAG_INSTRUCTIONS = """
You are a source-grounded assistant.

Rules:
- Use ONLY the provided sources to answer.
- If sources are insufficient, say what is missing and do not guess.
- Cite sources inline using [S1], [S2], ... matching the source ids.
- End with a "Sources" section listing every cited source with title, date, and url.
- If multiple sources conflict, mention the conflict and cite both.
""".strip()


def _format_sources(retrieved: List[Retrieved]) -> str:
    lines = []
    for r in retrieved:
        d = r.doc
        lines.append(
            f"[{d.id}] {d.title} | {d.date} | {d.location} | {d.url}\n"
            f"Excerpt: {r.excerpt}"
        )
    return "\n\n".join(lines)


class RagEngine(ChatEngine):
    def __init__(self, inner: ChatEngine, retriever: BM25Retriever):
        self.inner = inner
        self.retriever = retriever

    def chat(self, messages: List[Message], params: ChatParams, k: int = 5) -> Iterable[str]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        retrieved = self.retriever.search(last_user, k=k)

        sys = (
            params.system_prompt.strip()
            + "\n\n"
            + _RAG_INSTRUCTIONS
            + "\n\nRetrieved sources:\n"
            + (_format_sources(retrieved) if retrieved else "(no sources retrieved)")
        )

        new_params = replace(params, system_prompt=sys)
        yield from self.inner.chat(messages, new_params)
