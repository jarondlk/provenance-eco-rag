from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Optional, Any


@dataclass
class ChatParams:
    model: str
    system_prompt: str
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    num_predict: int
    num_ctx: int
    seed: Optional[int]
    stream: bool
    keep_last_n_messages: int


Message = Dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


class ChatEngine(Protocol):
    def chat(self, messages: List[Message], params: ChatParams) -> Iterable[str]:
        ...
