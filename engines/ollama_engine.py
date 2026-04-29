from __future__ import annotations

import json
from typing import Iterable, List, Dict, Any

import requests

from .base import ChatEngine, ChatParams, Message


class OllamaEngine(ChatEngine):
    def __init__(self, base_url: str, timeout_s: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        return sorted([m["name"] for m in data.get("models", [])])

    def chat(self, messages: List[Message], params: ChatParams) -> Iterable[str]:
        url = f"{self.base_url}/api/chat"

        trimmed = messages[-params.keep_last_n_messages :] if params.keep_last_n_messages > 0 else messages

        payload: Dict[str, Any] = {
            "model": params.model,
            "stream": params.stream,
            "messages": [{"role": "system", "content": params.system_prompt}] + trimmed,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "repeat_penalty": params.repeat_penalty,
                "num_predict": params.num_predict,
                "num_ctx": params.num_ctx,
            },
        }
        if params.seed is not None:
            payload["options"]["seed"] = params.seed

        with requests.post(url, json=payload, stream=True, timeout=self.timeout_s) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = obj.get("message") or {}
                chunk = (msg.get("content") if isinstance(msg, dict) else "") or ""
                if chunk:
                    yield chunk

                if obj.get("done") is True:
                    break
