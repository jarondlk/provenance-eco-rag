from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class SourceDoc:
    id: str
    title: str
    date: str
    location: str
    url: str
    text: str
    lat: Optional[float] = None
    lon: Optional[float] = None


def _to_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def load_jsonl(path: str | Path) -> List[SourceDoc]:
    p = Path(path)
    docs: List[SourceDoc] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj = json.loads(line)
            docs.append(
                SourceDoc(
                    id=obj["id"],
                    title=obj.get("title", ""),
                    date=obj.get("date", ""),
                    location=obj.get("location", ""),
                    url=obj.get("url", ""),
                    text=obj.get("text", ""),
                    lat=_to_float(obj.get("lat")),
                    lon=_to_float(obj.get("lon")),
                )
            )
    return docs
