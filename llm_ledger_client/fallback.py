"""Local, append-only fallback queue for when the ledger service is unreachable -- same resilience
discipline as this family of projects' own standing rule: never lose a real, already-computed record
because a network call failed. One JSON object per line, flushed on every append (never held only in
memory)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def queue_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def enqueue(path: str | Path, kind: str, payload: dict[str, Any]) -> None:
    """Appends one {"kind": "run"|"call", "payload": {...}} record, flushed immediately."""
    p = queue_path(path)
    with _lock, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False) + "\n")
        f.flush()


def read_all(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def clear(path: str | Path) -> None:
    p = Path(path)
    if p.is_file():
        p.unlink()
