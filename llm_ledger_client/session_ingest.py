"""Reads an agent-session transcript (Pi session JSONL, or `pi --mode json` streamed stdout) and posts
one LedgerClient.log_call() per assistant message, after the fact -- for work that made no live LLM call
through this library (Pi talks to DeepSeek's official API directly, not through litellm), so it left no
usage trail until its transcript is read once it's finished. Re-running on the same transcript posts
nothing new, because every record carries `external_id` and the ledger service enforces uniqueness per
run (see django-llm-ledger's LLMCall.Meta.constraints).

Ported faithfully from agentic_doc.agent_usage.PiAdapter (agentic-doc-treatement, 2026-10) -- the real
gotchas handled there (message vs. message_end events for the two session formats, external_id
construction, subscription-provider detection, latency from the parent entry) are preserved here, not
reinvented. Only the Pi adapter is ported in this first release; add further `source=` adapters here as
other projects' agent tools need them (same pattern: implement `read(path) -> Iterator[UsageRecord]`,
register in ADAPTERS)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger("llm_ledger_client")

_PI_SUBSCRIPTION_PROVIDERS = {"github-copilot"}


@dataclass
class UsageRecord:
    """One assistant message worth of usage, normalized across sources."""

    source: str
    external_id: str
    timestamp: datetime
    provider: str
    model: str  # "<provider>/<model>", the ModelPricing key
    billing: str  # "api" | "subscription"
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    latency_ms: int | None = None
    tool_calls: int = 0
    input_text: str = ""
    output_text: str = ""
    reported_cost_usd: float | None = None  # what the source itself claims; never used as the real cost


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                _logger.warning("%s line %d is not valid JSON; skipped", path, lineno)


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"naive timestamp in transcript: {value!r}")
    return dt.astimezone(timezone.utc)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class PiAdapter:
    """Pi session JSONL (~/.pi/agent/sessions/.../<ts>_<id>.jsonl, or --session-dir) and the
    `pi --mode json` stdout stream (pi >= 1.0 streams the session to stdout instead of writing a session
    file; message_end events there carry each message's final usage). Each assistant message carries
    usage {input, output, cacheRead, cacheWrite, reasoning, cost{...}}; `input` is already the uncached
    part and `output` already includes reasoning tokens. Pi's own cost figure is the peak list rate at
    every hour and lacks a cache-read price for some models, so it is kept as evidence only
    (`reported_cost_usd`), never posted as the real cost."""

    source = "pi"

    def read(self, path: Path, **_ignored) -> Iterator[UsageRecord]:
        session_id = ""
        entry_ts: dict[str, datetime] = {}
        pending_input: list = []
        for entry in _iter_jsonl(path):
            kind = entry.get("type")
            if kind == "session":
                session_id = entry.get("id", "")
                continue
            message = entry.get("message") or {}
            if kind == "message":
                entry_id = entry.get("id", "")
                try:
                    when = _parse_ts(entry["timestamp"])
                except (KeyError, ValueError):
                    _logger.warning("%s: message entry %s has no usable timestamp; skipped", path, entry_id)
                    continue
                parent_id = entry.get("parentId")
            elif kind == "message_end":
                if not message:
                    continue
                entry_id = message.get("responseId") or ""
                timestamp_ms = message.get("timestamp")
                if message.get("role") == "assistant":
                    if not isinstance(timestamp_ms, (int, float)):
                        _logger.warning("%s: message_end entry %s has no ms timestamp; skipped", path, entry_id)
                        continue
                    when = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
                else:  # user/system/toolResult events only feed pending_input: no timestamp needed
                    when = None
                parent_id = None
            else:
                continue
            if when is not None:
                entry_ts[entry_id] = when
            if message.get("role") != "assistant":
                pending_input.append({"role": message.get("role"), "content": message.get("content")})
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                _logger.warning("%s: assistant entry %s has no usage record; skipped", path, entry_id)
                pending_input = []
                continue
            provider = message.get("provider", "")
            model_id = message.get("model", "")
            response_id = message.get("responseId")
            external_id = f"pi:{response_id}:{message.get('timestamp')}" if response_id else f"pi:{session_id}:{entry_id}"
            parent = entry_ts.get(parent_id or "") if parent_id else None
            latency = int((when - parent).total_seconds() * 1000) if parent and when >= parent else None
            content = message.get("content") or []
            cost = usage.get("cost")
            reported = cost.get("total") if isinstance(cost, dict) else cost
            billing = "subscription" if provider in _PI_SUBSCRIPTION_PROVIDERS else "api"
            yield UsageRecord(
                source=self.source,
                external_id=external_id,
                timestamp=when,
                provider=provider,
                model=f"{provider}/{model_id}",
                billing=billing,
                prompt_tokens=int(usage.get("input") or 0),
                completion_tokens=int(usage.get("output") or 0),
                cache_read_tokens=int(usage.get("cacheRead") or 0),
                cache_creation_tokens=int(usage.get("cacheWrite") or 0),
                latency_ms=latency,
                tool_calls=sum(1 for block in content if isinstance(block, dict) and block.get("type") == "toolCall"),
                input_text=_dump(pending_input),
                output_text=_dump(content),
                reported_cost_usd=float(reported) if isinstance(reported, (int, float)) else None,
            )
            pending_input = []


ADAPTERS = {"pi": PiAdapter}


def read_usage(source: str, path: str | Path, **filters) -> list[UsageRecord]:
    if source not in ADAPTERS:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(ADAPTERS)}")
    return list(ADAPTERS[source]().read(Path(path), **filters))


def import_session(client, run_pk: int, source: str, path: str | Path, *, call_purpose: str = "") -> int:
    """Reads `path` with the given adapter and posts one log_call() per record. Returns the count
    posted (not the count read -- a record the ledger already has, by external_id, is a no-op there,
    not an error here)."""
    records = read_usage(source, path)
    for record in records:
        client.log_call(
            run_pk,
            provider=record.provider,
            model=record.model,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            cache_read_tokens=record.cache_read_tokens,
            cache_creation_tokens=record.cache_creation_tokens,
            billing=record.billing,
            external_id=record.external_id,
            call_purpose=call_purpose,
            status="success",
            latency_ms=record.latency_ms,
            input_text=record.input_text,
            output_text=record.output_text,
            timestamp=record.timestamp.isoformat(),
        )
    return len(records)
