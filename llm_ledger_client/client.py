"""Thin HTTP client for django-llm-ledger. Never trusts its own cost estimate as the real one -- it
posts raw token counts (and, optionally, a provider-computed estimate kept in a SEPARATE field) and lets
the ledger service compute cost_usd server-side from its own ModelPricing table, same design as the
project this library was extracted from.

Resilience: every write goes to the local fallback queue FIRST (flushed immediately, per
llm_ledger_client.fallback), then attempted against the live service. A successful live write does not
remove the queued copy immediately -- call `flush_fallback()` periodically (or at process exit) to
resend everything queued and clear what the service confirms it received. This means a crash between
"queued" and "sent" never loses the record; the cost is that `flush_fallback()` may resend an already-
received record on a crash mid-flush, which is why the ledger's own idempotency (`external_id` unique
per run) matters for anything replayed from a transcript ingester."""

from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any

import requests

from . import fallback

_logger = logging.getLogger("llm_ledger_client")

DEFAULT_FALLBACK_PATH = Path.home() / ".llm_ledger_client" / "fallback_queue.jsonl"


class LedgerClient:
    def __init__(
        self,
        base_url: str | None,
        *,
        project: str = "",
        fallback_path: str | Path = DEFAULT_FALLBACK_PATH,
        timeout: float = 5.0,
    ):
        """`base_url` is e.g. "http://localhost:8000/api" for an embedded django-llm-ledger mount, or
        the standalone service's own URL. `base_url=None` means "never attempt the network, queue only"
        -- useful for an offline dev run where you just want the fallback file populated correctly."""
        self.base_url = base_url.rstrip("/") if base_url else None
        self.project = project
        self.fallback_path = Path(fallback_path)
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict | None:
        if self.base_url is None:
            return None
        try:
            resp = requests.post(f"{self.base_url}/{path}/", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            _logger.warning("llm_ledger_client: %s POST failed, queued only: %s", path, exc)
            return None

    def create_run(
        self,
        run_id: str,
        *,
        label: str = "",
        purpose: str = "development",
        status: str = "running",
        triggered_by: str = "",
        **extra: Any,
    ) -> dict | None:
        payload = {
            "run_id": run_id, "project": self.project, "label": label, "purpose": purpose,
            "status": status, "triggered_by": triggered_by or "agent", **extra,
        }
        fallback.enqueue(self.fallback_path, "run", payload)
        return self._post("runs", payload)

    def complete_run(self, run_pk: int, *, status: str = "success", result_summary: dict | None = None) -> dict | None:
        if self.base_url is None:
            return None
        try:
            resp = requests.patch(
                f"{self.base_url}/runs/{run_pk}/",
                json={"status": status, "completed_at": datetime.now(dt_timezone.utc).isoformat(), "result_summary": result_summary or {}},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            _logger.warning("llm_ledger_client: complete_run failed: %s", exc)
            return None

    def log_call(
        self,
        run_pk: int,
        *,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        billing: str = "api",
        api_equivalent_cost_usd: float | None = None,
        external_id: str = "",
        call_purpose: str = "",
        status: str = "success",
        **extra: Any,
    ) -> dict | None:
        """Never pass a cost_usd here -- the server computes it. `external_id` makes this call
        idempotent (unique per run) when ingested from a transcript that might be replayed."""
        payload = {
            "run": run_pk, "provider": provider, "model": model, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens, "billing": billing,
            "api_equivalent_cost_usd": api_equivalent_cost_usd, "external_id": external_id,
            "call_purpose": call_purpose, "status": status, **extra,
        }
        fallback.enqueue(self.fallback_path, "call", payload)
        return self._post("calls", payload)

    def flush_fallback(self) -> tuple[int, int]:
        """Resends every queued record. Returns (sent, total). Clears the queue file only if every
        record was sent successfully -- a partial failure leaves the whole file in place so nothing
        queued is ever silently dropped (the next flush just resends the already-sent ones too; the
        service's own external_id uniqueness is what makes that safe for calls that carry one)."""
        if self.base_url is None:
            return 0, 0
        records = fallback.read_all(self.fallback_path)
        if not records:
            return 0, 0
        sent = 0
        for record in records:
            path = "runs" if record["kind"] == "run" else "calls"
            # self._post() only does the HTTP call -- it does NOT touch the fallback file, so this
            # can't double-queue a record that's already in it.
            if self._post(path, record["payload"]) is not None:
                sent += 1
        if sent == len(records):
            fallback.clear(self.fallback_path)
        return sent, len(records)
