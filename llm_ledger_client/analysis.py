"""The read/analysis-side SDK for django-llm-ledger's cost-attribution endpoints (Tag, Expense,
/cost-summary/, /cost-timeseries/, /cost-distribution/) -- so an agent asking "how much did project X
spend between these two dates, split by day, and what does the distribution of individual run costs
look like" calls one method instead of hand-rolling pagination, tag-AND filtering, date bucketing, and
percentile math against the raw REST API every time.

Unlike LedgerClient (client.py), which is write-heavy and designed to never lose a record (local
fallback queue, silent failure), this reader surfaces errors directly -- a cost analysis that silently
returned wrong/partial numbers would be worse than one that raised, since the whole point is to be
trusted for real financial decisions."""

from __future__ import annotations

from typing import Any

import requests


class LedgerAnalysis:
    def __init__(self, base_url: str, *, timeout: float = 10.0):
        """`base_url` is e.g. "http://127.0.0.1:8090/api" (standalone) or
        "https://your-host/api/ledger" (embedded mount)."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = requests.get(f"{self.base_url}/{path}/", params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> Any:
        resp = requests.post(f"{self.base_url}/{path}/", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # -- providers --------------------------------------------------------------------------------

    def get_or_create_provider(self, name: str, **fields: Any) -> dict:
        """Idempotent server-side by name. `fields` can set display_name/website_url/notes."""
        return self._post("providers", {"name": name, **fields})

    def add_provider_pricing_policy(
        self,
        provider_id: int,
        *,
        offpeak_multiplier: str | float,
        peak_windows_utc: dict,
        effective_from: str,
        effective_until: str | None = None,
        source_url: str = "",
        verification: str = "unverified",
        note: str = "",
    ) -> dict:
        """Appends a new peak/off-peak policy row for a provider -- NEVER edit an existing policy in
        place: a provider's schedule changing over time needs a new row with a new effective_from,
        exactly like a price change, or a past call's cost would retroactively change. Set
        `effective_until` on the OLD policy yourself (a PATCH to its id) when adding a replacement
        that starts where it ends."""
        payload: dict[str, Any] = {
            "provider": provider_id, "offpeak_multiplier": str(offpeak_multiplier),
            "peak_windows_utc": peak_windows_utc, "effective_from": effective_from,
            "source_url": source_url, "verification": verification, "note": note,
        }
        if effective_until:
            payload["effective_until"] = effective_until
        return self._post("provider-pricing-policies", payload)

    def provider_peak_status(self, provider_id: int, *, at: str | None = None) -> dict:
        """Is this provider in peak or off-peak pricing at `at` (default: now)? -- independent of any
        one model."""
        return self._get(f"providers/{provider_id}/peak_status", {"at": at} if at else None)

    # -- tags -----------------------------------------------------------------------------------

    def get_or_create_tag(self, key: str, value: str) -> dict:
        """Idempotent server-side: the same (key, value) always resolves to the same tag row."""
        return self._post("tags", {"key": key, "value": value})

    def list_tags(self, **filters: str) -> list[dict]:
        return self._paginate("tags", filters)

    def tag_cost(self, tag_id: int) -> dict:
        return self._get(f"tags/{tag_id}/cost")

    # -- expenses ---------------------------------------------------------------------------------

    def create_expense(
        self,
        source: str,
        amount_usd: str | float,
        *,
        category: str = "",
        occurred_at: str | None = None,
        description: str = "",
        external_id: str = "",
        raw: dict | None = None,
        tags: list[int] | None = None,
        run: int | None = None,
    ) -> dict:
        """Idempotent server-side by (source, external_id) when external_id is given -- safe to call
        again from a re-run importer without double-counting real spend."""
        payload: dict[str, Any] = {
            "source": source, "amount_usd": str(amount_usd), "category": category,
            "description": description, "external_id": external_id, "raw": raw or {},
            "tags": tags or [],
        }
        if occurred_at:
            payload["occurred_at"] = occurred_at
        if run is not None:
            payload["run"] = run
        return self._post("expenses", payload)

    def log_manual_expense(
        self,
        amount_usd: str | float,
        description: str,
        *,
        category: str = "",
        occurred_at: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict:
        """The "I topped up $5 for a library" / "I bought a one-off tool license" case -- a cost a
        human noticed and wants recorded, with no importer or API behind it. `source` is always
        "manual" so these are easy to tell apart from an automated import later; `tags` here takes
        key/value PAIRS (not tag ids like `create_expense`) and resolves/creates them for you, since a
        human logging a one-off cost shouldn't have to look up tag ids first."""
        tag_ids = [self.get_or_create_tag(k, v)["id"] for k, v in (tags or {}).items()]
        return self.create_expense(
            "manual", amount_usd, category=category, occurred_at=occurred_at,
            description=description, tags=tag_ids,
        )

    def add_expense_comment(self, expense_id: int, text: str, *, author: str = "") -> dict:
        return self._post(f"expenses/{expense_id}/comments", {"text": text, "author": author})

    def list_expense_comments(self, expense_id: int) -> list[dict]:
        return self._get(f"expenses/{expense_id}/comments")

    def void_expense(self, expense_id: int, reason: str, *, superseded_by: int | None = None) -> dict:
        """Marks an Expense voided (never delete one that may have been reported on already) -- the
        original row stays, with why and, optionally, a link to whatever replaced it."""
        payload: dict[str, Any] = {"reason": reason}
        if superseded_by is not None:
            payload["superseded_by"] = superseded_by
        return self._post(f"expenses/{expense_id}/void", payload)

    def list_expenses(self, **filters: Any) -> list[dict]:
        return self._paginate("expenses", filters)

    # -- cost analysis ----------------------------------------------------------------------------

    def cost_summary(self, tags: dict[str, str]) -> dict:
        """Total LLM-call + Expense cost under every given tag (AND, not OR) -- `tags={"project": "x",
        "type": "training"}` means "spent on project=x AND type=training", not either alone."""
        return self._get("cost-summary", self._tag_params(tags))

    def cost_timeseries(
        self,
        *,
        tags: dict[str, str] | None = None,
        start: str | None = None,
        end: str | None = None,
        bucket: str = "day",
    ) -> dict:
        """Cost split into day/week/month buckets between `start`/`end` (either or both optional;
        dates like "2026-10-01" or full ISO datetimes), optionally narrowed by tags. Returns
        {"series": [{"bucket_start", "llm_cost_usd", "expense_cost_usd", "total_cost_usd"}, ...],
        "total_cost_usd"}."""
        params = self._tag_params(tags or {})
        params["bucket"] = bucket
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._get("cost-timeseries", params)

    def cost_distribution(
        self,
        *,
        metric: str = "expense",
        tags: dict[str, str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        """count/sum/mean/median/stdev/min/max/percentiles over individual cost amounts -- `metric` is
        "expense" (e.g. individual GPU-pod runs) or "llm_call" (individual API call costs)."""
        params = self._tag_params(tags or {})
        params["metric"] = metric
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._get("cost-distribution", params)

    # -- internals --------------------------------------------------------------------------------

    @staticmethod
    def _tag_params(tags: dict[str, str]) -> dict:
        if not tags:
            return {}
        return {"key": list(tags.keys()), "value": list(tags.values())}

    def _paginate(self, path: str, params: dict) -> list[dict]:
        """DRF's default pagination wraps list responses in {"results": [...], "next": ...} -- this
        follows `next` so a caller always gets the full matching list, never just page one."""
        results: list[dict] = []
        url = f"{self.base_url}/{path}/"
        while url:
            resp = requests.get(url, params=params if results == [] else None, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data  # an unpaginated endpoint
            results.extend(data.get("results", []))
            url = data.get("next")
        return results
