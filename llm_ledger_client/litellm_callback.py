"""A litellm CustomLogger that posts every completion's usage to a LedgerClient. litellm already
computes `kwargs["response_cost"]` for providers it has pricing for -- we pass that through as
`api_equivalent_cost_usd` for cross-checking ONLY; the real `cost_usd` recorded is always the ledger
service's own server-side computation from its ModelPricing table (see client.py's docstring for why:
this project deliberately does not trust a bundled, possibly-stale price table as the source of truth).

Usage:
    import litellm
    from llm_ledger_client import LedgerClient
    from llm_ledger_client.litellm_callback import LedgerLoggingCallback

    client = LedgerClient("http://localhost:8000/api", project="my-project")
    run = client.create_run(run_id=str(uuid.uuid4()), label="batch-translate")
    litellm.callbacks = [LedgerLoggingCallback(client, run_pk=run["id"])]
    # ... now every litellm.completion()/acompletion() call anywhere in the process logs its cost.
"""

from __future__ import annotations

import logging

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "litellm_callback requires litellm -- install with `pip install llm-ledger-client[litellm]`"
    ) from exc

from .client import LedgerClient

_logger = logging.getLogger("llm_ledger_client")


def _usage_field(usage, key: str, default: int = 0) -> int:
    """`usage` from litellm is sometimes a dict, sometimes a Usage object with attributes -- handle both
    rather than assuming one shape."""
    if isinstance(usage, dict):
        return usage.get(key, default) or default
    return getattr(usage, key, default) or default


class LedgerLoggingCallback(CustomLogger):
    def __init__(self, client: LedgerClient, run_pk: int, *, call_purpose: str = ""):
        self.client = client
        self.run_pk = run_pk
        self.call_purpose = call_purpose

    def _log(self, kwargs: dict, response_obj, status: str) -> None:
        usage = getattr(response_obj, "usage", None) or kwargs.get("usage") or {}
        model = kwargs.get("model", "")
        provider = (kwargs.get("custom_llm_provider") or model.split("/", 1)[0] if "/" in model else "unknown")
        self.client.log_call(
            self.run_pk,
            provider=provider,
            model=model,
            prompt_tokens=_usage_field(usage, "prompt_tokens"),
            completion_tokens=_usage_field(usage, "completion_tokens"),
            billing="api",
            api_equivalent_cost_usd=kwargs.get("response_cost"),
            call_purpose=self.call_purpose,
            status=status,
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._log(kwargs, response_obj, "success")
        except Exception:  # never let a logging failure break the actual LLM call
            _logger.exception("llm_ledger_client: failed to log a successful litellm call")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._log(kwargs, response_obj, "error")
        except Exception:
            _logger.exception("llm_ledger_client: failed to log a failed litellm call")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
