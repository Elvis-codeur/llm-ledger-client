# llm-ledger-client

A thin client for [django-llm-ledger](https://github.com/Elvis-codeur/django-llm-ledger). Three things,
each usable on its own:

1. **`LedgerClient`** — the base HTTP client. Posts raw token counts, never a claimed cost — the ledger
   service always computes `cost_usd` server-side. Every write goes to a local, append-only fallback
   queue first, then is attempted against the live service, so a network failure never loses a
   real, already-computed record.
2. **A litellm callback** (`llm_ledger_client.litellm_callback.LedgerLoggingCallback`) — register once,
   every `litellm.completion()`/`acompletion()` call in the process logs its cost automatically.
3. **An agent-session ingester** (`llm_ledger_client.session_ingest`) — for agent tools that talk to a
   provider directly, not through litellm (e.g. [Pi](https://github.com/badlogic/pi-mono)/DeepSeek).
   Reads the tool's own session transcript after the fact and posts one call per assistant message,
   idempotently (`external_id`, enforced unique per run by the ledger service).

## Quick start

```python
import uuid
from llm_ledger_client import LedgerClient

client = LedgerClient("http://localhost:8000/api", project="my-project")
run = client.create_run(str(uuid.uuid4()), label="batch-translate", purpose="production")

client.log_call(run["id"], provider="deepseek", model="deepseek/deepseek-flash",
                 prompt_tokens=1200, completion_tokens=340, billing="api")

client.complete_run(run["id"], status="success")
client.flush_fallback()  # resend anything that failed live, clear the queue once confirmed
```

## litellm integration

```python
import litellm
from llm_ledger_client import LedgerClient
from llm_ledger_client.litellm_callback import LedgerLoggingCallback

client = LedgerClient("http://localhost:8000/api", project="my-project")
run = client.create_run(str(uuid.uuid4()))
litellm.callbacks = [LedgerLoggingCallback(client, run_pk=run["id"])]
# every litellm call in this process now logs its cost
```

Requires the `litellm` extra: `pip install llm-ledger-client[litellm]`.

## Agent-session ingestion (Pi)

```python
from llm_ledger_client.session_ingest import import_session

n = import_session(client, run_pk=run["id"], source="pi", path="~/.pi/agent/sessions/.../session.jsonl")
print(f"imported {n} calls")
```

Only the Pi adapter ships in this first release, ported faithfully from
`agentic_doc.agent_usage.PiAdapter` (the real gotchas it handles — the two different session-file event
shapes, idempotency keys, subscription-provider detection — are preserved, not reinvented). Add another
`source=` by implementing `read(path) -> Iterator[UsageRecord]` and registering it in
`session_ingest.ADAPTERS`.

## Why queue-then-send, not send-then-queue-on-failure

A crash DURING the live network call (not just a clean failure) must not lose the record — enqueueing
to the local file first, unconditionally, closes that window. The cost is that `flush_fallback()` may
re-send a record that already succeeded live; that's only safe because the ledger service's own write
paths are idempotent (`LLMCall.external_id`, `Run.run_id` — both confirmed, not assumed, by a real
cross-repo integration test between this library and django-llm-ledger during development).

## Status

Freshly extracted (2026-10-15) from a document-translation pipeline's own cost-tracking code. Tests:
`pytest` (17 passing), plus a real integration test against a live `django-llm-ledger` instance during
development (not shipped as an automated test here, since it needs a running server — see that repo's
own test suite for the server-side half of this contract).
