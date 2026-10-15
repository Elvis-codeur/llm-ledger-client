import json

import pytest

from llm_ledger_client.session_ingest import PiAdapter, import_session, read_usage


def _write_jsonl(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_pi_adapter_reads_an_assistant_message_entry(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, [
        {"type": "session", "id": "sess-1"},
        {"type": "message", "id": "m1", "timestamp": "2026-10-15T10:00:00Z",
         "message": {"role": "user", "content": "translate this"}},
        {"type": "message", "id": "m2", "timestamp": "2026-10-15T10:00:05Z", "parentId": "m1",
         "message": {"role": "assistant", "provider": "deepseek", "model": "deepseek-flash",
                      "responseId": "resp-1", "timestamp": "2026-10-15T10:00:05Z",
                      "usage": {"input": 100, "output": 50, "cacheRead": 10, "cacheWrite": 0,
                                "cost": {"total": 0.0012}},
                      "content": [{"type": "text", "text": "voici la traduction"}]}},
    ])

    records = read_usage("pi", path)
    assert len(records) == 1
    r = records[0]
    assert r.provider == "deepseek" and r.model == "deepseek/deepseek-flash"
    assert r.prompt_tokens == 100 and r.completion_tokens == 50 and r.cache_read_tokens == 10
    assert r.billing == "api"
    assert r.external_id == "pi:resp-1:2026-10-15T10:00:05Z"
    assert r.reported_cost_usd == pytest.approx(0.0012)
    assert r.latency_ms == 5000


def test_pi_adapter_marks_github_copilot_as_subscription_billing(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, [
        {"type": "message", "id": "m1", "timestamp": "2026-10-15T10:00:00Z",
         "message": {"role": "assistant", "provider": "github-copilot", "model": "gpt-5.4",
                      "responseId": "resp-9", "timestamp": "2026-10-15T10:00:00Z",
                      "usage": {"input": 10, "output": 10}, "content": []}},
    ])
    records = read_usage("pi", path)
    assert records[0].billing == "subscription"


def test_an_assistant_entry_with_no_usage_is_skipped_not_an_error(tmp_path, caplog):
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, [
        {"type": "message", "id": "m1", "timestamp": "2026-10-15T10:00:00Z",
         "message": {"role": "assistant", "provider": "deepseek", "model": "x", "content": []}},
    ])
    assert read_usage("pi", path) == []


def test_unknown_source_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="unknown source"):
        read_usage("not-a-real-source", "/tmp/doesnotmatter.jsonl")


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_call(self, run_pk, **kwargs):
        self.calls.append((run_pk, kwargs))
        return {"id": len(self.calls)}


def test_import_session_posts_one_log_call_per_record(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, [
        {"type": "message", "id": "m1", "timestamp": "2026-10-15T10:00:00Z",
         "message": {"role": "assistant", "provider": "deepseek", "model": "deepseek-flash",
                      "responseId": "r1", "timestamp": "2026-10-15T10:00:00Z",
                      "usage": {"input": 5, "output": 5}, "content": []}},
    ])
    client = _FakeClient()
    n = import_session(client, run_pk=42, source="pi", path=path)
    assert n == 1
    run_pk, kwargs = client.calls[0]
    assert run_pk == 42 and kwargs["provider"] == "deepseek" and kwargs["external_id"] == "pi:r1:2026-10-15T10:00:00Z"
