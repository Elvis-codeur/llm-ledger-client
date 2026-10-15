"""LedgerClient: queue-then-send resilience, and that it never trusts its own cost figure as real."""

import json

import pytest
import requests

from llm_ledger_client import LedgerClient
from llm_ledger_client import fallback


class _FakeResponse:
    def __init__(self, json_data, ok=True, status=200):
        self._json = json_data
        self.ok = ok
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"status {self.status_code}")


@pytest.fixture
def client(tmp_path):
    return LedgerClient("http://fake-ledger/api", project="test-project", fallback_path=tmp_path / "queue.jsonl")


class TestCreateRun:
    def test_posts_and_returns_the_server_response(self, client, monkeypatch):
        sent = {}

        def fake_post(url, json, timeout):
            sent["url"], sent["json"] = url, json
            return _FakeResponse({"id": 7, "run_id": json["run_id"]})

        monkeypatch.setattr(requests, "post", fake_post)
        result = client.create_run("abc-123", label="smoke")
        assert result == {"id": 7, "run_id": "abc-123"}
        assert sent["url"] == "http://fake-ledger/api/runs/"
        assert sent["json"]["project"] == "test-project"

    def test_always_queues_to_the_fallback_file_even_when_the_post_succeeds(self, client, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse({"id": 1}))
        client.create_run("abc-123")
        records = fallback.read_all(client.fallback_path)
        assert len(records) == 1 and records[0]["kind"] == "run"

    def test_a_network_failure_still_queues_and_returns_none_not_raising(self, client, monkeypatch):
        def fake_post(*a, **k):
            raise requests.ConnectionError("down")

        monkeypatch.setattr(requests, "post", fake_post)
        result = client.create_run("abc-123")
        assert result is None
        assert len(fallback.read_all(client.fallback_path)) == 1

    def test_base_url_none_never_attempts_the_network_queue_only(self, tmp_path, monkeypatch):
        c = LedgerClient(None, project="test", fallback_path=tmp_path / "q.jsonl")

        def should_not_be_called(*a, **k):
            raise AssertionError("must not call requests.post when base_url is None")

        monkeypatch.setattr(requests, "post", should_not_be_called)
        result = c.create_run("abc")
        assert result is None
        assert len(fallback.read_all(c.fallback_path)) == 1


class TestLogCall:
    def test_never_sends_a_claimed_cost_usd_field(self, client, monkeypatch):
        sent = {}

        def fake_post(url, json, timeout):
            sent.update(json)
            return _FakeResponse({"id": 1, "cost_usd": "0.000000"})

        monkeypatch.setattr(requests, "post", fake_post)
        client.log_call(1, provider="deepseek", model="deepseek/deepseek-flash", prompt_tokens=100)
        assert "cost_usd" not in sent


class TestFlushFallback:
    def test_resends_every_queued_record_and_clears_the_file_on_full_success(self, client, monkeypatch):
        # force two records into the queue via a failing network
        def fail(*a, **k):
            raise requests.ConnectionError("down")
        monkeypatch.setattr(requests, "post", fail)
        client.create_run("abc")
        client.log_call(1, provider="deepseek", model="m")
        assert len(fallback.read_all(client.fallback_path)) == 2

        sent = []
        def ok(url, json, timeout):
            sent.append(url)
            return _FakeResponse({"id": 1})
        monkeypatch.setattr(requests, "post", ok)
        sent_count, total = client.flush_fallback()
        assert (sent_count, total) == (2, 2)
        assert fallback.read_all(client.fallback_path) == []

    def test_a_partial_failure_leaves_the_whole_queue_in_place(self, client, monkeypatch):
        def fail(*a, **k):
            raise requests.ConnectionError("down")
        monkeypatch.setattr(requests, "post", fail)
        client.create_run("abc")
        client.log_call(1, provider="deepseek", model="m")

        calls = {"n": 0}
        def flaky(url, json, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse({"id": 1})
            raise requests.ConnectionError("still down")
        monkeypatch.setattr(requests, "post", flaky)
        sent_count, total = client.flush_fallback()
        assert sent_count == 1 and total == 2
        assert len(fallback.read_all(client.fallback_path)) == 2  # nothing dropped

    def test_nothing_queued_is_a_no_op(self, client):
        assert client.flush_fallback() == (0, 0)
