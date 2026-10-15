"""LedgerAnalysis: the read/cost-analysis SDK. Mocked HTTP, same convention as test_client.py -- this
package is meant to work against any django-llm-ledger deployment, so its unit tests don't require a
live Django service (see the project's README for the real, live-tested confirmation against the
actually running standalone instance)."""

import pytest
import requests

from llm_ledger_client import LedgerAnalysis


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
def analysis():
    return LedgerAnalysis("http://fake-ledger/api")


def test_get_or_create_tag_posts_key_value(analysis, monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"], sent["json"] = url, json
        return _FakeResponse({"id": 1, "key": json["key"], "value": json["value"]})

    monkeypatch.setattr(requests, "post", fake_post)
    result = analysis.get_or_create_tag("project", "reasoner")
    assert sent["url"] == "http://fake-ledger/api/tags/"
    assert sent["json"] == {"key": "project", "value": "reasoner"}
    assert result["id"] == 1


def test_cost_summary_sends_tags_as_repeated_key_value_params(analysis, monkeypatch):
    sent = {}

    def fake_get(url, params, timeout):
        sent["url"], sent["params"] = url, params
        return _FakeResponse({"total_cost_usd": "12.34"})

    monkeypatch.setattr(requests, "get", fake_get)
    result = analysis.cost_summary({"project": "reasoner", "type": "training"})
    assert sent["url"] == "http://fake-ledger/api/cost-summary/"
    assert sent["params"] == {"key": ["project", "type"], "value": ["reasoner", "training"]}
    assert result["total_cost_usd"] == "12.34"


def test_cost_summary_with_no_tags_sends_no_key_value_params(analysis, monkeypatch):
    sent = {}

    def fake_get(url, params, timeout):
        sent["params"] = params
        return _FakeResponse({"total_cost_usd": "0"})

    monkeypatch.setattr(requests, "get", fake_get)
    analysis.cost_summary({})
    assert sent["params"] == {}


def test_cost_timeseries_includes_bucket_and_optional_date_range(analysis, monkeypatch):
    sent = {}

    def fake_get(url, params, timeout):
        sent["url"], sent["params"] = url, params
        return _FakeResponse({"series": []})

    monkeypatch.setattr(requests, "get", fake_get)
    analysis.cost_timeseries(start="2026-10-01", end="2026-10-15", bucket="week", tags={"project": "x"})
    assert sent["url"] == "http://fake-ledger/api/cost-timeseries/"
    assert sent["params"] == {"key": ["project"], "value": ["x"], "bucket": "week", "start": "2026-10-01", "end": "2026-10-15"}


def test_cost_distribution_defaults_to_expense_metric(analysis, monkeypatch):
    sent = {}

    def fake_get(url, params, timeout):
        sent["params"] = params
        return _FakeResponse({"count": 0})

    monkeypatch.setattr(requests, "get", fake_get)
    analysis.cost_distribution()
    assert sent["params"]["metric"] == "expense"


def test_create_expense_omits_empty_occurred_at_and_run(analysis, monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["json"] = json
        return _FakeResponse({"id": 5})

    monkeypatch.setattr(requests, "post", fake_post)
    analysis.create_expense("runpod", "3.50", category="pod")
    assert "occurred_at" not in sent["json"]
    assert "run" not in sent["json"]


def test_void_expense_posts_reason_and_optional_superseded_by(analysis, monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"], sent["json"] = url, json
        return _FakeResponse({"id": 1, "voided": True})

    monkeypatch.setattr(requests, "post", fake_post)
    analysis.void_expense(1, "wrong amount", superseded_by=2)
    assert sent["url"] == "http://fake-ledger/api/expenses/1/void/"
    assert sent["json"] == {"reason": "wrong amount", "superseded_by": 2}


def test_list_tags_follows_pagination_next_links(analysis, monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=10.0):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse({"results": [{"id": 1}], "next": "http://fake-ledger/api/tags/?cursor=2"})
        return _FakeResponse({"results": [{"id": 2}], "next": None})

    monkeypatch.setattr(requests, "get", fake_get)
    result = analysis.list_tags()
    assert [r["id"] for r in result] == [1, 2]
    assert len(calls) == 2


def test_a_real_http_error_is_raised_not_swallowed(analysis, monkeypatch):
    """Unlike LedgerClient's write-side resilience (queue, never raise), the analysis/read SDK must
    surface a real failure -- silently returning wrong or partial numbers for a cost analysis would be
    worse than raising."""

    def fake_get(url, params, timeout):
        return _FakeResponse({}, ok=False, status=500)

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        analysis.cost_summary({"project": "x"})
