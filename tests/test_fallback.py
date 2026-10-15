from llm_ledger_client import fallback


def test_enqueue_and_read_all_round_trip(tmp_path):
    path = tmp_path / "queue.jsonl"
    fallback.enqueue(path, "run", {"run_id": "abc"})
    fallback.enqueue(path, "call", {"model": "x"})

    records = fallback.read_all(path)
    assert records == [{"kind": "run", "payload": {"run_id": "abc"}}, {"kind": "call", "payload": {"model": "x"}}]


def test_read_all_on_missing_file_is_empty_not_an_error(tmp_path):
    assert fallback.read_all(tmp_path / "does-not-exist.jsonl") == []


def test_clear_removes_the_file(tmp_path):
    path = tmp_path / "queue.jsonl"
    fallback.enqueue(path, "run", {"a": 1})
    fallback.clear(path)
    assert fallback.read_all(path) == []
    assert not path.exists()


def test_clear_on_missing_file_does_not_raise(tmp_path):
    fallback.clear(tmp_path / "nope.jsonl")  # must not raise
