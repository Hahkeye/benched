"""Tests for benched.db."""

import pytest
from pathlib import Path

from benched import db


@pytest.fixture
def database(tmp_path: Path) -> db.Database:
    path = tmp_path / "test.db"
    return db.Database(path)


def test_run_round_trip(database: db.Database) -> None:
    run_id = database.insert_run(
        backend="llama-cpp",
        model_path="/path/to/model.gguf",
        args_json='["-t", "4"]',
    )
    assert run_id == 1
    database.update_run_status(run_id, "success", ended_at="2024-01-01T00:00:00Z")
    run = database.get_run(run_id)
    assert run is not None
    assert run["status"] == "success"


def test_sample_aggregation(database: db.Database) -> None:
    run_id = database.insert_run(
        backend="llama-cpp",
        model_path="/path/to/model.gguf",
        args_json='["-t", "4"]',
    )
    for i in range(3):
        database.insert_sample(
            run_id=run_id,
            workload="synthetic",
            prompt_tokens=10,
            completion_tokens=20,
            ttft_ms=10.0 + i,
            tpot_ms=2.0 + i,
            total_latency_ms=50.0 + i,
            throughput_tok_per_sec=100.0 + i,
        )
    summary = database.get_run_summary(run_id)
    assert summary["sample_count"] == 3
    assert summary["median_throughput"] == 101.0


def test_best_run(database: db.Database) -> None:
    run1 = database.insert_run(
        backend="llama-cpp",
        model_path="/path/to/model.gguf",
        args_json='["-t", "4"]',
    )
    run2 = database.insert_run(
        backend="llama-cpp",
        model_path="/path/to/model.gguf",
        args_json='["-t", "8"]',
    )
    database.update_run_status(run1, "success")
    database.update_run_status(run2, "success")
    database.insert_sample(
        run_id=run1,
        workload="synthetic",
        throughput_tok_per_sec=100.0,
    )
    database.insert_sample(
        run_id=run2,
        workload="synthetic",
        throughput_tok_per_sec=200.0,
    )
    best = database.best_run("llama-cpp", "/path/to/model.gguf", "maximize throughput_tok_per_sec")
    assert best is not None
    assert best["id"] == run2
