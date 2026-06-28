"""Tests for benched.client."""

import pytest
import json

from benched import client


def test_tpot_and_throughput_calculation() -> None:
    sample = client.Sample(
        prompt_tokens=10,
        completion_tokens=10,
        ttft_ms=20.0,
        total_latency_ms=110.0,
    )
    assert sample.tpot_ms == pytest.approx(10.0)
    assert sample.throughput_tok_per_sec == pytest.approx(10 / 0.11)


def test_single_token_tpot() -> None:
    sample = client.Sample(
        prompt_tokens=10,
        completion_tokens=1,
        ttft_ms=20.0,
        total_latency_ms=20.0,
    )
    assert sample.tpot_ms == pytest.approx(20.0)
