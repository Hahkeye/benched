"""Scoring and ranking of benchmark results."""

from __future__ import annotations

import json
import statistics
from typing import Any

from benched.config import Objective, parse_objectives
from benched.db import Database


METRIC_KEYS = {
    "throughput_tok_per_sec",
    "ttft_ms",
    "tpot_ms",
    "total_latency_ms",
}


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sample rows into run-level metrics."""
    total = len(samples)
    errors = [s for s in samples if s.get("error")]
    ok = [s for s in samples if not s.get("error")]

    def _collect(key: str) -> list[float]:
        return [s[key] for s in ok if s.get(key) is not None]

    return {
        "sample_count": total,
        "error_count": len(errors),
        "error_rate": len(errors) / total if total else 0.0,
        "median_ttft_ms": _median(_collect("ttft_ms")),
        "median_tpot_ms": _median(_collect("tpot_ms")),
        "median_total_latency_ms": _median(_collect("total_latency_ms")),
        "median_throughput_tok_per_sec": _median(_collect("throughput_tok_per_sec")),
    }


def score_run(
    summary: dict[str, Any],
    objectives: list[Objective],
) -> float:
    """Compute a scalar score for a run; higher is better."""
    score = 0.0
    for obj in objectives:
        key = obj.metric
        value = summary.get(f"median_{key}", summary.get(key, 0.0))
        if value is None or not isinstance(value, (int, float)):
            value = 0.0
        # Normalize by adding a small epsilon to avoid division by zero.
        if obj.kind == "maximize":
            contribution = value
        else:
            contribution = 1.0 / (value + 1e-9)
        score += obj.weight * contribution
    return score


def rank_runs(
    runs: list[dict[str, Any]],
    objective_text: str,
) -> list[dict[str, Any]]:
    """Rank runs by the configured objective."""
    objectives = parse_objectives(objective_text)
    scored = []
    for run in runs:
        summary = run.get("summary") or aggregate_samples(run.get("samples", []))
        scored.append(
            {
                "run": run,
                "summary": summary,
                "score": score_run(summary, objectives),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def recommend(
    db: Database,
    backend: str,
    model: str,
    objective_text: str,
    top: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-N recommended runs for a backend/model."""
    runs = db.list_runs(backend=backend, model=model, limit=1000)
    for run in runs:
        run["summary"] = db.get_run_summary(run["id"])
        try:
            run["args"] = json.loads(run["args_json"])
        except json.JSONDecodeError:
            run["args"] = []
    ranked = rank_runs(runs, objective_text)
    return ranked[:top]
