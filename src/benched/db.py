"""SQLite schema and queries for benched."""

from __future__ import annotations

import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend TEXT NOT NULL,
    model_path TEXT NOT NULL,
    args_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    workload TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    ttft_ms REAL,
    tpot_ms REAL,
    total_latency_ms REAL,
    throughput_tok_per_sec REAL,
    error TEXT,
    sample_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_backend_model ON runs(backend, model_path);
CREATE INDEX IF NOT EXISTS idx_samples_run_id ON samples(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_db_path() -> Path:
    path = Path.home() / ".local" / "share" / "benched" / "benched.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_run(
        self,
        backend: str,
        model_path: str,
        args_json: str,
        status: str = "pending",
        started_at: str | None = None,
    ) -> int:
        started_at = started_at or _now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (backend, model_path, args_json, status, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (backend, model_path, args_json, status, started_at),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_run_status(
        self,
        run_id: int,
        status: str,
        error_message: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        ended_at = ended_at or _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, error_message = ?, ended_at = ?
                WHERE id = ?
                """,
                (status, error_message, ended_at, run_id),
            )
            conn.commit()

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def insert_sample(
        self,
        run_id: int,
        workload: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        ttft_ms: float | None = None,
        tpot_ms: float | None = None,
        total_latency_ms: float | None = None,
        throughput_tok_per_sec: float | None = None,
        error: str | None = None,
        sample_json: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO samples
                (run_id, workload, prompt_tokens, completion_tokens, ttft_ms, tpot_ms,
                 total_latency_ms, throughput_tok_per_sec, error, sample_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    workload,
                    prompt_tokens,
                    completion_tokens,
                    ttft_ms,
                    tpot_ms,
                    total_latency_ms,
                    throughput_tok_per_sec,
                    error,
                    sample_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_samples_for_run(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM samples WHERE run_id = ?", (run_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def list_runs(
        self,
        backend: str | None = None,
        model: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM runs"
        params: list[Any] = []
        conditions: list[str] = []
        if backend:
            conditions.append("backend = ?")
            params.append(backend)
        if model:
            conditions.append("model_path = ?")
            params.append(model)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_run_summary(self, run_id: int) -> dict[str, Any]:
        samples = self.get_samples_for_run(run_id)
        total = len(samples)
        errors = [s for s in samples if s.get("error")]
        ok = [s for s in samples if not s.get("error")]

        def _median(values: list[float]) -> float | None:
            return statistics.median(values) if values else None

        return {
            "run_id": run_id,
            "sample_count": total,
            "error_count": len(errors),
            "error_rate": len(errors) / total if total else 0.0,
            "median_ttft_ms": _median([s["ttft_ms"] for s in ok if s["ttft_ms"] is not None]),
            "median_tpot_ms": _median([s["tpot_ms"] for s in ok if s["tpot_ms"] is not None]),
            "median_total_latency_ms": _median(
                [s["total_latency_ms"] for s in ok if s["total_latency_ms"] is not None]
            ),
            "median_throughput": _median(
                [s["throughput_tok_per_sec"] for s in ok if s["throughput_tok_per_sec"] is not None]
            ),
        }

    def best_run(
        self,
        backend: str,
        model: str,
        objective: str,
    ) -> dict[str, Any] | None:
        """Return the best run for a backend/model according to an objective.

        The objective string must be in the form "maximize <metric>" or
        "minimize <metric>"."""
        parts = objective.split(None, 1)
        if len(parts) != 2:
            return None
        kind, metric = parts
        if kind not in ("maximize", "minimize"):
            return None
        order = "DESC" if kind == "maximize" else "ASC"

        metric_column = {
            "throughput_tok_per_sec": "throughput_tok_per_sec",
            "ttft_ms": "ttft_ms",
            "tpot_ms": "tpot_ms",
            "total_latency_ms": "total_latency_ms",
        }.get(metric)
        if metric_column is None:
            return None

        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT r.*, s.{metric_column}
                FROM runs r
                JOIN samples s ON s.run_id = r.id
                WHERE r.backend = ? AND r.model_path = ? AND r.status = 'success' AND s.{metric_column} IS NOT NULL
                ORDER BY s.{metric_column} {order}
                LIMIT 1
                """,
                (backend, model),
            ).fetchone()
            return dict(row) if row else None

    def run_with_summary(self, run_id: int) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        run["summary"] = self.get_run_summary(run_id)
        try:
            run["args"] = json.loads(run["args_json"])
        except json.JSONDecodeError:
            run["args"] = []
        return run

    def clear(self) -> None:
        """Delete all runs and samples. Useful for tests."""
        with self._connect() as conn:
            conn.execute("DELETE FROM samples")
            conn.execute("DELETE FROM runs")
            conn.commit()
