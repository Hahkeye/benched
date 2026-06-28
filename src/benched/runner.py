"""Sweep orchestration."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from benched.builders.base import ServerPaths
from benched.client import Sample
from benched.config import Config, matrix_combinations
from benched.db import Database
from benched.objectives import aggregate_samples
from benched.server import ServerStartupError, start_server
from benched.workloads import run_warmup, run_workload


class SweepError(Exception):
    """Raised when a sweep cannot proceed."""


def _resolve_paths(cfg: Config, **kwargs: Any) -> ServerPaths:
    """Find the server binary/venv — user must provide a path, no auto-build."""
    if cfg.backend == "llama-cpp":
        binary = kwargs.get("binary")
        if not binary:
            raise SweepError(
                "llama-server binary path required.\n"
                "  Build llama.cpp yourself, then pass:\n"
                "    benched run --binary /path/to/llama-server --config ..."
            )
        p = Path(binary)
        if not p.exists():
            raise SweepError(f"llama-server not found at {p}")
        return ServerPaths(str(p), [])

    if cfg.backend == "vllm":
        venv = kwargs.get("venv")
        if not venv:
            raise SweepError(
                "vLLM virtual environment path required.\n"
                "  Install vLLM yourself, then pass:\n"
                "    benched run --venv /path/to/venv --config ..."
            )
        venv_path = Path(venv)
        if sys.platform == "win32":
            py = venv_path / "Scripts" / "python.exe"
        else:
            py = venv_path / "bin" / "python"
        if not py.exists():
            raise SweepError(f"vLLM venv python not found at {py}")
        return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])

    raise SweepError(f"unknown backend: {cfg.backend}")


def _format_args(args: list[str]) -> str:
    """Format a list of CLI args into a readable key=value string."""
    parts = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-m", "--model", "--host", "--port"):
            i += 2
            continue
        if i + 1 < len(args) and not args[i + 1].startswith("-"):
            parts.append(f"{arg}={args[i + 1]}")
            i += 2
        else:
            parts.append(arg)
            i += 1
    return " ".join(parts)


async def run_single(
    cfg: Config,
    paths: ServerPaths,
    combination: list[str],
    db: Database,
    run_id: int,
) -> None:
    """Execute one run in the sweep."""
    try:
        async with start_server(cfg, paths, combination) as server:
            print(f"[runner] run {run_id} server healthy on port {server.port}")
            await run_warmup(cfg, server.base_url)
            samples = await run_workload(cfg, server.base_url)
    except ServerStartupError as exc:
        db.update_run_status(run_id, "error", error_message=exc.stderr)
        raise
    except Exception as exc:
        db.update_run_status(run_id, "error", error_message=str(exc))
        raise

    # Persist samples.
    for sample in samples:
        db.insert_sample(
            run_id=run_id,
            workload=cfg.workload.kind,
            prompt_tokens=sample.prompt_tokens,
            completion_tokens=sample.completion_tokens,
            ttft_ms=sample.ttft_ms,
            tpot_ms=sample.tpot_ms,
            total_latency_ms=sample.total_latency_ms,
            throughput_tok_per_sec=sample.throughput_tok_per_sec,
            error=sample.error,
            sample_json=json.dumps(sample.metadata or {}),
        )

    summary = aggregate_samples(
        [
            {
                "ttft_ms": s.ttft_ms,
                "tpot_ms": s.tpot_ms,
                "total_latency_ms": s.total_latency_ms,
                "throughput_tok_per_sec": s.throughput_tok_per_sec,
                "error": s.error,
            }
            for s in samples
        ]
    )
    db.update_run_status(run_id, "success")
    print(
        f"[runner] run {run_id} complete | "
        f"throughput={summary['median_throughput_tok_per_sec']:.2f} tok/s | "
        f"errors={summary['error_count']}/{summary['sample_count']}"
    )


async def run_sweep(
    config_path: str | Path,
    *,
    db: Database | None = None,
    dry_run: bool = False,
    continue_from: int | None = None,
    model_override: str | Path | None = None,
    **build_kwargs: Any,
) -> None:
    """Load a config and execute (or dry-run) the full sweep."""
    from benched.config import load_config

    cfg = load_config(config_path)
    if model_override:
        cfg.model = Path(model_override)
        if not cfg.model.exists():
            raise SweepError(f"override model path does not exist: {cfg.model}")

    db = db or Database()
    combinations = matrix_combinations(cfg.server.matrix)

    if dry_run:
        print(f"dry-run: {len(combinations)} configurations")
        for i, combo in enumerate(combinations, 1):
            print(f"  {i}/{len(combinations)} backend={cfg.backend} model={cfg.model} args={combo}")
        return

    paths = _resolve_paths(cfg, **build_kwargs)

    # Optional resume: find runs that already succeeded and skip them.
    successful_hashes: set[str] = set()
    if continue_from is not None:
        for run in db.list_runs(backend=cfg.backend, model=str(cfg.model), limit=10000):
            if run["status"] == "success":
                successful_hashes.add(run["args_json"])

    for i, combination in enumerate(combinations, 1):
        args_json = json.dumps(combination)
        if args_json in successful_hashes:
            print(f"[runner] skipping {i}/{len(combinations)} (already successful)")
            continue

        run_id = db.insert_run(
            backend=cfg.backend,
            model_path=str(cfg.model),
            args_json=args_json,
            status="running",
        )
        print(
            f"[runner] run {i}/{len(combinations)} | "
            f"backend={cfg.backend} | {_format_args(combination)}"
        )
        try:
            await run_single(cfg, paths, combination, db, run_id)
        except Exception as exc:
            print(f"[runner] run {run_id} failed: {exc}")
            # Continue with next configuration unless this is the last one.
            if i == len(combinations):
                raise


async def list_results(
    backend: str | None = None,
    model: str | None = None,
    limit: int = 20,
    db: Database | None = None,
) -> list[dict[str, Any]]:
    db = db or Database()
    runs = db.list_runs(backend=backend, model=model, limit=limit)
    for run in runs:
        run["summary"] = db.get_run_summary(run["id"])
        try:
            run["args"] = json.loads(run["args_json"])
        except json.JSONDecodeError:
            run["args"] = []
    return runs


async def show_run(run_id: int, db: Database | None = None) -> dict[str, Any] | None:
    db = db or Database()
    run = db.run_with_summary(run_id)
    if run is None:
        return None
    run["samples"] = db.get_samples_for_run(run_id)
    return run
