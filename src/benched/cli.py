"""Command-line interface for benched."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

from benched.config import Config, ConfigError, ServerConfig, SyntheticWorkload, default_sweep_matrix, load_config
from benched.dashboard import run_dashboard
from benched.db import Database
from benched.objectives import recommend
from benched.runner import list_results, run_sweep, show_run


def _run(args: argparse.Namespace) -> int:
    try:
        if args.config:
            # Config-file mode
            asyncio.run(
                run_sweep(
                    args.config,
                    dry_run=args.dry_run,
                    model_override=args.model,
                    continue_from=args.continue_from,
                    binary=args.binary,
                    venv=args.venv,
                )
            )
        else:
            # Auto-sweep mode: build a default Config from --backend/--model
            if not args.backend or not args.model:
                print("error: provide --config or both --backend and --model", file=sys.stderr)
                return 1
            model_path = Path(args.model)
            if not args.dry_run and not model_path.exists():
                print(f"error: model not found: {model_path}", file=sys.stderr)
                return 1
            matrix = default_sweep_matrix(args.backend)
            if args.mtp:
                matrix.append([[], ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]])
            cfg = Config(
                backend=args.backend,
                model=model_path,
                server=ServerConfig(matrix=matrix),
                workload=SyntheticWorkload(
                    kind="synthetic",
                    input_tokens=args.prompt_len,
                    output_tokens=args.gen_len,
                    concurrent_requests=args.concurrency,
                    total_requests=args.total_requests,
                ),
            )
            asyncio.run(
                run_sweep(
                    cfg,
                    dry_run=args.dry_run,
                    model_override=args.model,
                    continue_from=args.continue_from,
                    binary=args.binary,
                    venv=args.venv,
                )
            )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _list(args: argparse.Namespace) -> int:
    runs = asyncio.run(
        list_results(
            backend=args.backend,
            model=args.model,
            limit=args.limit,
        )
    )
    header = f"{'id':>6} {'backend':<12} {'model':<30} {'status':<10} {'throughput':>12} {'errors':>10}"
    print(header)
    for run in runs:
        summary = run.get("summary", {})
        print(
            f"{run['id']:>6} {run['backend']:<12} {Path(run['model_path']).name:<30} "
            f"{run['status']:<10} {summary.get('median_throughput', 0.0) or 0.0:>12.2f} "
            f"{summary.get('error_count', 0):>5}/{summary.get('sample_count', 0):<5}"
        )
    return 0


def _show(args: argparse.Namespace) -> int:
    run = asyncio.run(show_run(args.run_id))
    if run is None:
        print(f"run {args.run_id} not found", file=sys.stderr)
        return 1
    print(json.dumps(run, indent=2, default=str))
    return 0


def _reset(args: argparse.Namespace) -> int:
    db = Database()
    db.drop_db()
    print(f"deleted database: {db.path}")
    return 0


def _recommend(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    ranked = recommend(
        Database(),
        backend=cfg.backend,
        model=str(cfg.model),
        objective_text=cfg.objective,
        top=args.top,
    )
    print(f"{'rank':>5} {'throughput':>12} {'ttft':>10} {'tpot':>10} {'errors':>8} args")
    for i, entry in enumerate(ranked, 1):
        summary = entry["summary"]
        args_str = " ".join(entry["run"].get("args", []))
        print(
            f"{i:>5} "
            f"{summary.get('median_throughput_tok_per_sec', 0.0) or 0.0:>12.2f} "
            f"{summary.get('median_ttft_ms', 0.0) or 0.0:>10.2f} "
            f"{summary.get('median_tpot_ms', 0.0) or 0.0:>10.2f} "
            f"{summary.get('error_count', 0):>4}/{summary.get('sample_count', 0):<4} "
            f"{args_str}"
        )
    return 0


def _dashboard(args: argparse.Namespace) -> int:
    run_dashboard(port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benched", description="Auto-tune llama.cpp and vLLM parameters")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run a sweep (config file or auto-sweep)")
    run_parser.add_argument("--config", help="Path to sweep config YAML (omit for auto-sweep)")
    run_parser.add_argument("--backend", choices=["llama-cpp", "vllm"],
                            help="Backend for auto-sweep (requires --model)")
    run_parser.add_argument("--model", help="Model path for auto-sweep (requires --backend)")
    run_parser.add_argument("--prompt-len", type=int, default=512,
                            help="Synthetic prompt token count (auto-sweep, default 512)")
    run_parser.add_argument("--gen-len", type=int, default=256,
                            help="Synthetic generation token count (auto-sweep, default 256)")
    run_parser.add_argument("--concurrency", type=int, default=4,
                            help="Concurrent requests (auto-sweep, default 4)")
    run_parser.add_argument("--total-requests", type=int, default=16,
                            help="Total requests per combo (auto-sweep, default 16)")
    run_parser.add_argument("--dry-run", action="store_true", help="Print configurations without running")
    run_parser.add_argument("--binary", help="Path to existing llama-server binary")
    run_parser.add_argument("--venv", help="Path to existing vLLM virtualenv")
    run_parser.add_argument("--mtp", action="store_true",
                            help="Enable MTP speculative decoding sweep (--spec-type draft-mtp --spec-draft-n-max 4)")
    run_parser.add_argument(
        "--continue-from", type=int, default=None, dest="continue_from",
        help="Resume from a failed run, skipping already successful configurations",
    )
    run_parser.set_defaults(func=_run)

    # list
    list_parser = subparsers.add_parser("list", help="List stored runs")
    list_parser.add_argument("--backend", choices=["llama-cpp", "vllm"])
    list_parser.add_argument("--model")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.set_defaults(func=_list)

    # show
    show_parser = subparsers.add_parser("show", help="Show a single run")
    show_parser.add_argument("run_id", type=int)
    show_parser.set_defaults(func=_show)

    # reset
    reset_parser = subparsers.add_parser("reset", help="Delete the entire database")
    reset_parser.set_defaults(func=_reset)

    # recommend
    rec_parser = subparsers.add_parser("recommend", help="Recommend top configurations")
    rec_parser.add_argument("--config", required=True)
    rec_parser.add_argument("--top", type=int, default=5)
    rec_parser.set_defaults(func=_recommend)

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch local results dashboard")
    dash_parser.add_argument("--port", type=int, default=8080)
    dash_parser.set_defaults(func=_dashboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
