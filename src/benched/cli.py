"""Command-line interface for benched."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

from benched.builders import LlamaCppBuilder, VllmBuilder
from benched.config import ConfigError, load_config
from benched.dashboard import run_dashboard
from benched.db import Database
from benched.objectives import recommend
from benched.runner import list_results, run_sweep, show_run


def _build_llama(args: argparse.Namespace) -> int:
    builder = LlamaCppBuilder(
        ref=args.ref,
        gpu=args.gpu,
        binary=args.binary,
    )


def _build_vllm(args: argparse.Namespace) -> int:
    builder = VllmBuilder(
        ref=args.ref,
        venv=args.venv,
    )
    paths = asyncio.run(builder.ensure())
    print(paths.executable)
    return 0


def _run(args: argparse.Namespace) -> int:
    try:
        asyncio.run(
            run_sweep(
                args.config,
                dry_run=args.dry_run,
                model_override=args.model,
                continue_from=args.continue_from,
                ref=args.ref,
                gpu=args.gpu,
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
    print(f"{'id':>6} {'backend':<12} {'model':<30} {'status':<10} {'throughput':>12} {'errors':>10}")
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

    # build llama
    build_llama = subparsers.add_parser("build", help="Build a backend")
    build_sub = build_llama.add_subparsers(dest="backend", required=True)

    llama_parser = build_sub.add_parser("llama", help="Build llama.cpp server")
    llama_parser.add_argument("--ref", default="main")
    llama_parser.add_argument("--gpu", choices=["auto", "cuda", "vulkan", "rocm", "off"], default="auto",
                              help="GPU backend: auto (detect), cuda, vulkan, rocm, or off (CPU-only)")
    llama_parser.add_argument("--binary", help="Path to existing llama-server binary")
    llama_parser.set_defaults(func=_build_llama)

    vllm_parser = build_sub.add_parser("vllm", help="Build vLLM")
    vllm_parser.add_argument("--ref", default="main")
    vllm_parser.add_argument("--venv", help="Path to existing venv containing vLLM")
    vllm_parser.set_defaults(func=_build_vllm)

    # run
    run_parser = subparsers.add_parser("run", help="Run a sweep")
    run_parser.add_argument("--config", required=True, help="Path to sweep config YAML")
    run_parser.add_argument("--model", help="Override model path")
    run_parser.add_argument("--dry-run", action="store_true", help="Print configurations without running")
    run_parser.add_argument("--ref", default="main", help="Git ref for source builds")
    run_parser.add_argument("--gpu", choices=["auto", "cuda", "vulkan", "rocm", "off"], default="auto",
                            help="GPU backend for llama.cpp builds: auto (detect), cuda, vulkan, rocm, off")
    run_parser.add_argument("--binary", help="Existing llama-server binary")
    run_parser.add_argument("--venv", help="Existing vLLM virtualenv")
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
