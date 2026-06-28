"""YAML/JSON configuration loading and validation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigError(Exception):
    """Raised when a configuration file is invalid."""


BACKENDS = {"llama-cpp", "vllm"}

LLAMA_PARAMS = {
    "-ngl",
    "-c",
    "-b",
    "-ub",
    "-np",
    "-cb",
    "--no-cont-batching",
    "-fa",
    "-ctk",
    "-ctv",
    "--mmap",
    "--no-mmap",
    "--mlock",
    "--no-mlock",
    "-sm",
    "-t",
    "-tb",
}

VLLM_PARAMS = {
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--gpu-memory-utilization",
    "--max-num-seqs",
    "--max-model-len",
    "--dtype",
    "--kv-cache-dtype",
    "--enable-prefix-caching",
    "--no-enable-prefix-caching",
    "--enforce-eager",
    "--no-enforce-eager",
    "--max-num-batched-tokens",
    "--enable-chunked-prefill",
    "--no-enable-chunked-prefill",
    "--optimization-level",
    "--performance-mode",
    "--stream-interval",
}

METRICS = {"throughput_tok_per_sec", "ttft_ms", "tpot_ms", "total_latency_ms"}


class ServerConfig(BaseModel):
    base_args: list[str] = Field(default_factory=list)
    matrix: list[list[list[str]]] = Field(default_factory=list)

    @field_validator("matrix")
    @classmethod
    def _matrix_non_empty(cls, value: list[list[list[str]]]) -> list[list[list[str]]]:
        for i, dim in enumerate(value):
            if not dim:
                raise ValueError(f"matrix dimension {i} is empty")
        return value


class SyntheticWorkload(BaseModel):
    kind: Literal["synthetic"]
    input_tokens: int = Field(..., ge=1)
    output_tokens: int = Field(..., ge=1)
    concurrent_requests: int = Field(..., ge=1)
    total_requests: int | None = Field(default=None, ge=1)
    warmup_requests: int = Field(default=0, ge=0)


class CustomWorkload(BaseModel):
    kind: Literal["custom"]
    prompt_file: Path
    concurrent_requests: int = Field(..., ge=1)
    total_requests: int | None = Field(default=None, ge=1)
    warmup_requests: int = Field(default=0, ge=0)


class Objective(BaseModel):
    kind: Literal["maximize", "minimize"]
    metric: str
    weight: float = 1.0


class Config(BaseModel):
    backend: str
    model: Path
    server: ServerConfig
    workload: SyntheticWorkload | CustomWorkload
    objective: str = "maximize throughput_tok_per_sec"

    @field_validator("backend")
    @classmethod
    def _valid_backend(cls, value: str) -> str:
        if value not in BACKENDS:
            raise ValueError(f"unsupported backend {value!r}; choose one of {sorted(BACKENDS)}")
        return value

    @model_validator(mode="after")
    def _model_exists(self) -> "Config":
        if not self.model.exists():
            raise ConfigError(f"model path does not exist: {self.model}")
        return self

    @model_validator(mode="after")
    def _validate_matrix_params(self) -> "Config":
        allowed = LLAMA_PARAMS if self.backend == "llama-cpp" else VLLM_PARAMS
        for dim_index, dim in enumerate(self.server.matrix):
            for opt_index, option in enumerate(dim):
                if not isinstance(option, list):
                    raise ConfigError(
                        f"matrix[{dim_index}][{opt_index}] must be a list of argument strings"
                    )
                for arg in option:
                    if not isinstance(arg, str):
                        raise ConfigError(
                            f"matrix[{dim_index}][{opt_index}] contains non-string {arg!r}"
                        )
                    if arg.startswith("-"):
                        name = arg.split("=", 1)[0]
                        if name not in allowed:
                            raise ConfigError(
                                f"unsupported parameter {name!r} for backend {self.backend!r}"
                            )
        return self


def parse_objective(text: str) -> Objective:
    """Parse an objective string such as 'maximize throughput_tok_per_sec'."""
    text = text.strip()
    parts = text.split(None, 1)
    if len(parts) != 2:
        raise ConfigError(f"objective must be '<maximize|minimize> <metric>', got {text!r}")
    kind, metric = parts
    if kind not in {"maximize", "minimize"}:
        raise ConfigError(f"objective kind must be maximize or minimize, got {kind!r}")
    if metric not in METRICS:
        raise ConfigError(f"unsupported objective metric {metric!r}; choose one of {sorted(METRICS)}")
    return Objective(kind=kind, metric=metric)


def parse_objectives(text: str) -> list[Objective]:
    """Parse a single objective or a weighted combo separated by commas."""
    objectives: list[Objective] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        # Simple weight syntax: "0.5 minimize ttft_ms"
        weight = 1.0
        tokens = part.split()
        if tokens[0].replace(".", "", 1).replace("-", "", 1).isdigit():
            weight = float(tokens[0])
            part = " ".join(tokens[1:])
        obj = parse_objective(part)
        obj.weight = weight
        objectives.append(obj)
    if not objectives:
        raise ConfigError(f"no valid objectives parsed from {text!r}")
    return objectives


def matrix_combinations(matrix: list[list[list[str]]]) -> list[list[str]]:
    """Return the cartesian product of all matrix dimensions."""
    if not matrix:
        return [[]]
    combos: list[tuple[list[str], ...]] = list(itertools.product(*matrix))
    return [list(itertools.chain.from_iterable(combo)) for combo in combos]


def load_config(path: str | Path) -> Config:
    """Load and validate a configuration file (YAML or JSON)."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data: dict[str, Any] = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config file must contain a top-level mapping")

    # Normalize legacy workload shorthand if needed.
    workload = data.get("workload", {})
    if isinstance(workload, dict) and "kind" not in workload:
        if "prompt_file" in workload:
            workload["kind"] = "custom"
        else:
            workload["kind"] = "synthetic"
        data["workload"] = workload

    try:
        cfg = Config.model_validate(data)
    except Exception as exc:
        raise ConfigError(_format_pydantic_error(exc)) from exc

    # Validate objective separately to give a clearer error.
    parse_objectives(cfg.objective)

    return cfg


def _format_pydantic_error(exc: Exception) -> str:
    if hasattr(exc, "errors"):
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            messages.append(f"{loc}: {err.get('msg', err)}")
        return "; ".join(messages)
    return str(exc)


def render_args_for_run(
    cfg: Config,
    combination: list[str],
    port: int,
) -> list[str]:
    """Build the full CLI argument list for a single run."""
    args: list[str] = list(cfg.server.base_args)
    args.extend(combination)
    if cfg.backend == "llama-cpp":
        args.extend(["-m", str(cfg.model)])
    elif cfg.backend == "vllm":
        args.extend(["--model", str(cfg.model)])
    args.extend(["--host", "127.0.0.1", "--port", str(port)])
    return args


def workload_summary(cfg: Config) -> dict[str, Any]:
    """Return a JSON-serializable summary of the workload settings."""
    w = cfg.workload
    out: dict[str, Any] = {"kind": w.kind}
    if isinstance(w, SyntheticWorkload):
        out.update(
            {
                "input_tokens": w.input_tokens,
                "output_tokens": w.output_tokens,
                "concurrent_requests": w.concurrent_requests,
                "total_requests": w.total_requests,
                "warmup_requests": w.warmup_requests,
            }
        )
    else:
        out.update(
            {
                "prompt_file": str(w.prompt_file),
                "concurrent_requests": w.concurrent_requests,
                "total_requests": w.total_requests,
                "warmup_requests": w.warmup_requests,
            }
        )
    return out


def arg_dict(args: list[str]) -> dict[str, str | bool]:
    """Convert a flat arg list to a readable dict for display/logging."""
    result: dict[str, str | bool] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("-"):
            i += 1
            continue
        # Stop before backend-specific injected args.
        if arg in ("-m", "--model", "--host", "--port"):
            i += 1
            if i < len(args):
                i += 1
            continue
        if i + 1 < len(args) and not args[i + 1].startswith("-"):
            result[arg] = args[i + 1]
            i += 2
        else:
            result[arg] = True
            i += 1
    return result
