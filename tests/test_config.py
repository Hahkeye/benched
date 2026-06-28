"""Tests for benched.config."""

import pytest
from pathlib import Path

from benched import config


def test_load_minimal_synthetic(tmp_path: Path) -> None:
    model = tmp_path / "dummy.gguf"
    model.write_text("")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
backend: llama-cpp
model: {model}
server:
  matrix:
    - - ["-t", "4"]
      - ["-t", "8"]
workload:
  kind: synthetic
  input_tokens: 32
  output_tokens: 16
  concurrent_requests: 1
objective: maximize throughput_tok_per_sec
"""
    )
    loaded = config.load_config(cfg_path)
    assert loaded.backend == "llama-cpp"
    assert loaded.model == model.resolve()
    assert len(loaded.server.matrix) == 1
    assert len(loaded.server.matrix[0]) == 2


def test_matrix_cartesian_product() -> None:
    dims = [
        [["-t", "4"], ["-t", "8"]],
        [["-cb"], ["--no-cont-batching"]],
    ]
    combos = config.matrix_combinations(dims)
    assert len(combos) == 4
    assert ["-t", "4", "-cb"] in combos
    assert ["-t", "8", "--no-cont-batching"] in combos


def test_empty_dimension_rejected(tmp_path: Path) -> None:
    model = tmp_path / "dummy.gguf"
    model.write_text("")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
backend: llama-cpp
model: {model}
server:
  matrix:
    - []
workload:
  kind: synthetic
  input_tokens: 32
  output_tokens: 16
  concurrent_requests: 1
"""
    )
    with pytest.raises(config.ConfigError):
        config.load_config(cfg_path)


def test_model_missing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
backend: llama-cpp
model: /does/not/exist.gguf
server:
  matrix:
    - - ["-t", "4"]
workload:
  kind: synthetic
  input_tokens: 32
  output_tokens: 16
  concurrent_requests: 1
"""
    )
    with pytest.raises(config.ConfigError):
        config.load_config(cfg_path)


def test_unknown_backend(tmp_path: Path) -> None:
    model = tmp_path / "dummy.gguf"
    model.write_text("")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
backend: foobar
model: {model}
server:
  matrix:
    - - ["-t", "4"]
workload:
  kind: synthetic
  input_tokens: 32
  output_tokens: 16
  concurrent_requests: 1
"""
    )
    with pytest.raises(config.ConfigError):
        config.load_config(cfg_path)


def test_objective_validation() -> None:
    assert config.parse_objective("maximize throughput_tok_per_sec").kind == "maximize"
    assert config.parse_objective("minimize ttft_ms").kind == "minimize"
    with pytest.raises(config.ConfigError):
        config.parse_objective("maximize unknown_metric")
