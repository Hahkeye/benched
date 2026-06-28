# benched

Auto-tuning harness for [llama.cpp](https://github.com/ggml-org/llama.cpp) and [vLLM](https://github.com/vllm-project/vllm) inference servers. Runs OpenAI-compatible chat-completion benchmarks across a matrix of performance parameters, persists results in SQLite, and recommends optimal configurations.

## Install

```bash
git clone git@github.com:Hahkeye/benched.git
cd benched
pip install -e .          # core dependencies only
pip install -e ".[dev]"   # with dev dependencies (tests)
pip install -e ".[dashboard]"  # with web dashboard
```

### 1. Build a server binary

```bash
# llama.cpp — CPU
benched build llama

# llama.cpp — with GPU (Vulkan, CUDA, or ROCm)
benched build llama --gpu cuda

# Skip source build and use an existing binary
benched build llama --binary /path/to/llama-server
```

```bash
# vLLM — pip install into a venv
benched build vllm --ref main

# Skip source build and use an existing venv
benched build vllm --venv /path/to/venv
```

### 2. Configure a sweep

Create a YAML config (see `examples/`). Examples:

```yaml
# examples/sweep_cpu.yaml
backend: llama-cpp
model: /path/to/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
server:
  matrix:
    - - ["-t", "4"]
      - ["-t", "8"]
    - - ["-ngl", "0"]
      - ["-ngl", "10"]
    - - ["-cb"]
      - ["--no-cont-batching"]
workload:
  kind: synthetic
  input_tokens: 256
  output_tokens: 128
  concurrent_requests: 2
  total_requests: 8
objective: maximize throughput_tok_per_sec
```

```yaml
# examples/sweep_gpu.yaml
backend: vllm
model: /path/to/model
server:
  matrix:
    - - ["--gpu-memory-utilization", "0.85"]
      - ["--gpu-memory-utilization", "0.95"]
    - - ["--max-num-seqs", "64"]
      - ["--max-num-seqs", "256"]
    - - ["--enforce-eager"]
      - []
workload:
  kind: synthetic
  input_tokens: 512
  output_tokens: 256
  concurrent_requests: 4
  total_requests: 16
objective: maximize throughput_tok_per_sec
```

### 3. Dry-run first

```bash
benched run --config examples/sweep_cpu.yaml --dry-run
```

Prints every cartesian combination without starting a server.

### 4. Run the sweep

```bash
benched run --config examples/sweep_cpu.yaml --model /path/to/your/model.gguf
```

Each combination starts the server, runs the workload, records samples, then stops. Results go into `~/.local/share/benched/benched.db`. Live progress is printed as runs complete.

### 5. Review results

```bash
# List all runs
benched list

# Show a single run with sample histograms
benched show <run_id>

# Recommend top configurations
benched recommend --config examples/sweep_cpu.yaml --top 5
```

### 6. Launch the dashboard

```bash
benched dashboard --port 8080
```

Open `http://127.0.0.1:8080` to browse runs, view TTFT/TPOT/throughput histograms, and compare configurations on a scatter plot.

## CLI reference

| Command | Description |
|---|---|
| `benched build llama` | Build llama.cpp server from source |
| `benched build vllm` | Build vLLM from source |
| `benched run --config <file>` | Execute a sweep |
| `benched list` | List stored runs |
| `benched show <run_id>` | Show a single run |
| `benched recommend --config <file>` | Rank top configurations |
| `benched dashboard` | Launch the web UI |

### Build options

**llama.cpp:**
- `--gpu {auto,cuda,vulkan,rocm,off}` — GPU backend (default: auto-detect)
- `--ref <branch>` — git ref to clone (default: `main`)
- `--binary <path>` — skip source build, use existing binary

**vLLM:**
- `--ref <branch>` — git ref to clone (default: `main`)
- `--venv <path>` — skip source build, use existing virtualenv

### Run options

- `--model <path>` — override model path from config
- `--dry-run` — print configurations without running
- `--continue-from <run_id>` — resume after a failure, skipping successful runs
- `--gpu {auto,cuda,vulkan,rocm,off}` — GPU backend for llama.cpp builds

## GPU backends

| Flag | Backend | cmake flag | Detection |
|---|---|---|---|
| `cuda` | CUDA | `-DLLAMA_CUDA=ON` | `nvcc` or `torch.cuda` |
| `vulkan` | Vulkan | `-DLLAMA_VULKAN=ON` | `vulkaninfo` or `glslc` |
| `rocm` | ROCm | `-DLLAMA_HIPBLAS=ON` | `hipcc` or `/opt/rocm/bin/hipcc` |
| `off` | CPU-only | none | — |
| `auto` | first found | — | probes in order: cuda → vulkan → rocm → cpu |

## Config format

See [`examples/`](examples/) for complete examples. The configuration file is YAML with these top-level keys:

| Key | Description |
|---|---|
| `backend` | `llama-cpp` or `vllm` |
| `model` | Path to GGUF (llama.cpp) or HF model (vLLM) |
| `server.base_args` | Arguments common to every run |
| `server.matrix` | List of parameter dimensions to sweep (cartesian product) |
| `workload` | `synthetic` or `custom` workload definition |
| `objective` | Metric to optimize, e.g. `maximize throughput_tok_per_sec` |

### Matrix format

Each dimension in `server.matrix` is a list of CLI argument fragments. The harness concatenates one fragment from each dimension to produce the full arg list for a run.

```yaml
server:
  matrix:
    # Dimension 1: batch size
    - - ["-b", "512"]
      - ["-b", "1024"]
    # Dimension 2: contiguous batching (boolean)
    - - ["-cb"]
      - ["--no-cont-batching"]
```

Use an empty list `[]` to represent "no args" for a toggle (e.g. `--enforce-eager` vs default).

## Tests

```bash
pip install "benched[dev]"
pytest tests/
```

## Data location

Results are stored in SQLite at `~/.local/share/benched/benched.db`.
