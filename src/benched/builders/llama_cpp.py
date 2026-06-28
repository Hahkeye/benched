"""llama.cpp source builder."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from benched.builders.base import Builder, ServerPaths


REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

GPU_BACKENDS = {"cuda", "vulkan", "rocm"}


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _detect_gpu() -> str | None:
    """Auto-detect available GPU backends; returns first found."""
    if _has_tool("nvcc"):
        return "cuda"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if _has_tool("vulkaninfo") or _has_tool("glslc"):
        return "vulkan"
    if _has_tool("hipcc") or Path("/opt/rocm/bin/hipcc").exists():
        return "rocm"
    return None


def _check_gpu_backend(backend: str) -> bool:
    """Return True if the specific GPU backend is available."""
    if backend == "cuda":
        if _has_tool("nvcc"):
            return True
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False
    if backend == "vulkan":
        return _has_tool("vulkaninfo") or _has_tool("glslc")
    if backend == "rocm":
        return _has_tool("hipcc") or Path("/opt/rocm/bin/hipcc").exists()
    return False


def _cmake_gpu_flags(gpu: str | None) -> list[str]:
    """Return cmake -D flags for the selected GPU backend, or CPU-only."""
    # Always start with CUDA=OFF (llama.cpp defaults to CUDA=ON when CUDA is
    # found by cmake, so we must be explicit).
    if gpu is None or gpu == "off":
        return ["-DLLAMA_CUDA=OFF"]
    if gpu == "cuda":
        return ["-DLLAMA_CUDA=ON"]
    if gpu == "vulkan":
        return ["-DLLAMA_CUDA=OFF", "-DLLAMA_VULKAN=ON"]
    if gpu == "rocm":
        return ["-DLLAMA_CUDA=OFF", "-DLLAMA_HIPBLAS=ON"]
    return ["-DLLAMA_CUDA=OFF"]


def _is_branch(repo: Path, ref: str) -> bool:
    """Return True if `ref` is a local branch."""
    result = asyncio.run(
        _run_cmd(
            "git",
            ["-C", str(repo), "rev-parse", "--verify", f"refs/heads/{ref}"],
            cwd=repo,
            check=False,
            capture_output=True,
        )
    )
    return result.returncode == 0


async def _run_cmd(
    name: str,
    args: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        name,
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE if capture_output else None,
        stderr=asyncio.subprocess.PIPE if capture_output else None,
    )
    stdout, stderr = await proc.communicate()
    if check and proc.returncode != 0:
        err = (stderr or b"").decode(errors="replace")[-2000:]
        raise RuntimeError(f"command failed: {name} {' '.join(args)}\n{err}")
    return proc

async def _run_cmd_live(
    name: str,
    args: list[str],
    cwd: Path | None = None,
) -> asyncio.subprocess.Process:
    """Run a command and stream its stdout/stderr to the terminal in real time."""
    proc = await asyncio.create_subprocess_exec(
        name,
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(f"  {line.decode(errors='replace').rstrip()}")
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {name} {' '.join(args)} (exit {proc.returncode})")
    return proc


class LlamaCppBuilder(Builder):
    def __init__(
        self,
        ref: str = "main",
        gpu: str = "auto",
        binary: str | Path | None = None,
    ) -> None:
        self.ref = ref
        self.gpu = gpu
        self.binary = Path(binary) if binary else None

    async def ensure(self) -> ServerPaths:
        if self.binary:
            if not self.binary.exists():
                raise FileNotFoundError(f"provided llama-server binary not found: {self.binary}")
            return ServerPaths(str(self.binary), [])

        base = self.cache_dir() / "llama.cpp"
        repo = base / self.ref
        repo.mkdir(parents=True, exist_ok=True)

        if not (repo / ".git").exists():
            print(f"[llama.cpp] cloning {REPO_URL} @ {self.ref}")
            await _run_cmd(
                "git", ["clone", "--depth", "1", "--branch", self.ref, REPO_URL, str(repo)]
            )
        elif _is_branch(repo, self.ref):
            print(f"[llama.cpp] pulling branch {self.ref}")
            await _run_cmd("git", ["-C", str(repo), "pull", "origin", self.ref])
        else:
            print(f"[llama.cpp] ref {self.ref} is not a branch; skipping pull")

        build_dir = repo / "build"
        server_bin = build_dir / "bin" / "llama-server"

        # Resolve GPU backend -------------------------------------------------
        gpu_backend: str | None
        if self.gpu == "off":
            gpu_backend = None
        elif self.gpu == "auto":
            gpu_backend = _detect_gpu()
            if gpu_backend is None:
                print("[llama.cpp] no GPU backend detected; building CPU-only")
            else:
                print(f"[llama.cpp] auto-detected GPU backend: {gpu_backend}")
        else:
            # User-specified backend (cuda / vulkan / rocm)
            if not _check_gpu_backend(self.gpu):
                raise RuntimeError(
                    f"[llama.cpp] requested GPU backend '{self.gpu}' is not available"
                    f" (checked tools: {_gpu_tools(self.gpu)})"
                )
            gpu_backend = self.gpu

        # Skip rebuild if binary is up to date --------------------------------
        source_mtime = max(
            (
                p.stat().st_mtime
                for p in repo.rglob("*")
                if p.is_file() and ".git" not in p.parts
            ),
            default=0.0,
        )
        bin_mtime = server_bin.stat().st_mtime if server_bin.exists() else 0.0

        if server_bin.exists() and bin_mtime >= source_mtime:
            print(f"[llama.cpp] using cached binary: {server_bin}")
            return ServerPaths(str(server_bin), [])

        # cmake configure -----------------------------------------------------
        print(f"[llama.cpp] configuring with cmake (gpu={gpu_backend or 'off'})")
        cmake_args = [
            "-B",
            str(build_dir),
            "-DLLAMA_BUILD_SERVER=ON",
            *_cmake_gpu_flags(gpu_backend),
        ]
        await _run_cmd_live("cmake", cmake_args, cwd=repo)

        # Build ---------------------------------------------------------------
        print("[llama.cpp] building llama-server")
        build_args = [
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "-j",
        ]
        await _run_cmd_live("cmake", build_args, cwd=repo)

        if not server_bin.exists():
            raise FileNotFoundError(f"llama-server binary not found after build: {server_bin}")
        return ServerPaths(str(server_bin), [])


def _gpu_tools(backend: str) -> str:
    """Return human-readable tool names for a GPU backend."""
    return {
        "cuda": "nvcc, torch.cuda",
        "vulkan": "vulkaninfo, glslc",
        "rocm": "hipcc, /opt/rocm/bin/hipcc",
    }.get(backend, "unknown")
