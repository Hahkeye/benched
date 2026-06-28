"""vLLM source builder."""

from __future__ import annotations

import asyncio
import glob
import os
import shutil
import sys
from pathlib import Path

from benched.builders.base import Builder, ServerPaths


REPO_URL = "https://github.com/vllm-project/vllm.git"


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


async def _git_commit_timestamp(repo: Path) -> float:
    proc = await _run_cmd(
        "git",
        ["-C", str(repo), "log", "-1", "--format=%ct"],
        cwd=repo,
        capture_output=True,
    )
    out = (await proc.stdout.read()).decode().strip()
    return float(out) if out else 0.0


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

def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"

def _venv_pip(venv: Path) -> Path:
    """Return the path to pip inside a virtual environment."""
    if sys.platform == "win32":
        return venv / "Scripts" / "pip.exe"
    return venv / "bin" / "pip"


class VllmBuilder(Builder):
    def __init__(
        self,
        ref: str = "main",
        venv: str | Path | None = None,
    ) -> None:
        self.ref = ref
        self.venv = Path(venv) if venv else None

    async def ensure(self) -> ServerPaths:
        if self.venv:
            py = _venv_python(self.venv)
            if not py.exists():
                raise FileNotFoundError(f"provided venv python not found: {py}")
            return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])

        base = self.cache_dir() / "vllm"
        repo = base / self.ref
        venv = base / self.ref / "venv"
        repo.mkdir(parents=True, exist_ok=True)

        if not (repo / ".git").exists():
            print(f"[vllm] cloning {REPO_URL} @ {self.ref}")
            await _run_cmd("git", ["clone", "--depth", "1", "--branch", self.ref, REPO_URL, str(repo)])
        else:
            print(f"[vllm] pulling branch {self.ref}")
            await _run_cmd("git", ["-C", str(repo), "pull", "origin", self.ref])

        py = _venv_python(venv)
        if not py.exists():
            print("[vllm] creating virtual environment")
            await _run_cmd(sys.executable, ["-m", "venv", str(venv)])

        # Ensure pip is available in the venv ---------------------------------
        if not _venv_pip(venv).exists():
            print("[vllm] pip not found in venv; running ensurepip")
            try:
                await _run_cmd(str(py), ["-m", "ensurepip", "--upgrade"], cwd=repo, check=True)
            except RuntimeError:
                print("[vllm] ensurepip failed; falling back to system pip bootstrap")
                # Attempt to symlink/copy pip from system into the venv
                await _run_cmd(
                    sys.executable, ["-m", "pip", "install", "--target", str(py.parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"), "pip"],
                    cwd=repo,
                )
        # Decide whether re-install is needed by comparing checkout time to dist-info.
        need_install = True
        commit_ts = await _git_commit_timestamp(repo)
        dist_info_pattern = str(venv / "lib" / "python*" / "site-packages" / "vllm-*.dist-info" / "RECORD")
        records = glob.glob(dist_info_pattern)
        if records:
            record_mtime = max(os.stat(p).st_mtime for p in records)
            if record_mtime >= commit_ts:
                print("[vllm] vllm installation is up to date")
                need_install = False

        if need_install:
            print("[vllm] installing package into venv (this may take a while)")
            await _run_cmd_live(str(py), ["-m", "pip", "install", "--upgrade", "pip"], cwd=repo)
            await _run_cmd_live(str(py), ["-m", "pip", "install", "-e", "."], cwd=repo)

        return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])
