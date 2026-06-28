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

_WIN = sys.platform == "win32"


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
    """Run a command, streaming stdout+stderr to the terminal in real time."""
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
    return venv / ("Scripts" if _WIN else "bin") / "python.exe" if _WIN else venv / "bin" / "python"


def _venv_pip(venv: Path) -> Path:
    return venv / ("Scripts" if _WIN else "bin") / "pip.exe" if _WIN else venv / "bin" / "pip"


def _dist_info_records(venv: Path) -> list[str]:
    return glob.glob(
        str(venv / "lib" / "python*" / "site-packages" / "vllm-*.dist-info" / "RECORD")
    )


class VllmBuilder(Builder):
    def __init__(
        self,
        ref: str = "main",
        venv: str | Path | None = None,
        wheel: bool = False,
    ) -> None:
        self.ref = ref
        self.venv = Path(venv) if venv else None
        self.wheel = wheel

    async def ensure(self) -> ServerPaths:
        if self.venv:
            py = _venv_python(self.venv)
            if not py.exists():
                raise FileNotFoundError(f"provided venv python not found: {py}")
            return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])

        uv = shutil.which("uv")
        base = self.cache_dir() / "vllm"
        repo = base / self.ref
        venv = repo / "venv"
        py = _venv_python(venv)

        # ── already installed? ─────────────────────────────────────────────
        if py.exists() and _dist_info_records(venv):
            if self.wheel:
                print("[vllm] vllm wheel already installed, skipping")
                return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])
            if (repo / ".git").exists():
                commit_ts = await _git_commit_timestamp(repo)
                last_install = max(os.stat(p).st_mtime for p in _dist_info_records(venv))
                if last_install >= commit_ts:
                    print("[vllm] vllm installation is up to date")
                    return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])

        # ── wheel install (no clone) ───────────────────────────────────────
        if self.wheel:
            print("[vllm] installing vllm wheel")
            if not py.exists():
                if uv:
                    await _run_cmd("uv", ["venv", str(venv)])
                else:
                    await _run_cmd(sys.executable, ["-m", "venv", str(venv)])
            if uv:
                await _run_cmd_live(
                    "uv", ["pip", "install", "--python", str(py), "vllm", "--torch-backend", "auto"],
                    cwd=self.cache_dir(),
                )
            else:
                if not _venv_pip(venv).exists():
                    await _run_cmd(
                        str(py), ["-m", "ensurepip", "--upgrade"],
                        cwd=self.cache_dir(), check=False,
                    )
                await _run_cmd_live(str(py), ["-m", "pip", "install", "vllm"], cwd=self.cache_dir())
            return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])

        # ── source build (clone + pip install -e .) ────────────────────────
        repo.mkdir(parents=True, exist_ok=True)
        if not (repo / ".git").exists():
            print(f"[vllm] cloning {REPO_URL} @ {self.ref}")
            await _run_cmd(
                "git", ["clone", "--depth", "1", "--branch", self.ref, REPO_URL, str(repo)]
            )
        else:
            print(f"[vllm] pulling branch {self.ref}")
            await _run_cmd("git", ["-C", str(repo), "pull", "origin", self.ref])

        if not py.exists():
            if uv:
                print("[vllm] creating virtual environment with uv")
                await _run_cmd("uv", ["venv", str(venv)])
            else:
                print("[vllm] creating virtual environment")
                await _run_cmd(sys.executable, ["-m", "venv", str(venv)])

        if uv:
            print("[vllm] installing package with uv (this may take a while)")
            await _run_cmd_live(
                "uv", ["pip", "install", "--python", str(py), "-e", "."], cwd=repo
            )
        else:
            print("[vllm] installing package into venv (this may take a while)")
            if not _venv_pip(venv).exists():
                await _run_cmd(str(py), ["-m", "ensurepip", "--upgrade"], cwd=repo, check=False)
            await _run_cmd_live(str(py), ["-m", "pip", "install", "-e", "."], cwd=repo)

        return ServerPaths(str(py), ["-m", "vllm.entrypoints.openai.api_server"])
