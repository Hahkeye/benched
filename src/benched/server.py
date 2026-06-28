"""Server lifecycle: start, health-check, stop a backend server."""

from __future__ import annotations

import asyncio
import signal
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx

from benched.builders.base import ServerPaths
from benched.config import Config, matrix_combinations, render_args_for_run


HEALTH_PATH = "/health"


class ServerStartupError(Exception):
    """Raised when a server fails to start or become healthy."""

    def __init__(self, message: str, stderr: str | None = None) -> None:
        super().__init__(message)
        self.stderr = stderr or ""


@dataclass
class ServerInstance:
    """A running backend server."""

    process: asyncio.subprocess.Process
    port: int
    base_url: str
    stderr_lines: list[str]


async def _read_stderr(proc: asyncio.subprocess.Process, buffer: list[str]) -> None:
    if proc.stderr is None:
        return
    while True:
        line = await proc.stderr.readline()
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        buffer.append(text)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


async def _wait_for_health(
    base_url: str,
    timeout: float = 120.0,
    interval: float = 0.5,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(HEALTH_PATH)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(interval)
    raise ServerStartupError(f"server did not become healthy within {timeout}s")


def _build_command(paths: ServerPaths, run_args: list[str]) -> list[str]:
    cmd: list[str] = [paths.executable]
    cmd.extend(paths.args)
    cmd.extend(run_args)
    return cmd


@asynccontextmanager
async def start_server(
    cfg: Config,
    paths: ServerPaths,
    combination: list[str],
    health_timeout: float = 120.0,
) -> AsyncIterator[ServerInstance]:
    """Start a backend server, yield once healthy, then tear it down."""
    port = _free_port()
    run_args = render_args_for_run(cfg, combination, port)
    cmd = _build_command(paths, run_args)

    print(f"[server] starting: {' '.join(cmd)}")
    stderr_buffer: list[str] = []
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ServerStartupError(f"executable not found: {cmd[0]}") from exc

    stderr_task = asyncio.create_task(_read_stderr(proc, stderr_buffer))
    instance = ServerInstance(
        process=proc,
        port=port,
        base_url=f"http://127.0.0.1:{port}",
        stderr_lines=stderr_buffer,
    )

    try:
        await _wait_for_health(instance.base_url, timeout=health_timeout)
        yield instance
    except Exception:
        # Capture last stderr lines to help diagnose startup failures.
        await asyncio.sleep(0.5)
        raise ServerStartupError(
            "server failed to start",
            stderr="\n".join(stderr_buffer[-50:]),
        )
    finally:
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass

        if proc.returncode is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                print("[server] process did not terminate; sending SIGKILL")
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()


def arg_signature(args: list[str]) -> dict[str, str]:
    """Return a display mapping for a rendered arg list (excluding model/host/port)."""
    result: dict[str, str] = {}
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("-m", "--model", "--host", "--port"):
            skip_next = True
            continue
        if i + 1 < len(args) and not args[i + 1].startswith("-"):
            result[arg] = args[i + 1]
            skip_next = True
        else:
            result[arg] = ""
    return result
