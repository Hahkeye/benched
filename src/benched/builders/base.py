"""Abstract base builder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerPaths:
    """Paths required to start a backend server."""

    executable: str
    """Primary executable path or python -m invocation."""

    args: list[str]
    """Base arguments to prepend before per-run flags."""

    env: dict[str, str] | None = None
    """Optional environment overrides."""

    @property
    def is_python_module(self) -> bool:
        return self.executable.endswith("python") or self.executable.endswith("python3")


class Builder(ABC):
    """Builds or locates a backend server binary/venv."""

    @abstractmethod
    async def ensure(self) -> ServerPaths:
        """Ensure the server is built/available and return paths to run it."""
        ...

    def cache_dir(self) -> Path:
        return Path.home() / ".cache" / "benched"
