"""Synthetic and custom prompt workloads."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from benched.client import Sample, chat_completion
from benched.config import Config, CustomWorkload, SyntheticWorkload


FRAGMENT = "hello "


def _approximate_tokens(text: str) -> int:
    return len(text.split())


def _synthetic_messages(input_tokens: int) -> list[dict[str, Any]]:
    text = ""
    # Over-allocate slightly then trim.
    while _approximate_tokens(text) < input_tokens:
        text += FRAGMENT
    while _approximate_tokens(text) > input_tokens:
        text = text.rsplit(" ", 1)[0]
    return [{"role": "user", "content": text}]


@dataclass
class WorkloadRequest:
    messages: list[dict[str, Any]]
    max_tokens: int


class Workload(AsyncIterator[WorkloadRequest]):
    """Base workload iterator."""

    def __init__(self, total_requests: int) -> None:
        self.total_requests = total_requests
        self.issued = 0

    def __aiter__(self) -> "Workload":
        return self

    async def __anext__(self) -> WorkloadRequest:
        if self.issued >= self.total_requests:
            raise StopAsyncIteration
        self.issued += 1
        return self._next_request()

    def _next_request(self) -> WorkloadRequest:
        raise NotImplementedError


class SyntheticWorkloadIterator(Workload):
    def __init__(self, cfg: SyntheticWorkload) -> None:
        total = cfg.total_requests if cfg.total_requests is not None else cfg.concurrent_requests * 4
        super().__init__(total)
        self.messages = _synthetic_messages(cfg.input_tokens)
        self.max_tokens = cfg.output_tokens

    def _next_request(self) -> WorkloadRequest:
        return WorkloadRequest(messages=self.messages, max_tokens=self.max_tokens)


class CustomWorkloadIterator(Workload):
    def __init__(self, cfg: CustomWorkload) -> None:
        entries = []
        with open(cfg.prompt_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        if not entries:
            raise ValueError(f"custom workload file is empty: {cfg.prompt_file}")
        self.entries = entries
        total = cfg.total_requests if cfg.total_requests is not None else len(entries)
        super().__init__(total)

    def _next_request(self) -> WorkloadRequest:
        entry = self.entries[self.issued % len(self.entries)]
        return WorkloadRequest(
            messages=entry["messages"],
            max_tokens=entry["max_tokens"],
        )


def build_workload(cfg: Config) -> Workload:
    if isinstance(cfg.workload, SyntheticWorkload):
        return SyntheticWorkloadIterator(cfg.workload)
    return CustomWorkloadIterator(cfg.workload)


def total_requests(cfg: Config) -> int:
    w = cfg.workload
    if isinstance(w, SyntheticWorkload):
        return w.total_requests if w.total_requests is not None else w.concurrent_requests * 4
    entries = sum(1 for _ in open(w.prompt_file) if _.strip())
    return w.total_requests if w.total_requests is not None else entries


async def run_workload(
    cfg: Config,
    base_url: str,
    on_sample: Any | None = None,
) -> list[Sample]:
    """Run the configured workload against a healthy server."""
    workload = build_workload(cfg)
    concurrency = cfg.workload.concurrent_requests
    samples: list[Sample] = []

    async def _task(req: WorkloadRequest) -> Sample:
        sample = await chat_completion(
            base_url=base_url,
            messages=req.messages,
            max_tokens=req.max_tokens,
        )
        samples.append(sample)
        if on_sample:
            await on_sample(sample)
        return sample

    pending: set[asyncio.Task] = set()
    async for req in workload:
        if len(pending) >= concurrency:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                await task
        pending.add(asyncio.create_task(_task(req)))

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    return samples


async def run_warmup(
    cfg: Config,
    base_url: str,
) -> list[Sample]:
    """Issue a small number of warmup requests and discard results."""
    count = cfg.workload.warmup_requests
    if count <= 0:
        return []
    w = cfg.workload
    messages: list[dict[str, Any]]
    max_tokens: int
    if isinstance(w, SyntheticWorkload):
        messages = _synthetic_messages(min(w.input_tokens, 64))
        max_tokens = min(w.output_tokens, 16)
    else:
        with open(w.prompt_file, "r", encoding="utf-8") as fh:
            entry = json.loads(next(fh))
        messages = entry["messages"]
        max_tokens = entry.get("max_tokens", 16)

    samples = []
    for _ in range(count):
        samples.append(
            await chat_completion(
                base_url=base_url,
                messages=messages,
                max_tokens=max_tokens,
            )
        )
    return samples
