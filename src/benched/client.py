"""OpenAI-compatible streaming client and benchmark metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Sample:
    """Metrics for a single chat-completion request."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    total_latency_ms: float = 0.0
    throughput_tok_per_sec: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.completion_tokens is None:
            self.completion_tokens = 0
        ct = self.completion_tokens or 0
        denom = max(ct - 1, 1)
        if self.total_latency_ms > self.ttft_ms:
            self.tpot_ms = (self.total_latency_ms - self.ttft_ms) / denom
        else:
            self.tpot_ms = self.ttft_ms / denom if ct <= 1 else 0.0
        if self.total_latency_ms > 0:
            self.throughput_tok_per_sec = ct / (self.total_latency_ms / 1000.0)
        else:
            self.throughput_tok_per_sec = 0.0


async def chat_completion(
    base_url: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    api_key: str = "dummy",
    model: str | None = None,
) -> Sample:
    """Send a streaming chat-completion request and return timing metrics."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    # Always include a model name — both llama.cpp and vLLM require it.
    model_name = model or "default"
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "model": model_name,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    start = time.perf_counter()
    first_chunk_time: float | None = None
    total_chunks = 0
    content_chunks = 0
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}
    error: str | None = None

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    error = f"HTTP {resp.status_code}: {body.decode(errors='replace')[:500]}"
                    return Sample(error=error)

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()

                    choice = chunk.get("choices", [{}])[0]
                    # Count all chunks for debugging
                    total_chunks += 1

                    # OpenAI streaming format: choices[0].delta.content
                    # Some llama.cpp versions use choices[0].text
                    content = (
                        choice.get("delta", {}).get("content")
                        or choice.get("text")
                    )
                    if content:
                        content_chunks += 1

                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage = chunk_usage
                    # Final chunk may carry usage but no delta content.
                    if chunk.get("choices") and not choice.get("delta"):
                        if choice.get("finish_reason"):
                            final_usage = choice.get("usage")
                            if final_usage:
                                usage = final_usage
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    total_latency_ms = (time.perf_counter() - start) * 1000.0
    ttft_ms = (first_chunk_time - start) * 1000.0 if first_chunk_time else total_latency_ms

    print(f"  [client] result: total_chunks={total_chunks}, content_chunks={content_chunks}, "
          f"usage={usage}, error={error}")

    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    else:
        prompt_tokens = None
        completion_tokens = content_chunks

    if error:
        return Sample(error=error)

    return Sample(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        metadata=metadata,
    )
