"""Concurrency-sweep benchmark for any OpenAI-compatible chat endpoint
(vLLM, SGLang, Nebius Token Factory, ...).

Measures per request: time-to-first-token (TTFT), end-to-end latency,
output tokens and decode speed. Aggregates per concurrency level:
p50/p95 latency, p50/p95 TTFT, request rate and output tokens/s.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from bench.prompts import build_prompt


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float = 0.0
    latency_s: float = 0.0
    output_tokens: int = 0
    error: str = ""


async def one_request(client: httpx.AsyncClient, url: str, headers: dict,
                      model: str, prompt: str, max_tokens: int) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    chunks = 0
    usage_tokens = None
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:200]
                return RequestResult(False, error=f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        chunks += 1
    except Exception as exc:  # network errors, timeouts
        return RequestResult(False, error=f"{type(exc).__name__}: {exc}")

    latency = time.perf_counter() - start
    if ttft is None:
        return RequestResult(False, error="no tokens received")
    # Prefer server-reported usage; fall back to streamed chunk count.
    tokens = usage_tokens if usage_tokens is not None else chunks
    return RequestResult(True, ttft, latency, tokens)


def pct(values: list, q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[idx]


async def run_level(args, concurrency: int) -> dict:
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env, "")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=concurrency * 2)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def guarded(i: int) -> RequestResult:
            async with sem:
                return await one_request(client, url, headers, args.model,
                                         build_prompt(args.prompt_words, i),
                                         args.max_tokens)

        # warmup (not measured)
        for i in range(args.warmup):
            await one_request(client, url, headers, args.model,
                              build_prompt(args.prompt_words, -1 - i), 16)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(args.num_requests)))
        wall = time.perf_counter() - wall_start

    good = [r for r in results if r.ok]
    errors = [r.error for r in results if not r.ok]
    total_tokens = sum(r.output_tokens for r in good)
    decode_speeds = [
        r.output_tokens / (r.latency_s - r.ttft_s)
        for r in good if r.latency_s > r.ttft_s and r.output_tokens > 1
    ]
    return {
        "concurrency": concurrency,
        "requests": args.num_requests,
        "succeeded": len(good),
        "failed": len(errors),
        "first_error": errors[0] if errors else "",
        "wall_s": round(wall, 3),
        "req_per_s": round(len(good) / wall, 3) if wall else 0,
        "output_tok_per_s": round(total_tokens / wall, 2) if wall else 0,
        "ttft_p50_s": round(pct([r.ttft_s for r in good], 0.50), 4),
        "ttft_p95_s": round(pct([r.ttft_s for r in good], 0.95), 4),
        "latency_p50_s": round(pct([r.latency_s for r in good], 0.50), 4),
        "latency_p95_s": round(pct([r.latency_s for r in good], 0.95), 4),
        "decode_tok_per_s_per_req_p50": round(statistics.median(decode_speeds), 2)
        if decode_speeds else 0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="Environment variable that holds the API key (if any)")
    p.add_argument("--concurrency", default="1,2,4,8,16",
                   help="Comma-separated concurrency levels to sweep")
    p.add_argument("--num-requests", type=int, default=32,
                   help="Requests per concurrency level")
    p.add_argument("--prompt-words", type=int, default=200,
                   help="Approximate prompt length in words")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--label", default="run",
                   help="Label stored with results (e.g. fp16, awq-int4)")
    p.add_argument("--output", default="results/results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    rows = []
    for c in levels:
        print(f"[{args.label}] concurrency={c} ...", flush=True)
        row = asyncio.run(run_level(args, c))
        rows.append(row)
        print(f"  {row['output_tok_per_s']} tok/s | TTFT p95 {row['ttft_p95_s']}s | "
              f"latency p95 {row['latency_p95_s']}s | failed {row['failed']}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "config": {k: getattr(args, k) for k in
                   ("num_requests", "prompt_words", "max_tokens", "warmup")},
        "results": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
