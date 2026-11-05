"""Tiny OpenAI-compatible streaming server for testing the benchmark offline.

    uvicorn tests.mock_server:app --port 8000
"""
import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(body: dict):
    n = int(body.get("max_tokens", 32))

    async def gen():
        await asyncio.sleep(0.05)  # fake prefill
        for i in range(n):
            await asyncio.sleep(0.01)  # fake decode step
            chunk = {"choices": [{"delta": {"content": f"tok{i} "}}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        usage = {"choices": [], "usage": {"completion_tokens": n}}
        yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
