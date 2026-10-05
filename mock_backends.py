"""
Mock llama-servers for end-to-end pipeline testing without GPUs.

Run this first, then start the proxy with default settings (it points at
localhost:8080/8081 out of the box), then fire curl requests at the proxy:

    python mock_backends.py                        # terminal 1
    python main.py                                 # terminal 2
    curl -N http://localhost:5000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"rp-two-stage","stream":true,"messages":[{"role":"user","content":"hi"}]}'

Stage-1 latency is configurable via MOCK_STAGE1_DELAY (seconds) so the
keepalive behaviour can be exercised with a slow "model".
"""

import asyncio
import json
import os

from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

STAGE1_DELAY = float(os.getenv("MOCK_STAGE1_DELAY", "1.0"))
# Set MOCK_STAGE2_REFUSE=1 to make the stage-2 mock emit a safety refusal so
# the proxy's refusal-fallback path can be tested end to end.
STAGE2_REFUSE = os.getenv("MOCK_STAGE2_REFUSE") == "1"
REFUSAL_TEXT = "I'm sorry, but I cannot assist with that request."

# Non-ASCII punctuation (curly quotes, em dash) is intentional: it exercises
# the UTF-8 passthrough path of the proxy's SSE re-serialization.
#
# MESSY_DRAFT mimics real stage-1 output from a reasoning model: leaked
# thinking blocks (both well-formed and malformed), a meta preamble line,
# the story body, then MVU-style functional blocks. It lets us verify that
# (a) the proxy strips thinking tags deterministically and (b) functional
# blocks survive the whole pipeline byte-identical.
FUNCTIONAL_TAIL = (
    "\n<UpdateVariable>\n<JSONPatch>\n"
    '[ { "op": "replace", "path": "/world/time", "value": "23:35" } ]'
    "\n</JSONPatch>\n</UpdateVariable>\n"
    "<scene>Desert camp - 23:35</scene>\n"
    "<StatusPlaceHolderImpl/>"
)

MESSY_DRAFT = (
    "<think>Plan: keep second person, then update variables.</think>\n"
    "<｜begin▁of▁thinking｜>Need maybe no markdown. Check prohibited words. "
    "Final answer now.<｜begin▁of▁thinking｜>\n"
    "Sure, here is the final output you requested:\n"
    '*She pushes the door open.* "We need to talk," she says.'
    + FUNCTIONAL_TAIL
)

POLISHED = (
    "*She eases the heavy door open, hinges sighing.* "
    "\u201cWe need to talk,\u201d she says \u2014 voice low."
    + FUNCTIONAL_TAIL
)


stage1 = FastAPI()
stage2 = FastAPI()


def _model_list(model_id: str):
    return {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "mock"}]}


# --- Stage 1 mock: slow non-streaming draft generator -----------------------
@stage1.get("/v1/models")
async def stage1_models():
    return _model_list("mock-qwen")


@stage1.post("/v1/chat/completions")
async def stage1_chat(body: dict = Body(...)):
    """Stream like a real reasoning server: prefill delay, content chunks
    (including leaked thinking tags so the proxy's strip logic is exercised),
    a finish chunk, then a usage chunk as stream_options requests."""
    if not body.get("stream"):
        await asyncio.sleep(STAGE1_DELAY)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": MESSY_DRAFT},
                    "finish_reason": "stop",
                }
            ]
        }

    async def gen():
        await asyncio.sleep(STAGE1_DELAY)  # simulate prefill latency
        words = MESSY_DRAFT.split(" ")
        for tok in words:
            chunk = {
                "id": "mock-s1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "mock-qwen",
                "choices": [{"index": 0, "delta": {"content": tok + " "}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)
        final = {
            "id": "mock-s1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-qwen",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        usage = {
            "id": "mock-s1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-qwen",
            "choices": [],
            "usage": {
                "prompt_tokens": 1234,
                "completion_tokens": len(words),
                "total_tokens": 1234 + len(words),
            },
        }
        yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# --- Stage 2 mock: token-by-token streaming polisher ------------------------
@stage2.get("/v1/models")
async def stage2_models():
    return _model_list("mock-gemma")


@stage2.post("/v1/chat/completions")
async def stage2_chat(body: dict = Body(...)):
    # Echo the received messages to this mock's stdout so tests can verify
    # what the proxy actually sent: system prompt present, user content with
    # reasoning blocks already stripped.
    print("STAGE2_RX " + json.dumps(body.get("messages", []), ensure_ascii=False), flush=True)
    if not body.get("stream"):
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": POLISHED},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 321,
                "completion_tokens": len(POLISHED.split()),
                "total_tokens": 321 + len(POLISHED.split()),
            },
        }

    async def gen():
        for tok in POLISHED.split(" "):
            chunk = {
                "id": "mock-chunk",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "mock-gemma",
                "choices": [{"index": 0, "delta": {"content": tok + " "}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)
        final = {
            "id": "mock-chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-gemma",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        usage = {
            "id": "mock-chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-gemma",
            "choices": [],
            "usage": {
                "prompt_tokens": 321,
                "completion_tokens": len(POLISHED.split()),
                "total_tokens": 321 + len(POLISHED.split()),
            },
        }
        yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _serve():
    cfg1 = uvicorn.Config(stage1, host="127.0.0.1", port=8080, log_level="warning")
    cfg2 = uvicorn.Config(stage2, host="127.0.0.1", port=8081, log_level="warning")
    await asyncio.gather(uvicorn.Server(cfg1).serve(), uvicorn.Server(cfg2).serve())


if __name__ == "__main__":
    asyncio.run(_serve())