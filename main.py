"""
Pure API relay: SillyTavern -> Stage 1 (content model) -> Stage 2 (style model) -> ST.

No prompt injection, no message rewriting. Stage 1 receives ST's request
verbatim; stage 2 receives stage-1 output as a single user message; the
streamed polish is forwarded chunk-by-chunk to ST.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from config import settings
from pipeline import (
    build_stage2_payload,
    call_nonstreaming,
    log_request_summary,
    run_stage1,
    stream_stage2,
)
from schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelListResponse,
)

logger = logging.getLogger("uvicorn.error")
logger.setLevel(settings.log_level)


# ---------------------------------------------------------------------------
# Startup probe: fail loudly (in logs) if either backend is misconfigured
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Probe both backends at startup.

    Why /v1/models and not /health: llama-server exposes /health at the
    server ROOT, not under /v1, so probing "<base>/health" (where base
    includes /v1) always 404s. /v1/models is part of the OpenAI-compatible
    surface we actually depend on, making it the reliable liveness check.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for name, url in [
            ("Stage 1", settings.stage1_url),
            ("Stage 2", settings.stage2_url),
        ]:
            try:
                resp = await client.get(f"{url}/models")
                resp.raise_for_status()
                logger.info("%s backend OK: %s", name, url)
            except Exception as exc:
                logger.warning(
                    "%s backend unreachable (%s): %s — proxy starts anyway, "
                    "but requests will fail until it is up.",
                    name, url, exc,
                )
    yield


app = FastAPI(title="AI-RP-Proxy", version="0.3.0", lifespan=lifespan)


# Alias routes without the /v1 prefix: OpenAI clients disagree on whether
# the base URL should include it (SillyTavern appends /models to whatever
# base URL is configured, so forgetting the suffix must not 404).
@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """ST probes this endpoint when testing the connection."""
    return ModelListResponse(
        data=[
            ModelInfo(
                id=settings.virtual_model_name,
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    # --- Streaming path (ST's default) ---
    if request.stream:
        return StreamingResponse(
            stream_stage2(request, raw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Disable proxy buffering if nginx/traefik ever sits in front.
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-streaming fallback (useful for curl tests and simple clients) ---
    # Stage 1 runs here in the endpoint, so its failures surface as a clean
    # HTTP 502 instead of a broken stream.
    stats: dict = {}
    split: dict | None = {"body_ready": asyncio.Event()} if settings.stage1_tail_split else None
    raw_text = await run_stage1(request, stats, split)
    if not raw_text or not raw_text.strip():
        raise HTTPException(502, "Stage 1 returned empty response")

    # When the tail split fired, stage 2 polishes only the story body and the
    # functional tail is appended verbatim; otherwise polish everything.
    if split is not None and split.get("marker_found"):
        body_text = split["body"]
        tail_text = split.get("tail", "")
        logger.info(
            "Stage 1: tail split — %d body chars to polish, %d tail chars appended raw",
            len(body_text), len(tail_text),
        )
    else:
        body_text, tail_text = raw_text, ""

    payload = build_stage2_payload(request, body_text, stream=False)
    usage_out: dict = {}
    s2_start = time.monotonic()
    polished = await call_nonstreaming(
        settings.stage2_url, payload, settings.stage2_timeout, usage_out
    )
    stats["stage2"] = {
        "total_s": time.monotonic() - s2_start,
        "prompt_tokens": usage_out.get("prompt_tokens"),
        "completion_tokens": usage_out.get("completion_tokens"),
    }
    # If the polish pass comes back empty for any reason, the raw draft is
    # strictly better than returning nothing.
    final = (polished if polished and polished.strip() else body_text) + tail_text
    log_request_summary(stats)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=settings.virtual_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final),
                finish_reason="stop",
            )
        ],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Convenience runner: `python main.py` works without remembering uvicorn flags.
    import uvicorn

    uvicorn.run(app, host=settings.proxy_host, port=settings.proxy_port)