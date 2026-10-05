"""
Pure API relay: SillyTavern -> Stage 1 (content model) -> Stage 2 (style model) -> ST.

No prompt injection, no message rewriting. Stage 1 receives ST's request
verbatim; stage 2 receives stage-1 output as a single user message; the
streamed polish is forwarded chunk-by-chunk to ST.
"""

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
    looks_like_refusal,
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


@app.get("/v1/models")
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
    raw_text = await run_stage1(request)
    if not raw_text or not raw_text.strip():
        raise HTTPException(502, "Stage 1 returned empty response")

    payload = build_stage2_payload(request, raw_text, stream=False)
    polished = await call_nonstreaming(settings.stage2_url, payload, settings.stage2_timeout)
    # Empty or refusal-looking polish output is discarded: serving the raw
    # stage-1 draft is strictly better than serving nothing or a refusal.
    # Only the opening is scanned for refusal markers (same rationale as the
    # streaming path: markers can legitimately appear mid-story).
    if polished and polished.strip() and not looks_like_refusal(
        polished[: settings.refusal_check_chars]
    ):
        final = polished
    else:
        if polished and polished.strip():
            logger.warning("Stage 2 refused; serving raw stage-1 text")
        final = raw_text

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