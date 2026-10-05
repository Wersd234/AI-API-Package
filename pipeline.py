"""
Two-stage pipe with zero prompt injection.

Stage 1: forward SillyTavern's request verbatim to the content model
         (non-streaming, hidden from the user).
Stage 2: feed stage-1 output as a single user message to the style model
         (streaming), forwarding SSE chunks back to SillyTavern.

Transparency rules:
- Every parameter ST sends (sampling, penalties, vendor extras) is forwarded
  unchanged to stage 1.
- Stage 2 inherits the same parameters, except the fields that must differ
  (model / messages / stream / token cap) plus any optional .env overrides.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from fastapi import HTTPException, Request

from config import settings
from schemas import ChatCompletionRequest

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Leaked chain-of-thought cleanup (stage-1 output hygiene)
# ---------------------------------------------------------------------------
# Content models sometimes leak their reasoning into `content` instead of the
# separate `reasoning_content` field (e.g. when --reasoning-format is not set
# on the server). These patterns cover the formats observed in the wild.
_REASONING_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<｜begin▁of▁thinking｜>.*?<｜end▁of▁thinking｜>", re.DOTALL),
    # Malformed variant actually observed: the block gets "closed" by a second
    # OPENING tag instead of a closing one.
    re.compile(r"<｜begin▁of▁thinking｜>.*?<｜begin▁of▁thinking｜>", re.DOTALL),
]
# Orphaned tags left over after the paired patterns above have run.
_ORPHAN_REASONING_TAGS = [
    "</think>",
    "</｜begin▁of▁thinking｜>",
    "<｜end▁of▁thinking｜>",
    "<｜begin▁of▁thinking｜>",
    "<think>",
]


def strip_reasoning(text: str) -> str:
    """
    Remove leaked chain-of-thought blocks from stage-1 output.

    Why deterministic regex instead of letting the style model handle it:
    tag-delimited reasoning is a mechanical pattern — stripping it costs zero
    GPU and zero risk, and every token of garbage we remove here is a token
    stage 2 does not have to read. UNdelimited reasoning prose and meta
    commentary (e.g. preamble lines) are left for the stage-2 prompt, because
    recognising those requires language understanding, not pattern matching.
    """
    cleaned = text
    for pattern in _REASONING_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for tag in _ORPHAN_REASONING_TAGS:
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


def extract_style_blocks(request: ChatCompletionRequest) -> str:
    """
    Extract configured <tag>...</tag> blocks (e.g. style directives) from the
    incoming ST messages so they can be forwarded into the stage-2 system
    prompt. Stage 1 sees these blocks because ST includes them in its request;
    stage 2 would never see them without this bridge, since its only input is
    stage 1's output text.

    Blocks are returned VERBATIM (tags included), deduplicated, in order of
    first appearance. Returns "" when disabled or nothing matched.
    """
    tags = [t.strip() for t in settings.stage2_style_tags.split(",") if t.strip()]
    if not tags:
        return ""
    found: list = []
    seen = set()
    for tag in tags:
        pattern = re.compile(
            rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL
        )
        for msg in request.messages:
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            for m in pattern.finditer(content):
                block = m.group(0)
                if block not in seen:
                    seen.add(block)
                    found.append(block)
    if found:
        logger.info("Stage 2: forwarding %d style block(s) from ST request", len(found))
    return "\n".join(found)


def load_stage2_prompt() -> str:
    """
    Read the stage-2 system prompt from its text file on every request.

    Why re-read instead of caching: this prompt is the single most-tuned knob
    in the pipeline. Reading a few KB per request is effectively free, and it
    means edits to stage2_prompt.txt take effect immediately with no restart.
    """
    if not settings.stage2_prompt_file:
        return ""
    configured = Path(settings.stage2_prompt_file)
    local_default = Path(__file__).resolve().parent / "stage2_prompt.txt"
    path = configured if configured.is_absolute() else local_default.parent / configured
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        # OSError covers FileNotFoundError plus the nastier cases seen in the
        # wild (e.g. a stray DIRECTORY at the mount path from Docker's
        # auto-create behaviour, permission errors). If a non-default path
        # fails, fall back to the copy baked into the project directory so a
        # bad mount never silently kills the polishing stage.
        logger.warning("Stage 2 prompt unreadable at %s (%s)", path, exc)
        if path != local_default:
            try:
                logger.info("Stage 2 prompt: falling back to baked-in %s", local_default)
                return local_default.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        logger.warning("Stage 2 running WITHOUT a system prompt")
        return ""

# Fields never copied from the ST request into the stage-2 payload:
# messages/model/stream must differ by design; the token cap gets its own
# floor logic; n is forced to 1 because this is a single-candidate pipeline.
_STAGE2_EXCLUDE = {"messages", "model", "stream", "max_tokens", "max_completion_tokens", "n"}


def _base_params(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Dump the full ST request including unknown extra fields so every sampling
    parameter is preserved. None values are dropped because some backends
    reject explicit JSON nulls for numeric fields.
    """
    data = request.model_dump(exclude_none=True)
    stop = data.get("stop")
    if isinstance(stop, str):
        # OpenAI allows stop as a bare string; llama-server wants a list.
        data["stop"] = [stop]
    return data


# ---------------------------------------------------------------------------
# Low-level non-streaming call with the backend's own error body surfaced
# ---------------------------------------------------------------------------
async def call_nonstreaming(
    base_url: str,
    payload: Dict[str, Any],
    timeout: int,
    usage_out: Optional[dict] = None,
) -> Optional[str]:
    """
    POST a non-streaming chat completion.

    Why HTTPException(502) with the backend body: llama-server's own error
    message (OOM, bad grammar, slot busy...) is the single most useful piece
    of debugging information, so it must reach the caller instead of being
    swallowed into a bare "HTTP error".
    """
    url = f"{base_url}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Backend unreachable ({base_url}): {exc}") from exc

    if resp.status_code >= 400:
        detail = resp.text[:500]
        logger.error("Backend %s HTTP %d: %s", base_url, resp.status_code, detail)
        raise HTTPException(502, f"Backend HTTP {resp.status_code}: {detail}")

    data = resp.json()
    # Let the caller harvest token-usage stats for the timing summary.
    if usage_out is not None:
        usage_out.update(data.get("usage") or {})
    choices = data.get("choices", [])
    return choices[0].get("message", {}).get("content", "") if choices else None


# ---------------------------------------------------------------------------
# Stage 1: content generation
# ---------------------------------------------------------------------------
def build_stage1_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    payload = _base_params(request)
    payload["model"] = settings.stage1_model
    payload["stream"] = False
    # Single-candidate pipeline: choices[1:] would be silently dropped, so
    # refuse ambiguity up front instead of wasting GPU on unused candidates.
    payload["n"] = 1
    # Token budget = story budget + thinking headroom. llama.cpp counts
    # reasoning tokens against max_tokens, so a thinking model needs its cap
    # inflated or the story itself gets truncated mid-sentence.
    headroom = settings.stage1_thinking_headroom
    if "max_tokens" in payload:
        payload["max_tokens"] += headroom
    elif "max_completion_tokens" in payload:
        payload["max_completion_tokens"] += headroom
    else:
        payload["max_tokens"] = settings.stage1_max_tokens + headroom
    return payload


async def run_stage1(request: ChatCompletionRequest, stats: Optional[dict] = None) -> Optional[str]:
    """
    Forward ST's request to the content model.

    Stage 1 is streamed INTERNALLY (still invisible to the user): consuming
    the token stream lets the proxy log llama.cpp-style live progress —
    TTFT (prefill time), token count split into reasoning vs story, and tok/s
    — instead of staring at a silent connection for minutes while a thinking
    model works. The full story text is assembled and returned whole, so the
    rest of the pipeline is unchanged.
    """
    payload = build_stage1_payload(request)
    payload["stream"] = True
    # Ask for a final usage chunk so exact prompt/completion token counts can
    # be logged; llama-server supports this, servers that don't will ignore it.
    payload["stream_options"] = {"include_usage": True}

    url = f"{settings.stage1_url}/chat/completions"
    logger.info("Stage 1: %d messages -> %s", len(payload["messages"]), url)

    content_parts: list = []
    reasoning_n = 0
    story_n = 0
    usage: Optional[Dict[str, Any]] = None
    start = time.monotonic()
    first_token_at: Optional[float] = None

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.stage1_timeout, connect=10.0)
        ) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    logger.error("Stage 1 HTTP %d: %s", resp.status_code, body)
                    raise HTTPException(502, f"Stage 1 backend HTTP {resp.status_code}: {body}")

                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    # stream_options final chunk: empty choices + usage stats
                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    if first_token_at is None and (delta.get("content") or delta.get("reasoning_content")):
                        first_token_at = time.monotonic()
                        logger.info("Stage 1: prompt processed, TTFT %.1fs", first_token_at - start)

                    if delta.get("reasoning_content"):
                        reasoning_n += 1
                    piece = delta.get("content") or ""
                    if piece:
                        story_n += 1
                        content_parts.append(piece)

                    # One chunk ~= one token for llama-server; llama.cpp-style log.
                    total_n = reasoning_n + story_n
                    if total_n and total_n % 100 == 0:
                        tg = total_n / max(time.monotonic() - (first_token_at or start), 1e-6)
                        logger.info(
                            "Stage 1: n=%d (reasoning=%d, story=%d), tg=%.1f t/s",
                            total_n, reasoning_n, story_n, tg,
                        )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Stage 1 backend unreachable: {exc}") from exc

    total_s = time.monotonic() - start
    text = "".join(content_parts)

    # Record per-stage timing for the end-of-request summary. Token counts
    # come from the usage chunk when the server provides one, otherwise from
    # chunk counting (one chunk ~= one token on llama-server).
    ttft_s = (first_token_at - start) if first_token_at else None
    if stats is not None:
        stats["stage1"] = {
            "total_s": total_s,
            "ttft_s": ttft_s,
            "gen_s": (total_s - ttft_s) if ttft_s else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens") or (reasoning_n + story_n) or None,
            "reasoning_n": reasoning_n,
            "story_n": story_n,
        }
    logger.info(
        "Stage 1 done: completion=%s tokens (reasoning=%d, story=%d), total=%.1fs",
        (usage or {}).get("completion_tokens", "?"), reasoning_n, story_n, total_s,
    )

    if text and settings.stage1_strip_reasoning:
        stripped = strip_reasoning(text)
        removed = len(text) - len(stripped)
        if removed > 0:
            logger.info("Stage 1: stripped %d chars of leaked reasoning", removed)
        text = stripped
    return text


# ---------------------------------------------------------------------------
# Stage 2: style refinement
# ---------------------------------------------------------------------------
def build_stage2_payload(request: ChatCompletionRequest, raw_text: str, stream: bool) -> Dict[str, Any]:
    data = _base_params(request)
    payload = {k: v for k, v in data.items() if k not in _STAGE2_EXCLUDE}

    payload["model"] = settings.stage2_model
    # The polishing system prompt lives in an external text file so it can be
    # tuned without touching code or restarting the proxy. If the file is
    # missing/empty, stage 2 simply runs without a system message.
    prompt = load_stage2_prompt()
    style_blocks = extract_style_blocks(request)
    if prompt and "{STYLE_BLOCKS}" in prompt:
        # The prompt file defines WHERE style directives land in the system
        # prompt; the code only supplies the raw extracted content.
        prompt = prompt.replace("{STYLE_BLOCKS}", style_blocks if style_blocks else "(none)")
    elif style_blocks:
        # Prompt file has no placeholder — append rather than silently drop.
        prompt = (prompt + "\n\n" + style_blocks).strip() if prompt else style_blocks
    messages = [{"role": "system", "content": prompt}] if prompt else []
    messages.append({"role": "user", "content": raw_text})
    payload["messages"] = messages
    payload["stream"] = stream
    if stream:
        # Exact completion-token counts for the timing summary; servers that
        # do not support stream_options simply ignore it.
        payload["stream_options"] = {"include_usage": True}

    # The polish pass may expand the text; its budget must be at least what
    # stage 1 was allowed to produce, otherwise output gets truncated.
    st_cap = data.get("max_tokens") or data.get("max_completion_tokens") or 0
    payload["max_tokens"] = max(settings.stage2_max_tokens, st_cap)

    # Optional .env overrides win over inherited ST values when set.
    if settings.stage2_temperature is not None:
        payload["temperature"] = settings.stage2_temperature
    if settings.stage2_top_p is not None:
        payload["top_p"] = settings.stage2_top_p
    return payload


async def _stream_stage2_impl(
    request: ChatCompletionRequest,
    raw_request: Request,
    stats: dict,
) -> AsyncGenerator[str, None]:
    """
    Run stage 1, then stream stage 2's SSE output to ST.

    Stage-1 errors are delivered as SSE error events rather than HTTP error
    codes: by the time this generator runs, StreamingResponse has already
    committed a 200 status, so raising HTTPException here would just kill the
    stream with no useful client-facing error.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # --- Stage 1 with downstream keepalives ---
    # A large content model can take a long time on big contexts. SSE comment
    # lines (starting with ':') are spec-compliant no-ops that keep ST and any
    # intermediate proxy from timing out the idle connection, and checking
    # is_disconnected lets us cancel the upstream GPU work when the user
    # aborts the generation in ST.
    stage1_task = asyncio.create_task(run_stage1(request, stats))
    try:
        while True:
            if await raw_request.is_disconnected():
                logger.info("Client disconnected during stage 1, cancelling upstream call")
                stage1_task.cancel()
                return
            try:
                raw_text = await asyncio.wait_for(
                    asyncio.shield(stage1_task),
                    timeout=settings.keepalive_interval,
                )
                break
            except asyncio.TimeoutError:
                yield ": waiting for stage 1\n\n"
    except HTTPException as exc:
        yield _sse_error(str(exc.detail))
        return
    except Exception as exc:
        logger.exception("Stage 1 unexpected error")
        yield _sse_error(f"Stage 1 failed: {exc}")
        return
    finally:
        # If the generator itself gets cancelled (client went away mid-wait),
        # make sure the upstream stage-1 request does not keep burning GPU.
        if not stage1_task.done():
            stage1_task.cancel()

    if not raw_text or not raw_text.strip():
        yield _sse_error("Stage 1 returned empty response")
        return

    payload = build_stage2_payload(request, raw_text, stream=True)
    url = f"{settings.stage2_url}/chat/completions"
    logger.info("Stage 2: streaming from %s (%d chars in)", settings.stage2_url, len(raw_text))

    # llama.cpp-style progress accounting for the forwarded stream
    s2_start = time.monotonic()
    s2_first: Optional[float] = None
    s2_n = 0
    s2_usage: Optional[dict] = None
    disconnected = False

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.stage2_timeout, connect=10.0)
        ) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    logger.error("Stage 2 HTTP %d: %s", resp.status_code, body)
                    yield _sse_error(f"Stage 2 HTTP {resp.status_code}: {body}")
                    return

                async for line in resp.aiter_lines():
                    # Early exit saves GPU: once ST goes away, consuming more
                    # upstream tokens is pure waste.
                    if await raw_request.is_disconnected():
                        logger.info("Client disconnected, stopping stage 2 stream")
                        disconnected = True
                        break
                    if not line:
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                        except json.JSONDecodeError:
                            yield line + "\n\n"
                            continue

                        choices = chunk.get("choices") or []
                        if chunk.get("usage"):
                            s2_usage = chunk["usage"]
                        # Usage-only chunks carry bookkeeping WE asked for;
                        # ST never requested it, so it is consumed here and
                        # never forwarded downstream.
                        if not choices:
                            continue

                        # One chunk ~= one token; llama.cpp-style progress.
                        piece = (choices[0].get("delta") or {}).get("content") or ""
                        if piece:
                            if s2_first is None:
                                s2_first = time.monotonic()
                                logger.info("Stage 2: TTFT %.2fs", s2_first - s2_start)
                            s2_n += 1
                            if s2_n % 100 == 0:
                                tg = s2_n / max(time.monotonic() - s2_first, 1e-6)
                                logger.info("Stage 2: n_gen=%d, tg=%.1f t/s", s2_n, tg)

                        # Consistent id/model so ST sees one coherent
                        # completion session across all chunks.
                        chunk["id"] = chat_id
                        chunk["model"] = settings.virtual_model_name
                        # ensure_ascii=False keeps non-ASCII text (e.g.
                        # Chinese RP prose) as compact raw UTF-8 instead
                        # of bloated \uXXXX escape sequences.
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    else:
                        # Covers "data: [DONE]" and any ":" comment lines.
                        yield line + "\n\n"
    except httpx.HTTPError as exc:
        logger.error("Stage 2 connection error: %s", exc)
        yield _sse_error(f"Stage 2 connection failed: {exc}")
    finally:
        # Record stage-2 timing even on partial/disconnected streams so the
        # end-of-request summary stays complete.
        s2_total = time.monotonic() - s2_start
        stats["stage2"] = {
            "total_s": s2_total,
            "ttft_s": (s2_first - s2_start) if s2_first else None,
            "gen_s": (time.monotonic() - s2_first) if s2_first else None,
            "completion_tokens": (s2_usage or {}).get("completion_tokens") or (s2_n or None),
            "partial": disconnected,
        }


async def stream_stage2(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> AsyncGenerator[str, None]:
    """
    Thin wrapper around the streaming pipeline that guarantees the
    per-request timing summary is logged exactly once, however the request
    ends (complete, error, or client disconnect).
    """
    stats: dict = {}
    try:
        async for event in _stream_stage2_impl(request, raw_request, stats):
            yield event
    finally:
        log_request_summary(stats)


def _fmt_stage(s: dict) -> str:
    """Format one stage's timing: total Xs | prefill Ys (N tok, S t/s) | gen Zs (N tok, S t/s)."""
    parts = [f"total {s['total_s']:.1f}s"]
    if s.get("ttft_s") is not None:
        seg = f"prefill {s['ttft_s']:.1f}s"
        pt = s.get("prompt_tokens")
        if pt and s["ttft_s"] > 0:
            seg += f" ({pt} tok, {pt / s['ttft_s']:.1f} t/s)"
        parts.append(seg)
    if s.get("gen_s") is not None:
        seg = f"gen {s['gen_s']:.1f}s"
        ct = s.get("completion_tokens")
        if ct and s["gen_s"] > 0:
            seg += f" ({ct} tok, {ct / s['gen_s']:.1f} t/s)"
        parts.append(seg)
    elif s.get("completion_tokens"):
        parts.append(f"completion {s['completion_tokens']} tok")
    return " | ".join(parts)


def log_request_summary(stats: dict) -> None:
    """
    llama.cpp-style per-request timing breakdown: how long each stage spent
    on prefill vs generation and at what speed, plus the end-to-end total.
    """
    s1 = stats.get("stage1")
    s2 = stats.get("stage2")
    if not s1 and not s2:
        return
    logger.info("---- request timing ----")
    if s1:
        split = ""
        if s1.get("reasoning_n"):
            split = f" [reasoning {s1['reasoning_n']} + story {s1['story_n']}]"
        logger.info("Stage 1 (content): %s%s", _fmt_stage(s1), split)
    if s2:
        partial = " (partial)" if s2.get("partial") else ""
        logger.info("Stage 2 (style):   %s%s", _fmt_stage(s2), partial)
    total = sum(s.get("total_s", 0.0) for s in (s1, s2) if s)
    logger.info("Total: %.1fs", total)


def _sse_error(message: str) -> str:
    """Format an error as an SSE event followed by [DONE] so ST can display it."""
    return f"data: {json.dumps({'error': message}, ensure_ascii=False)}\n\ndata: [DONE]\n\n"