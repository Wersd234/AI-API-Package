"""Configuration via .env — URLs, model names, optional stage-2 overrides.

Only connection/behaviour knobs live here. There are deliberately NO prompt
settings: this proxy is a transparent relay and never injects text.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Proxy ---
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 5000
    virtual_model_name: str = "rp-two-stage"
    log_level: str = "INFO"

    # --- Stage 1: content generation (remote device A) ---
    stage1_url: str = "http://localhost:8080/v1"
    stage1_model: str = "qwen-3.8-flash"
    # Fallback generation cap. llama-server defaults to UNBOUNDED generation
    # when the client sends no max_tokens, which would stall the pipeline on
    # long contexts, so we enforce a cap whenever ST omits one.
    # Default is generous because MVU-style variable blocks (JSONPatch tails)
    # are token-heavy: the story body alone rarely exceeds ~600 tokens but
    # the structured tail can easily double that.
    stage1_max_tokens: int = 1024
    # Thinking headroom ADDED ON TOP of the story token budget. Reasoning
    # tokens count toward max_tokens in llama.cpp, so without this a long
    # thinking chain would eat the budget and truncate the actual story.
    # Set to 0 if your stage-1 model does not think.
    stage1_thinking_headroom: int = 2048
    # Strip leaked chain-of-thought blocks (<think>..., deepseek-style
    # variants) from stage-1 output before handing it to stage 2. Deterministic
    # regex removal is cheaper and more reliable than asking the style model
    # to recognise them. Disable if your stage-1 server already separates
    # reasoning via --reasoning-format.
    stage1_strip_reasoning: bool = True
    # A large thinking MoE on a long context can legitimately need tens of
    # minutes: at ~2.6 tk/s, 2048 thinking tokens + 1024 story tokens is
    # already ~20 min. This timeout is a hang-detector, not a deadline.
    stage1_timeout: int = 1800

    # --- Stage 2: style refinement (remote device B) ---
    stage2_url: str = "http://localhost:8081/v1"
    stage2_model: str = "gemma-4-26b"
    # Floor for the stage-2 token cap: max(this, ST's max_tokens). The polish
    # pass may expand text slightly, so it must never get a smaller budget
    # than stage 1 was allowed to produce.
    stage2_max_tokens: int = 1536
    stage2_timeout: int = 120
    # Path to the stage-2 system prompt (Chinese polishing instructions).
    # Kept OUTSIDE the code so it can be edited freely without restarting —
    # the file is re-read on every request. Relative paths resolve against
    # the project directory. Set empty to run stage 2 without a system prompt.
    stage2_prompt_file: Optional[str] = "stage2_prompt.txt"
    # Comma-separated tag names whose <tag>...</tag> blocks get extracted
    # from the incoming ST messages and forwarded into the stage-2 system
    # prompt (via the {STYLE_BLOCKS} placeholder in stage2_prompt.txt).
    # This lets per-card style directives reach the style model automatically.
    # Empty = extraction disabled. The tag VALUE lives in .env (data), not
    # here, to keep Chinese text out of the code.
    stage2_style_tags: str = ""

    # Optional sampling overrides for stage 2. None = inherit whatever ST
    # sent, which keeps the relay transparent by default.
    stage2_temperature: Optional[float] = None
    stage2_top_p: Optional[float] = None

    # Interval (seconds) between SSE keepalive comment lines sent downstream
    # while stage 1 is still generating.
    keepalive_interval: float = 5.0


settings = Settings()