"""Permissive OpenAI-compatible models.

The request model intentionally keeps messages as raw dicts and allows
arbitrary extra fields: this proxy is a transparent relay, so whatever
sampling parameters SillyTavern sends (temperature, penalties, logit_bias,
vendor-specific extras) must survive the round trip untouched.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class ChatCompletionRequest(BaseModel):
    # extra="allow" keeps every unknown field ST sends, so model_dump()
    # returns a faithful copy of the original request body.
    model_config = ConfigDict(extra="allow")

    # Messages stay as raw dicts: role/content/name are passed through
    # verbatim, and multimodal content arrays won't fail validation.
    messages: List[Dict[str, Any]]
    stream: bool = False
    model: str = "rp-two-stage"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ai-rp-proxy"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]