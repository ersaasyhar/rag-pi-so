"""LLM and Embedding client for 9router and local multilingual embeddings.

Includes TokenMeter context manager for tracking latency and token usage.
"""

from __future__ import annotations

import contextvars
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

BASE_URL = os.getenv("BASE_URL", "http://localhost:8080/v1").rstrip("/")
if not BASE_URL.endswith("/v1"):
    BASE_URL = f"{BASE_URL}/v1"

API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "code")
CHAT_REASONING_EFFORT = os.getenv("CHAT_REASONING_EFFORT", "low")
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "code")
RETRIEVAL_REASONING_EFFORT = os.getenv("RETRIEVAL_REASONING_EFFORT", "low")

# Propagate to external SDKs that read OPENAI_* env vars
os.environ.setdefault("OPENAI_BASE_URL", BASE_URL)
if API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", API_KEY)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    call_count: int = 0
    model_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(
        self,
        prompt: int,
        completion: int,
        latency: float = 0.0,
        model: str = "default",
    ) -> None:
        self.input_tokens += prompt
        self.output_tokens += completion
        self.total_tokens += prompt + completion
        self.latency_seconds += latency
        self.call_count += 1
        if model not in self.model_usage:
            self.model_usage[model] = {"input_tokens": 0, "output_tokens": 0}
        self.model_usage[model]["input_tokens"] += prompt
        self.model_usage[model]["output_tokens"] += completion


_CURRENT_METER: contextvars.ContextVar[TokenUsage | None] = contextvars.ContextVar(
    "current_meter", default=None
)


@contextmanager
def token_meter() -> Generator[TokenUsage, None, None]:
    """Context manager to measure token consumption and latency."""
    meter = TokenUsage()
    token = _CURRENT_METER.set(meter)
    start_time = time.perf_counter()
    try:
        yield meter
    finally:
        meter.latency_seconds = time.perf_counter() - start_time
        _CURRENT_METER.reset(token)


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


_DEFAULT_CHAT_MODEL: str | None = None
_EMBED_MODEL_INSTANCE: Any = None


def discover_chat_model() -> str:
    """Discover first usable chat model from 9router gateway."""
    global _DEFAULT_CHAT_MODEL
    if _DEFAULT_CHAT_MODEL:
        return _DEFAULT_CHAT_MODEL

    preferred = ["code", "flash", "gemini", "gpt", "fla/auto"]
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

    try:
        r = requests.get(f"{BASE_URL}/models", headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            available = {m["id"] for m in data.get("data", [])}
            for pref in preferred:
                if pref in available:
                    _DEFAULT_CHAT_MODEL = pref
                    return pref
            if data.get("data"):
                _DEFAULT_CHAT_MODEL = data["data"][0]["id"]
                return _DEFAULT_CHAT_MODEL
    except Exception:
        pass

    _DEFAULT_CHAT_MODEL = "code"
    return _DEFAULT_CHAT_MODEL


def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> ChatResponse:
    """Perform chat completion for final answer generation."""
    target_model = model or CHAT_MODEL or discover_chat_model()
    effort = reasoning_effort if reasoning_effort is not None else CHAT_REASONING_EFFORT
    return _invoke_llm(messages, target_model, temperature, max_tokens, effort)


def retrieval_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> ChatResponse:
    """Perform completion for tree index navigation and retrieval selection."""
    target_model = model or RETRIEVAL_MODEL or discover_chat_model()
    effort = reasoning_effort if reasoning_effort is not None else RETRIEVAL_REASONING_EFFORT
    return _invoke_llm(messages, target_model, temperature, max_tokens, effort)


def _invoke_llm(
    messages: list[dict[str, str]],
    target_model: str,
    temperature: float,
    max_tokens: int | None,
    reasoning_effort: str | None,
) -> ChatResponse:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    start_time = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    latency = time.perf_counter() - start_time

    if r.status_code != 200:
        raise RuntimeError(f"Chat completion failed [{r.status_code}]: {r.text}")

    data = r.json()
    content = ""
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    meter = _CURRENT_METER.get()
    if meter:
        meter.add(prompt_tokens, completion_tokens, latency, model=target_model)

    return ChatResponse(
        content=content,
        model=target_model,
        usage=usage,
        latency_seconds=latency,
    )


def record_tokens(
    input_tokens: int,
    output_tokens: int,
    latency: float = 0.0,
    model: str = "default",
) -> None:
    """Record arbitrary token usage (e.g. from Jev System One) to active meter."""
    meter = _CURRENT_METER.get()
    if meter:
        meter.add(input_tokens, output_tokens, latency, model=model)


def get_embedding_model() -> Any:
    global _EMBED_MODEL_INSTANCE
    if _EMBED_MODEL_INSTANCE is None:
        from fastembed import TextEmbedding

        _EMBED_MODEL_INSTANCE = TextEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _EMBED_MODEL_INSTANCE


def get_embedding_dimension() -> int:
    return 384


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Generate dense embeddings using local fast multilingual ONNX model."""
    if not texts:
        return []

    model = get_embedding_model()
    start_time = time.perf_counter()
    embeddings = list(model.embed(texts, batch_size=batch_size))
    latency = time.perf_counter() - start_time

    # Approximate token count (4 chars ~ 1 token)
    approx_tokens = sum(len(t) // 4 for t in texts)
    meter = _CURRENT_METER.get()
    if meter:
        meter.add(approx_tokens, 0, latency)

    return [e.tolist() for e in embeddings]
