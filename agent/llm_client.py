import asyncio
import json
import re
from typing import Any, Type

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .config import get_llm_config


def _make_chat(model_override: str | None = None, temperature: float | None = None) -> ChatOllama:
    cfg = get_llm_config()
    return ChatOllama(
        base_url=cfg.base_url,
        model=model_override or cfg.model,
        temperature=cfg.temperature if temperature is None else temperature,
    )


async def _ainvoke_with_retries(messages: list, *, timeout: int | None = None) -> str:
    cfg = get_llm_config()
    chat = _make_chat()
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            msg = await asyncio.wait_for(chat.ainvoke(messages), timeout=timeout or cfg.timeout)
            return (msg.content or "").strip()
        except Exception as e:
            last_err = e
            if attempt >= cfg.max_retries:
                break
            await asyncio.sleep(cfg.retry_backoff * (2 ** attempt))
    if last_err:
        raise last_err
    raise RuntimeError("LLM invocation failed without an exception")


def _parse_json_best_effort(content: str) -> dict[str, Any]:
    """Parse STRICT JSON with small guardrails for code fences and extra text.

    - Strips leading ```json/``` fences and trailing ``` if present.
    - Falls back to the first {...} blob if direct parse fails.
    """
    s = (content or "").strip()
    # Strip common code fences
    lower = s.lower()
    if lower.startswith("```json"):
        s = s[7:].strip()  # len("```json") == 7
    elif lower.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            return json.loads(m.group(0))
        raise


async def llm_json(system: str, user: str, model: str | None = None, timeout: int | None = None) -> dict[str, Any]:
    """Call the chat model and parse STRICT JSON (best effort).

    Maintains backward-compatible signature. Uses centralized config for
    model, timeout, and retries.
    """
    if model:
        # If caller overrides model, build a one-off chat with that model
        chat = _make_chat(model_override=model)
        content = (await asyncio.wait_for(
            chat.ainvoke([SystemMessage(content=system), HumanMessage(content=user)]),
            timeout=timeout or get_llm_config().timeout,
        )).content or ""
    else:
        content = await _ainvoke_with_retries([SystemMessage(content=system), HumanMessage(content=user)], timeout=timeout)
    return _parse_json_best_effort(content)


async def llm_json_typed(system: str, user: str, schema: Type[BaseModel], *, model: str | None = None, timeout: int | None = None) -> BaseModel:
    """Like llm_json, but validates against a Pydantic schema."""
    obj = await llm_json(system, user, model=model, timeout=timeout)
    return schema.model_validate(obj)
