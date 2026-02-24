"""
llm_providers.py — Multi-provider LLM adapter for the CFM Data Chatbot.
Supports OpenAI, Anthropic, and Google Gemini with streaming.
"""
from __future__ import annotations

import json
from typing import Generator

# ---------------------------------------------------------------------------
# Model catalogues per provider
# ---------------------------------------------------------------------------
PROVIDER_MODELS = {
    "openai": [
        "gpt-5-codex",
        "gpt-5",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "anthropic": [
        "claude-opus-4-6",
        "claude-opus-4-0-20250618",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ],
}

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
}


# ---------------------------------------------------------------------------
# Unified chat interface
# ---------------------------------------------------------------------------
def chat(
    provider: str,
    model: str,
    api_key: str,
    messages: list[dict],
    stream: bool = True,
) -> str | Generator[str, None, None]:
    """
    Send *messages* to the chosen LLM and return the assistant reply.

    Parameters
    ----------
    provider : "openai" | "anthropic" | "gemini"
    model    : model id string (must be in PROVIDER_MODELS[provider])
    api_key  : user-supplied API key
    messages : list of {"role": "system"|"user"|"assistant", "content": str}
    stream   : if True return a generator yielding text chunks

    Returns
    -------
    str (non-streaming) or Generator[str] (streaming)
    """
    provider = provider.lower().strip()
    if provider == "openai":
        return _chat_openai(model, api_key, messages, stream)
    elif provider == "anthropic":
        return _chat_anthropic(model, api_key, messages, stream)
    elif provider == "gemini":
        return _chat_gemini(model, api_key, messages, stream)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
def _chat_openai(model: str, api_key: str, messages: list[dict], stream: bool):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    if stream:
        def _gen():
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        return _gen()
    else:
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
def _chat_anthropic(model: str, api_key: str, messages: list[dict], stream: bool):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Anthropic requires system prompt separated from messages
    system_text = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            chat_messages.append({"role": m["role"], "content": m["content"]})

    kwargs = dict(
        model=model,
        max_tokens=4096,
        messages=chat_messages,
    )
    if system_text.strip():
        kwargs["system"] = system_text.strip()

    if stream:
        def _gen():
            with client.messages.stream(**kwargs) as resp:
                for text in resp.text_stream:
                    yield text
        return _gen()
    else:
        resp = client.messages.create(**kwargs)
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------
def _chat_gemini(model: str, api_key: str, messages: list[dict], stream: bool):
    from google import genai

    client = genai.Client(api_key=api_key)

    # Build Gemini contents — merge system into first user message
    system_text = ""
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

    # Prepend system context to the first user message
    if system_text.strip() and contents:
        first = contents[0]
        first["parts"][0]["text"] = system_text.strip() + "\n\n" + first["parts"][0]["text"]

    if stream:
        def _gen():
            resp = client.models.generate_content_stream(
                model=model,
                contents=contents,
            )
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        return _gen()
    else:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
        )
        return resp.text or ""
