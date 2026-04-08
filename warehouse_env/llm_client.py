"""
LLM Client — Warehouse Order Fulfillment
==========================================
Centralized, reusable wrapper around the OpenAI Python client.
Works with any OpenAI-compatible API (OpenAI, Groq, Together, etc.)
by setting environment variables.

Environment variables:
  GROQ_API_KEY   — API key (checked first)
  OPENAI_API_KEY — Fallback API key
  HF_TOKEN       — HuggingFace Token (optional)
  API_BASE_URL   — Base URL
  MODEL_NAME     — Model to use
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np


def get_llm_client():
    """Create and return an OpenAI-compatible client configured via env vars.

    Returns:
        tuple: (client, model_name, api_key) or (None, model_name, "") if no key.
    """
    # Check for API key — support GROQ_API_KEY, OPENAI_API_KEY, and HF_TOKEN
    groq_key = os.environ.get("GROQ_API_KEY", "") or os.environ.get("Groq_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    # Priority: GROQ > OPENAI > HF
    if groq_key:
        api_key = groq_key
        default_base = "https://api.groq.com/openai/v1"
        provider = "Groq"
    elif openai_key:
        api_key = openai_key
        default_base = "https://api.openai.com/v1"
        provider = "OpenAI"
    elif hf_token:
        api_key = hf_token
        default_base = "https://api-inference.huggingface.co/v1"
        provider = "HuggingFace"
    else:
        api_key = ""
        default_base = "https://api.groq.com/openai/v1"
        provider = "None"

    api_base = os.environ.get("API_BASE_URL", default_base)
    model_name = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

    if not api_key:
        return None, model_name, ""

    print(f"    [INFO] Using LLM provider: {provider}")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=8.0,
            max_retries=0,
        )
        return client, model_name, api_key
    except ImportError:
        # openai package not installed — fall back to groq SDK
        try:
            from groq import Groq

            client = Groq(api_key=api_key, timeout=8.0, max_retries=0)
            return client, model_name, api_key
        except ImportError:
            return None, model_name, api_key


def llm_chat(
    client,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
) -> Optional[str]:
    """Send a chat completion request. Returns the text or None on failure.

    Works with both the OpenAI and Groq client objects since they share
    the same `.chat.completions.create()` interface.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [WARN] LLM API error: {e}")
        return None


def parse_action(text: Optional[str], valid_actions: list[int]) -> Optional[int]:
    """Extract an integer action from LLM text. Returns None if invalid."""
    if text is None:
        return None
    match = re.search(r"\d+", text)
    if match:
        action = int(match.group())
        if action in valid_actions:
            return action
    return None


def random_valid_action(
    env,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Pick a random valid action from the environment (fallback)."""
    valid = np.where(env.queue_proc_time > 0)[0]
    if len(valid) == 0:
        return env.max_queue  # no-op
    if rng is not None:
        return int(rng.choice(valid))
    return int(np.random.choice(valid))
