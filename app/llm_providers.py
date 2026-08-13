"""LLM provider abstraction for the generation stage.

Keeps the concrete LLM API (provider, model, key) swappable behind a small
interface so app/rag.py never needs to change when the provider changes.
Provider, model, and key are all read from environment variables — never
hard-coded.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class LLMProvider(ABC):
    """Minimal interface every concrete LLM provider must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a fully-formed prompt to the LLM and return its text response."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini provider, via LangChain's ChatGoogleGenerativeAI."""

    def __init__(self, model: str, api_key: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._client = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return response.content


def get_llm_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Build the configured LLM provider from explicit args or env vars.

    Reads LLM_PROVIDER, LLM_MODEL, and LLM_API_KEY when not passed
    explicitly. Raises a clear, actionable error if required configuration
    is missing — never falls back to inventing a key or switching to a
    different provider silently.
    """
    provider_name = (provider_name or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)).lower()

    if provider_name == "gemini":
        model_name = model_name or os.getenv("LLM_MODEL", DEFAULT_GEMINI_MODEL)
        api_key = api_key or os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. To use the Gemini provider, set these "
                "environment variables (e.g. in a .env file):\n"
                "  LLM_PROVIDER=gemini\n"
                "  LLM_MODEL=gemini-2.5-flash   (optional, this is the default)\n"
                "  LLM_API_KEY=<your Google AI Studio API key>\n"
                "Get a key at https://aistudio.google.com/apikey"
            )
        return GeminiProvider(model=model_name, api_key=api_key)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider_name!r}. Supported providers: 'gemini'."
    )
