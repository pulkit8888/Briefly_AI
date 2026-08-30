"""Mistral LLM configuration with a second Mistral model as fallback."""

import os

from langchain_mistralai import ChatMistralAI


def get_llm(temperature: float = 0.3):
    """Use Mistral Small first; retry with Mistral Large on an API error."""
    primary = ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
    )

    fallback = ChatMistralAI(
        model=os.getenv("MISTRAL_FALLBACK_MODEL", "mistral-large-latest"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
    )
    return primary.with_fallbacks([fallback])
