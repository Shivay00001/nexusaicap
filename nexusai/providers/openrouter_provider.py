"""OpenRouter Provider — Access to many free and paid cloud LLMs."""
from __future__ import annotations
import os
from nexusai.providers.openai_provider import OpenAIProvider
from nexusai.core.registry import ProviderRegistry

class OpenRouterProvider(OpenAIProvider):
    """Uses the OpenAI SDK to connect to OpenRouter's API."""
    name = "openrouter"
    
    def __init__(self, **kwargs):
        # Default OpenRouter Base URL if not provided
        if "base_url" not in kwargs or not kwargs["base_url"]:
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
        super().__init__(**kwargs)
        
    @property
    def available_models(self) -> list[str]:
        # OpenRouter has hundreds of models. These are some notable free/popular ones.
        return [
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-pro-1.5-exp",
            "nousresearch/hermes-3-llama-3.1-405b",
            "mistralai/mistral-nemo",
            "qwen/qwen-2.5-72b-instruct"
        ]

ProviderRegistry.register("openrouter", OpenRouterProvider)
