"""LLM providers. Switch with LLM_PROVIDER = pollinations | ollama | fake."""
from app import config
from app.llm.base import LLMError, Provider, Usage


def get_provider(name: str | None = None) -> Provider:
    name = (name or config.LLM_PROVIDER).lower()
    if name == "pollinations":
        from app.llm.pollinations import PollinationsProvider
        return PollinationsProvider()
    if name == "ollama":
        from app.llm.ollama import OllamaProvider
        return OllamaProvider()
    if name == "fake":
        from app.llm.fake import FakeProvider
        return FakeProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{name}' (use pollinations, ollama or fake)")


__all__ = ["LLMError", "Provider", "Usage", "get_provider"]
