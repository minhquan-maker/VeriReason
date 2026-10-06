"""Provider-agnostic text-completion interface used by generators, extractors and repair."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    model: str                      # model id that actually served the request
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)


class LLM(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int,
                 temperature: float | None = None) -> Completion:
        ...


def build_llm(backend: str, model: str | None = None, **kwargs) -> LLM:
    if backend == "anthropic":
        from verireason.llm.anthropic_backend import AnthropicLLM
        return AnthropicLLM(model=model or "claude-opus-5-5", **kwargs)
    if backend == "openai_compat":
        from verireason.llm.openai_compat import OpenAICompatLLM
        if not model:
            raise ValueError("openai_compat backend needs an explicit model id")
        return OpenAICompatLLM(model=model, **kwargs)
    raise ValueError(f"unknown LLM backend {backend!r} (offline backends are handled by callers)")
