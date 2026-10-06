"""OpenAI-compatible chat endpoint, used for open-weight generators served by vLLM, Together, etc.

Configure with OPENAI_BASE_URL / OPENAI_API_KEY (see .env.example).
"""

from __future__ import annotations

from openai import OpenAI

from verireason.llm.base import LLM, Completion


class OpenAICompatLLM(LLM):
    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.name = model
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()

    def complete(self, system: str, user: str, *, max_tokens: int,
                 temperature: float | None = None) -> Completion:
        kwargs = {"temperature": temperature} if temperature is not None else {}
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs)
        choice = resp.choices[0]
        usage = resp.usage.model_dump() if resp.usage else {}
        return Completion(text=choice.message.content or "", model=resp.model,
                          stop_reason=choice.finish_reason, usage=usage)
