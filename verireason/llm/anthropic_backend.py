"""Claude backend (official `anthropic` SDK). Credentials: ANTHROPIC_API_KEY or `ant auth login`.

Server-side refusal fallbacks are intentionally NOT enabled: a silent switch to another model
would mix generators inside one experimental condition. The served model id is recorded instead.
"""

from __future__ import annotations

import anthropic

from verireason.llm.base import LLM, Completion


class AnthropicLLM(LLM):
    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = None):
        self.model = model
        self.name = model
        self.effort = effort
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, *, max_tokens: int,
                 temperature: float | None = None) -> Completion:
        # Sampling parameters are not accepted by current Claude models, so `temperature` is
        # ignored here; run-to-run variance is measured with multiple samples instead.
        kwargs = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], **kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} refused the request: {resp.stop_details}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text=text, model=resp.model, stop_reason=resp.stop_reason,
                          usage={"input_tokens": resp.usage.input_tokens,
                                 "output_tokens": resp.usage.output_tokens})
