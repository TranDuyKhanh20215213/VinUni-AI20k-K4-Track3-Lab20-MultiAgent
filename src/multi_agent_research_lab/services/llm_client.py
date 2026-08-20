"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

# USD per 1M tokens. Update if pricing changes; fall back to gpt-4o-mini rates for unknown models.
_PRICING_PER_MILLION_TOKENS = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}
_DEFAULT_PRICING = _PRICING_PER_MILLION_TOKENS["gpt-4o-mini"]

_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, APIStatusError)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    input_rate, output_rate = _PRICING_PER_MILLION_TOKENS.get(model, _DEFAULT_PRICING)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI's Chat Completions API."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self._settings.openai_api_key:
                raise AgentExecutionError(
                    "OPENAI_API_KEY is not set. Add it to .env before calling LLMClient.complete."
                )
            self._client = OpenAI(
                api_key=self._settings.openai_api_key,
                timeout=self._settings.timeout_seconds,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion, with retry/timeout/token accounting handled here."""

        client = self._get_client()
        model = self._settings.openai_model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
        except _RETRYABLE_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any provider error as a domain error
            raise AgentExecutionError(f"LLM completion failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(model, input_tokens, output_tokens),
        )
