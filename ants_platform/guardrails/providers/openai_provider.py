from __future__ import annotations

import time
from typing import Any, Optional

from ..client import AntsGuardrailsClient
from ..errors import GuardrailViolationError
from .._tracing import start_trace_span, start_generation_span, end_generation_span, end_trace_span, send_trace_via_ingestion
from ._guardrail_utils import effective_text, overall_guardrail_result


class AntsOpenAI:
    """OpenAI client wrapper with guardrail enforcement + automatic OTEL tracing.

    Guardrail checks only run when ``agent_id`` is provided and a policy exists.
    OTEL traces are only created when guardrail checks pass (PASS or SANITIZED).
    Blocked requests produce no OTEL trace.

    Usage::

        from ants_platform.guardrails.providers.openai_provider import AntsOpenAI

        client = AntsOpenAI(
            api_key="sk-...",
            ants_api_key="pk:sk",
            agent_id="agent_123",
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        ants_api_key: str,
        ants_base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        guardrail_service_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key, **kwargs)
        self._guardrails = AntsGuardrailsClient(
            ants_api_key, ants_base_url, agent_id, guardrail_service_url
        )
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url or "https://app.antsplatform.com"
        self.chat = _Chat(self._client, self._guardrails, agent_id, agent_name, ants_api_key, self._ants_base_url)


class _Chat:
    def __init__(self, client: Any, guardrails: AntsGuardrailsClient, agent_id: Optional[str], agent_name: Optional[str] = None, ants_api_key: str = "", ants_base_url: str = "") -> None:
        self.completions = _Completions(client, guardrails, agent_id, agent_name, ants_api_key, ants_base_url)


class _Completions:
    def __init__(self, client: Any, guardrails: AntsGuardrailsClient, agent_id: Optional[str], agent_name: Optional[str] = None, ants_api_key: str = "", ants_base_url: str = "") -> None:
        self._client = client
        self._guardrails = guardrails
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")
        input_text = _extract_input_text(messages)
        guardrail_active = self._guardrails.enabled
        input_check = None
        output_check = None

        # STEP 1: Input guardrail check - no spans yet
        effective_kwargs = kwargs
        if guardrail_active:
            input_check = self._guardrails.check_input(input_text)
            if input_check.result == "BLOCKED":
                raise GuardrailViolationError("input", input_check)
            if input_check.result == "SANITIZED" and input_check.sanitized_text is not None:
                effective_kwargs = {
                    **kwargs,
                    "messages": [{"role": "user", "content": input_check.sanitized_text}],
                }
        effective_messages = effective_kwargs.get("messages", messages)
        effective_input_text = effective_text(input_text, input_check)

        # STEP 2: LLM call
        start_time = time.time()
        response = self._client.chat.completions.create(**effective_kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        output_text = response.choices[0].message.content or "" if response.choices else ""
        usage = {}
        if response.usage:
            usage = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }

        # STEP 3: Output guardrail check - still no spans
        if guardrail_active and output_text:
            output_check = self._guardrails.check_output(output_text, effective_input_text)
            if output_check.result == "BLOCKED":
                raise GuardrailViolationError("output", output_check)
            output_text = effective_text(output_text, output_check)
            response = _apply_sanitized_output(response, output_text)

        guardrail_result = overall_guardrail_result(
            guardrail_active, input_check, output_check
        )

        # STEP 4: Both checks passed - NOW create and end spans
        with start_trace_span(
            self._agent_name or f"openai/{model}",
            input_data=effective_messages,
            agent_id=self._agent_id,
            provider="openai",
            tags=["ants-sdk", "openai"],
        ) as trace_span:
            with start_generation_span(
                "generation",
                model=model,
                input_data=effective_messages,
                provider="openai",
            ) as gen_span:
                end_generation_span(
                    gen_span,
                    output_data=output_text,
                    usage=usage,
                    latency_ms=latency_ms,
                    guardrail_result=guardrail_result,
                )
                end_trace_span(trace_span, output_data=output_text)

        # Fallback: send via ingestion API if OTEL is not configured
        send_trace_via_ingestion(
            ants_api_key=self._ants_api_key,
            base_url=self._ants_base_url,
            model=model,
            provider="openai",
            agent_id=self._agent_id,
            input_data=effective_messages,
            output_data=output_text,
            usage=usage,
            latency_ms=latency_ms,
            tags=["ants-sdk", "openai"],
            guardrail_result=guardrail_result,
        )

        return response


def _apply_sanitized_output(response: Any, output_text: str) -> Any:
    if not getattr(response, "choices", None):
        return response

    try:
        first_choice = response.choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            return response

        if (
            hasattr(response, "model_copy")
            and hasattr(first_choice, "model_copy")
            and hasattr(message, "model_copy")
        ):
            updated_message = message.model_copy(update={"content": output_text})
            updated_choices = list(response.choices)
            updated_choices[0] = first_choice.model_copy(update={"message": updated_message})
            return response.model_copy(update={"choices": updated_choices})

        message.content = output_text
    except Exception:
        pass

    return response


def _extract_input_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)
