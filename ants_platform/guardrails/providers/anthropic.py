from __future__ import annotations

import time
from typing import Any, Optional

from ..client import AntsGuardrailsClient
from ..errors import GuardrailViolationError
from .._tracing import start_trace_span, start_generation_span, end_generation_span, end_trace_span, send_trace_via_ingestion
from ._guardrail_utils import effective_text, overall_guardrail_result


class AntsAnthropic:
    """Anthropic client wrapper with guardrail enforcement + automatic OTEL tracing.

    OTEL traces are only created when guardrail checks pass (PASS or SANITIZED).
    Blocked requests produce no OTEL trace.
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
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key, **kwargs)
        self._guardrails = AntsGuardrailsClient(
            ants_api_key, ants_base_url, agent_id, guardrail_service_url
        )
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url or "https://app.antsplatform.com"
        self.messages = _Messages(self._client, self._guardrails, agent_id, agent_name, ants_api_key, self._ants_base_url)


class _Messages:
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
                effective_kwargs = {**kwargs, "messages": [{"role": "user", "content": input_check.sanitized_text}]}
        effective_messages = effective_kwargs.get("messages", messages)
        effective_input_text = effective_text(input_text, input_check)

        # STEP 2: LLM call
        start_time = time.time()
        response = self._client.messages.create(**effective_kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        output_text = _extract_output_text(response)
        usage = {}
        if response.usage:
            usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens,
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
            self._agent_name or f"anthropic/{model}",
            input_data=effective_messages,
            agent_id=self._agent_id,
            provider="anthropic",
            tags=["ants-sdk", "anthropic"],
        ) as trace_span:
            with start_generation_span(
                "generation",
                model=model,
                input_data=effective_messages,
                provider="anthropic",
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
            provider="anthropic",
            agent_id=self._agent_id,
            input_data=effective_messages,
            output_data=output_text,
            usage=usage,
            latency_ms=latency_ms,
            tags=["ants-sdk", "anthropic"],
            guardrail_result=guardrail_result,
        )

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


def _extract_output_text(response: Any) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _apply_sanitized_output(response: Any, output_text: str) -> Any:
    content = getattr(response, "content", None)
    if not content:
        return response

    updated_content: list[Any] = []
    replaced = False
    for block in content:
        if getattr(block, "type", None) == "text":
            replacement_text = output_text if not replaced else ""
            if hasattr(block, "model_copy"):
                updated_block = block.model_copy(update={"text": replacement_text})
            else:
                try:
                    block.text = replacement_text
                except Exception:
                    pass
                updated_block = block
            replaced = True
            updated_content.append(updated_block)
        else:
            updated_content.append(block)

    if not replaced:
        return response
    if hasattr(response, "model_copy"):
        return response.model_copy(update={"content": updated_content})
    try:
        response.content = updated_content
    except Exception:
        pass
    return response
