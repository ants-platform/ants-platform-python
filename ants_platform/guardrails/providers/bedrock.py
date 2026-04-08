from __future__ import annotations

import time
from typing import Any, Optional

from ..client import AntsGuardrailsClient
from ..errors import GuardrailViolationError
from .._tracing import start_trace_span, start_generation_span, end_generation_span, end_trace_span, send_trace_via_ingestion
from ._guardrail_utils import effective_text, overall_guardrail_result


class AntsBedrock:
    """AWS Bedrock client wrapper with guardrail enforcement + automatic OTEL tracing.

    OTEL traces are only created when guardrail checks pass (PASS or SANITIZED).
    Blocked requests produce no OTEL trace.
    """

    def __init__(
        self,
        *,
        ants_api_key: str,
        ants_base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        guardrail_service_url: Optional[str] = None,
        region_name: str = "us-east-1",
        **kwargs: Any,
    ) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=region_name, **kwargs)
        self._guardrails = AntsGuardrailsClient(
            ants_api_key, ants_base_url, agent_id, guardrail_service_url
        )
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url or "https://app.antsplatform.com"

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        messages = kwargs.get("messages", [])
        model = kwargs.get("modelId", "unknown")
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
                effective_kwargs = {**kwargs, "messages": [
                    {"role": "user", "content": [{"text": input_check.sanitized_text}]}
                ]}
        effective_messages = effective_kwargs.get("messages", messages)
        effective_input_text = effective_text(input_text, input_check)

        # STEP 2: LLM call
        start_time = time.time()
        response = self._client.converse(**effective_kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        output_text = _extract_output_text(response)
        bedrock_usage = response.get("usage", {})
        usage = {
            "input": bedrock_usage.get("inputTokens", 0),
            "output": bedrock_usage.get("outputTokens", 0),
            "total": bedrock_usage.get("totalTokens", 0),
        }

        # STEP 3: Output guardrail check - still no spans
        if guardrail_active and output_text:
            output_check = self._guardrails.check_output(output_text, effective_input_text)
            if output_check.result == "BLOCKED":
                raise GuardrailViolationError("output", output_check)
            output_text = effective_text(output_text, output_check)
            _apply_sanitized_output(response, output_text)

        guardrail_result = overall_guardrail_result(
            guardrail_active, input_check, output_check
        )

        # STEP 4: Both checks passed - NOW create and end spans
        with start_trace_span(
            self._agent_name or f"bedrock/{model}",
            input_data=effective_messages,
            agent_id=self._agent_id,
            provider="bedrock",
            tags=["ants-sdk", "bedrock"],
        ) as trace_span:
            with start_generation_span(
                "generation",
                model=model,
                input_data=effective_messages,
                provider="bedrock",
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
            provider="bedrock",
            agent_id=self._agent_id,
            input_data=effective_messages,
            output_data=output_text,
            usage=usage,
            latency_ms=latency_ms,
            tags=["ants-sdk", "bedrock"],
            guardrail_result=guardrail_result,
        )

        return response


def _extract_input_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        for block in m.get("content", []):
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
    return "\n".join(parts)


def _extract_output_text(response: dict[str, Any]) -> str:
    output = response.get("output", {})
    message = output.get("message", {})
    return "".join(
        block["text"]
        for block in message.get("content", [])
        if isinstance(block, dict) and "text" in block
    )


def _apply_sanitized_output(response: dict[str, Any], output_text: str) -> None:
    output = response.setdefault("output", {})
    message = output.setdefault("message", {})
    content = message.get("content")
    if not isinstance(content, list):
        message["content"] = [{"text": output_text}]
        return

    replaced = False
    for block in content:
        if not isinstance(block, dict) or "text" not in block:
            continue
        block["text"] = output_text if not replaced else ""
        replaced = True

    if not replaced:
        content.append({"text": output_text})
