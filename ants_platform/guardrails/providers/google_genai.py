from __future__ import annotations

import time
from typing import Any, Optional

from ..client import AntsGuardrailsClient
from ..errors import GuardrailViolationError
from .._tracing import start_trace_span, start_generation_span, end_generation_span, end_trace_span, send_trace_via_ingestion


class AntsGoogleGenAI:
    """Google GenAI client wrapper with guardrail enforcement + automatic OTEL tracing.

    OTEL traces are only created when guardrail checks pass (PASS or SANITIZED).
    Blocked requests produce no OTEL trace.
    """

    def __init__(
        self,
        *,
        api_key: str,
        ants_api_key: str,
        ants_base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        guardrail_service_url: Optional[str] = None,
    ) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._guardrails = AntsGuardrailsClient(
            ants_api_key, ants_base_url, agent_id, guardrail_service_url
        )
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url or "https://app.antsplatform.com"
        self.models = _Models(self._client, self._guardrails, agent_id, agent_name, ants_api_key, self._ants_base_url)


class _Models:
    def __init__(self, client: Any, guardrails: AntsGuardrailsClient, agent_id: Optional[str], agent_name: Optional[str] = None, ants_api_key: str = "", ants_base_url: str = "") -> None:
        self._client = client
        self._guardrails = guardrails
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        input_text = _extract_input_text(contents)
        guardrail_active = self._guardrails.enabled

        # STEP 1: Input guardrail check - no spans yet
        effective_contents = contents
        if guardrail_active:
            input_check = self._guardrails.check_input(input_text)
            if input_check.result == "BLOCKED":
                raise GuardrailViolationError("input", input_check)
            if input_check.result == "SANITIZED" and input_check.sanitized_text:
                effective_contents = input_check.sanitized_text

        # STEP 2: LLM call
        start_time = time.time()
        response = self._client.models.generate_content(
            model=model, contents=effective_contents, config=config
        )
        latency_ms = int((time.time() - start_time) * 1000)

        output_text = response.text or ""
        usage_meta = response.usage_metadata if hasattr(response, "usage_metadata") and response.usage_metadata else None
        usage = {
            "input": getattr(usage_meta, "prompt_token_count", 0) or 0,
            "output": getattr(usage_meta, "candidates_token_count", 0) or 0,
            "total": getattr(usage_meta, "total_token_count", 0) or 0,
        }

        # STEP 3: Output guardrail check - still no spans
        if guardrail_active and output_text:
            output_check = self._guardrails.check_output(output_text, input_text)
            if output_check.result == "BLOCKED":
                raise GuardrailViolationError("output", output_check)

        # STEP 4: Both checks passed - NOW create and end spans
        with start_trace_span(
            self._agent_name or f"gemini/{model}",
            input_data=contents,
            agent_id=self._agent_id,
            provider="gemini",
            tags=["ants-sdk", "gemini"],
        ) as trace_span:
            with start_generation_span(
                "generation",
                model=model,
                input_data=contents,
                provider="gemini",
            ) as gen_span:
                end_generation_span(
                    gen_span,
                    output_data=output_text,
                    usage=usage,
                    latency_ms=latency_ms,
                    guardrail_result="PASS" if guardrail_active else "NOT_CONFIGURED",
                )
                end_trace_span(trace_span, output_data=output_text)

        # Fallback: send via ingestion API if OTEL is not configured
        send_trace_via_ingestion(
            ants_api_key=self._ants_api_key,
            base_url=self._ants_base_url,
            model=model,
            provider="gemini",
            agent_id=self._agent_id,
            input_data=contents,
            output_data=output_text,
            usage=usage,
            latency_ms=latency_ms,
            tags=["ants-sdk", "gemini"],
            guardrail_result="PASS" if guardrail_active else "NOT_CONFIGURED",
        )

        return response


def _extract_input_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    parts: list[str] = []
    if isinstance(contents, list):
        for c in contents:
            if isinstance(c, dict):
                for p in c.get("parts", []):
                    if isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
    return "\n".join(parts)
