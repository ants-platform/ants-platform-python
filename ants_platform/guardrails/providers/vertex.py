from __future__ import annotations

import time
from typing import Any, Optional

from ..client import AntsGuardrailsClient
from ..errors import GuardrailViolationError
from .._tracing import start_trace_span, start_generation_span, end_generation_span, end_trace_span, send_trace_via_ingestion
from ._guardrail_utils import effective_text, overall_guardrail_result


class AntsVertexAI:
    """Vertex AI client wrapper with guardrail enforcement + automatic OTEL tracing.

    OTEL traces are only created when guardrail checks pass (PASS or SANITIZED).
    Blocked requests produce no OTEL trace.

    Usage::

        from ants_platform.guardrails.providers.vertex import AntsVertexAI

        client = AntsVertexAI(
            project="my-project",
            location="us-central1",
            ants_api_key="pk:sk",
            agent_id="agent_123",
        )
        model = client.get_generative_model("gemini-1.5-pro")
        response = model.generate_content("Hello")
    """

    def __init__(
        self,
        *,
        project: str,
        location: str,
        ants_api_key: str,
        ants_base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        guardrail_service_url: Optional[str] = None,
    ) -> None:
        import vertexai

        vertexai.init(project=project, location=location)
        self._guardrails = AntsGuardrailsClient(
            ants_api_key, ants_base_url, agent_id, guardrail_service_url
        )
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url or "https://app.antsplatform.com"

    def get_generative_model(self, model_name: str) -> _GuardedModel:
        from vertexai.generative_models import GenerativeModel

        model = GenerativeModel(model_name)
        return _GuardedModel(model, model_name, self._guardrails, self._agent_id, self._agent_name, self._ants_api_key, self._ants_base_url)


class _GuardedModel:
    def __init__(
        self, model: Any, model_name: str, guardrails: AntsGuardrailsClient, agent_id: Optional[str], agent_name: Optional[str] = None, ants_api_key: str = "", ants_base_url: str = ""
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._guardrails = guardrails
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._ants_api_key = ants_api_key
        self._ants_base_url = ants_base_url

    def generate_content(self, contents: Any, **kwargs: Any) -> Any:
        input_text = _extract_input_text(contents)
        guardrail_active = self._guardrails.enabled
        input_check = None
        output_check = None

        # STEP 1: Input guardrail check - no spans yet
        effective_contents = contents
        if guardrail_active:
            input_check = self._guardrails.check_input(input_text)
            if input_check.result == "BLOCKED":
                raise GuardrailViolationError("input", input_check)
            if input_check.result == "SANITIZED" and input_check.sanitized_text is not None:
                effective_contents = input_check.sanitized_text
        effective_input_text = effective_text(input_text, input_check)

        # STEP 2: LLM call
        start_time = time.time()
        response = self._model.generate_content(effective_contents, **kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        output_text = response.text if hasattr(response, "text") else ""
        usage_meta = response.usage_metadata if hasattr(response, "usage_metadata") and response.usage_metadata else None
        usage = {
            "input": getattr(usage_meta, "prompt_token_count", 0) or 0,
            "output": getattr(usage_meta, "candidates_token_count", 0) or 0,
            "total": getattr(usage_meta, "total_token_count", 0) or 0,
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
            self._agent_name or f"vertex/{self._model_name}",
            input_data=effective_contents,
            agent_id=self._agent_id,
            provider="vertex",
            tags=["ants-sdk", "vertex"],
        ) as trace_span:
            with start_generation_span(
                "generation",
                model=self._model_name,
                input_data=effective_contents,
                provider="vertex",
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
            model=self._model_name,
            provider="vertex",
            agent_id=self._agent_id,
            input_data=effective_contents,
            output_data=output_text,
            usage=usage,
            latency_ms=latency_ms,
            tags=["ants-sdk", "vertex"],
            guardrail_result=guardrail_result,
        )

        return response


def _extract_input_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        parts: list[str] = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "parts"):
                for p in item.parts:
                    if hasattr(p, "text"):
                        parts.append(p.text)
        return "\n".join(parts)
    return str(contents)


def _apply_sanitized_output(response: Any, output_text: str) -> None:
    try:
        response.text = output_text
    except Exception:
        pass

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return

    replaced = False
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            if not hasattr(part, "text"):
                continue
            try:
                part.text = output_text if not replaced else ""
                replaced = True
            except Exception:
                pass
