"""Internal OTEL tracing helpers for guardrail provider wrappers.

Auto-detects if OTEL TracerProvider is configured. If not, falls back to
sending traces via the /api/public/ingestion HTTP API.
Uses the exact same attribute names as the core ants-platform SDK so traces
appear identically in the platform UI.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger("ants_platform.guardrails.tracing")

# OTEL attribute names (must match ants_platform SDK / OtelIngestionProcessor)
_T = "ants_platform.trace"
_O = "ants_platform.observation"
_A = "ants_platform.agent"

try:
    from opentelemetry import trace as otel_trace

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    otel_trace = None  # type: ignore[assignment]


def _get_tracer():
    if not _OTEL_AVAILABLE:
        return None
    provider = otel_trace.get_tracer_provider()
    # Check if a real provider is configured (not the default NoOp)
    if type(provider).__name__ == "ProxyTracerProvider":
        inner = getattr(provider, "_real_provider", None)
        if inner is None or type(inner).__name__ == "NoOpTracerProvider":
            return None
    return otel_trace.get_tracer("ants_platform.guardrails")


def _otel_active() -> bool:
    return _get_tracer() is not None


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


class _NoopSpan:
    """Drop-in replacement when OTEL is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@contextmanager
def start_trace_span(
    name: str,
    *,
    input_data: Any = None,
    agent_id: Optional[str] = None,
    provider: Optional[str] = None,
    tags: Optional[list[str]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """Create a root trace span with standard platform attributes."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return

    with tracer.start_as_current_span(name) as span:
        if input_data is not None:
            span.set_attribute(f"{_T}.input", _serialize(input_data))
        if agent_id:
            span.set_attribute(f"{_A}.id", agent_id)
        if user_id:
            span.set_attribute("user.id", user_id)
        if session_id:
            span.set_attribute("session.id", session_id)
        if tags:
            span.set_attribute(f"{_T}.tags", json.dumps(tags))
        if metadata:
            span.set_attribute(f"{_T}.metadata", json.dumps(metadata))
        if provider:
            span.set_attribute(f"{_T}.metadata.provider", provider)

        yield span


@contextmanager
def start_generation_span(
    name: str,
    *,
    model: str,
    input_data: Any = None,
    provider: Optional[str] = None,
):
    """Create a child generation span with model/usage attributes."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return

    with tracer.start_as_current_span(name) as span:
        span.set_attribute(f"{_O}.type", "GENERATION")
        span.set_attribute(f"{_O}.model.name", model)
        if input_data is not None:
            span.set_attribute(f"{_O}.input", _serialize(input_data))
        if provider:
            span.set_attribute(f"{_O}.metadata.provider", provider)

        yield span


def end_generation_span(
    span: Any,
    *,
    output_data: Any = None,
    usage: Optional[dict] = None,
    cost: Optional[dict] = None,
    latency_ms: Optional[int] = None,
    level: str = "DEFAULT",
    status_message: Optional[str] = None,
    guardrail_result: Optional[str] = None,
) -> None:
    """Finalize a generation span with output, usage, cost."""
    if isinstance(span, _NoopSpan):
        return

    if output_data is not None:
        span.set_attribute(f"{_O}.output", _serialize(output_data))
    if usage:
        span.set_attribute(f"{_O}.usage_details", json.dumps(usage))
    if cost:
        span.set_attribute(f"{_O}.cost_details", json.dumps(cost))
    if level != "DEFAULT":
        span.set_attribute(f"{_O}.level", level)
    if status_message:
        span.set_attribute(f"{_O}.status_message", status_message)
    if latency_ms is not None:
        span.set_attribute(f"{_O}.metadata.latencyMs", str(latency_ms))
    if guardrail_result:
        span.set_attribute(f"{_O}.metadata.guardrailResult", guardrail_result)


def end_trace_span(span: Any, *, output_data: Any = None) -> None:
    """Set output on a trace span."""
    if isinstance(span, _NoopSpan):
        return
    if output_data is not None:
        span.set_attribute(f"{_T}.output", _serialize(output_data))


# ── Ingestion API fallback (when OTEL is not configured) ─────────────────

def send_trace_via_ingestion(
    *,
    ants_api_key: str,
    base_url: str,
    model: str,
    provider: str,
    agent_id: Optional[str] = None,
    input_data: Any = None,
    output_data: Any = None,
    usage: Optional[dict] = None,
    latency_ms: Optional[int] = None,
    tags: Optional[list[str]] = None,
    guardrail_result: Optional[str] = None,
) -> None:
    """Send trace via /api/public/ingestion when OTEL is not configured.
    Fire-and-forget in a background thread."""
    if _otel_active():
        return  # OTEL handles it

    def _send():
        try:
            import httpx

            parts = ants_api_key.split(":")
            creds = base64.b64encode(f"{parts[0]}:{parts[1]}".encode()).decode()
            trace_id = str(uuid.uuid4())
            obs_id = str(uuid.uuid4())
            now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

            input_tokens = (usage or {}).get("input", 0)
            output_tokens = (usage or {}).get("output", 0)
            total_tokens = (usage or {}).get("total", 0)

            all_tags = list(tags or []) + ["ants-sdk"]
            if agent_id:
                all_tags.append(f"agent:{agent_id}")

            trace_metadata = {
                "source": "ants-sdk",
                "source_platform": provider,
                "provider": provider,
                "model": model,
                "status": "success",
                "input_tokens": str(input_tokens),
                "output_tokens": str(output_tokens),
                "total_tokens": str(total_tokens),
            }
            if agent_id:
                trace_metadata["agent_id"] = agent_id
            if latency_ms is not None:
                trace_metadata["latency_ms"] = str(latency_ms)
            if guardrail_result:
                trace_metadata["guardrail_result"] = guardrail_result

            batch = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "timestamp": now,
                    "body": {
                        "id": trace_id,
                        "timestamp": now,
                        "name": f"{provider}/{model}",
                        "input": input_data,
                        "output": output_data,
                        "tags": all_tags,
                        "metadata": trace_metadata,
                        "agentId": agent_id,
                    },
                },
                {
                    "id": str(uuid.uuid4()),
                    "type": "generation-create",
                    "timestamp": now,
                    "body": {
                        "id": obs_id,
                        "traceId": trace_id,
                        "name": model,
                        "startTime": now,
                        "endTime": now,
                        "model": model,
                        "input": input_data,
                        "output": output_data,
                        "usage": {
                            "promptTokens": input_tokens,
                            "completionTokens": output_tokens,
                            "totalTokens": total_tokens,
                        } if usage else None,
                        "agentId": agent_id,
                        "metadata": {
                            "source_platform": provider,
                            "provider": provider,
                            "model": model,
                            "latency_ms": str(latency_ms) if latency_ms else None,
                            "guardrail_result": guardrail_result,
                        },
                    },
                },
            ]

            httpx.post(
                f"{base_url}/api/public/ingestion",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {creds}",
                },
                json={"batch": batch},
                timeout=10,
            )
        except Exception as e:
            logger.debug(f"Ingestion fallback failed: {e}")

    threading.Thread(target=_send, daemon=True).start()
