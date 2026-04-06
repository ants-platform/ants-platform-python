""".. include:: ../README.md"""

from ._client import client as _client_module
from ._client.attributes import AntsPlatformOtelSpanAttributes
from ._client.constants import ObservationTypeLiteral
from ._client.get_client import get_client
from ._client.observe import observe
from ._client.span import (
    AntsPlatformEvent,
    AntsPlatformGeneration,
    AntsPlatformSpan,
    AntsPlatformAgent,
    AntsPlatformTool,
    AntsPlatformChain,
    AntsPlatformEmbedding,
    AntsPlatformEvaluator,
    AntsPlatformRetriever,
    AntsPlatformGuardrail,
)

AntsPlatform = _client_module.AntsPlatform

# Guardrails (available as ants_platform.guardrails or top-level imports)
from .guardrails import (
    AntsGuardrailsClient,
    GuardrailViolationError,
    GuardrailResult,
    Violation,
)

__all__ = [
    "AntsPlatform",
    "get_client",
    "observe",
    "ObservationTypeLiteral",
    "AntsPlatformSpan",
    "AntsPlatformGeneration",
    "AntsPlatformEvent",
    "AntsPlatformOtelSpanAttributes",
    "AntsPlatformAgent",
    "AntsPlatformTool",
    "AntsPlatformChain",
    "AntsPlatformEmbedding",
    "AntsPlatformEvaluator",
    "AntsPlatformRetriever",
    "AntsPlatformGuardrail",
    # Guardrails
    "AntsGuardrailsClient",
    "GuardrailViolationError",
    "GuardrailResult",
    "Violation",
]
