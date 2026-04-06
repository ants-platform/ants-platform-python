"""ANTS Guardrails — LLM input/output policy enforcement.

Works in two modes:

**Mode 1: Standalone** (guardrails only, no platform tracing)::

    from ants_platform.guardrails import AntsGuardrailsClient, GuardrailViolationError

    guardrails = AntsGuardrailsClient(
        ants_api_key="pk:sk",
        agent_id="agent_123",
    )

    result = guardrails.check_input("My SSN is 123-45-6789")
    if result.result == "BLOCKED":
        raise GuardrailViolationError("input", result)

**Mode 2: Platform-integrated** (auto-creates guardrail spans)::

    from ants_platform import AntsPlatform, observe
    from ants_platform.guardrails import AntsGuardrailsClient

    # Initialize platform (enables tracing)
    ants = AntsPlatform(public_key="pk", secret_key="sk")
    guardrails = AntsGuardrailsClient(ants_api_key="pk:sk", agent_id="agent_123")

    @observe(as_type="generation")
    def call_llm(prompt):
        check = guardrails.check_input(prompt)   # auto-creates guardrail child span
        ...

**Provider wrappers** (drop-in replacements with guardrails built-in)::

    from ants_platform.guardrails.providers.bedrock import AntsBedrock
    from ants_platform.guardrails.providers.anthropic import AntsAnthropic
    from ants_platform.guardrails.providers.openai_provider import AntsOpenAI
    from ants_platform.guardrails.providers.google_genai import AntsGoogleGenAI
    from ants_platform.guardrails.providers.vertex import AntsVertexAI
"""

from .client import AntsGuardrailsClient
from .errors import GuardrailViolationError
from .types import GuardrailResult, Violation

__all__ = [
    "AntsGuardrailsClient",
    "GuardrailViolationError",
    "GuardrailResult",
    "Violation",
]
