"""Guardrail checking client.

Guardrails are only active when an ``agent_id`` is provided AND the agent has
a guardrail policy configured on the platform. If no policy exists, the client
caches that fact and skips HTTP calls on subsequent requests.

Platform tracing (OTEL spans in ``default.traces``) is always recorded via
the provider wrappers regardless of guardrail state.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx

from .types import GuardrailResult

logger = logging.getLogger("ants_platform.guardrails")

DEFAULT_BASE_URL = "https://app.antsplatform.com"

# Returned when guardrails are disabled or no policy is configured.
_PASS_RESULT = GuardrailResult(
    result="PASS",
    risk_score=0.0,
    risk_level="LOW",
    sanitized_text=None,
    violations=[],
)


class AntsGuardrailsClient:
    """HTTP client for the ANTS guardrail ML-check API.

    Args:
        ants_api_key: API key in ``"publicKey:secretKey"`` format.
        base_url: Platform URL. Defaults to ``https://app.antsplatform.com``.
        agent_id: Agent ID for policy resolution (Redis-cached path).
        guardrail_service_url: URL of a self-hosted guardrail-ml-service.
        timeout: HTTP timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        ants_api_key: str,
        base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        guardrail_service_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url or DEFAULT_BASE_URL
        self._agent_id = agent_id
        self._guardrail_service_url = guardrail_service_url
        parts = ants_api_key.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "Invalid ants_api_key format. Expected 'publicKey:secretKey'."
            )
        credentials = base64.b64encode(
            f"{parts[0]}:{parts[1]}".encode()
        ).decode()
        self._auth_header = f"Basic {credentials}"
        self._client = httpx.Client(timeout=timeout)

        # Cache: None = not checked, True = policy exists, False = no policy
        self._policy_exists: Optional[bool] = None

    @property
    def enabled(self) -> bool:
        """Whether guardrail checks are enabled (requires agent_id)."""
        return bool(self._agent_id)

    # ── Public API ────────────────────────────────────────────────────────

    def check_input(self, text: str) -> GuardrailResult:
        """Check input text against configured guardrail policies."""
        return self._check(text, "input")

    def check_output(self, text: str, input_text: Optional[str] = None) -> GuardrailResult:
        """Check LLM output text against configured guardrail policies."""
        return self._check(text, "output", input_text)

    # ── Internal ──────────────────────────────────────────────────────────

    def _check(self, text: str, direction: str, input_text: Optional[str] = None) -> GuardrailResult:
        # No agentId → no guardrail configured → skip
        if not self._agent_id:
            return _PASS_RESULT

        # Cached: we already know there's no policy for this agent
        if self._policy_exists is False:
            return _PASS_RESULT

        body: dict = {"text": text, "direction": direction, "agentId": self._agent_id}
        if direction == "output" and input_text:
            body["inputText"] = input_text

        if self._guardrail_service_url:
            url = f"{self._guardrail_service_url}/api/v1/guardrail-check"
        else:
            url = f"{self._base_url}/api/public/v1/guardrails/ml-check"

        resp = self._client.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": self._auth_header,
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        result = GuardrailResult.from_dict(data)

        # If the response has no guardrailAction, no policy was found.
        # Cache this to skip future HTTP calls for this agent.
        if not data.get("guardrailAction"):
            self._policy_exists = False
            return _PASS_RESULT

        self._policy_exists = True
        return result

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
