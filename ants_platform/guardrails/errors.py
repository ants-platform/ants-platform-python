from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import GuardrailResult


class GuardrailViolationError(Exception):
    """Raised when a guardrail check blocks content."""

    def __init__(self, direction: str, guardrail_result: GuardrailResult) -> None:
        blocked_message = (guardrail_result.blocked_message or "").strip()
        if blocked_message:
            message = blocked_message
        else:
            violations_str = "; ".join(
                f"{v.scanner}: {v.details}" for v in guardrail_result.violations
            )
            message = (
                f"Guardrail violation on {direction}: "
                f"{violations_str or 'Content blocked'}"
            )
        super().__init__(message)
        self.direction = direction
        self.guardrail_result = guardrail_result
