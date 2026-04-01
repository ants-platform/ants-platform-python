from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import GuardrailResult


class GuardrailViolationError(Exception):
    """Raised when a guardrail check blocks content."""

    def __init__(self, direction: str, guardrail_result: GuardrailResult) -> None:
        violations_str = "; ".join(
            f"{v.scanner}: {v.details}" for v in guardrail_result.violations
        )
        message = f"Guardrail violation on {direction}: {violations_str or 'Content blocked'}"
        super().__init__(message)
        self.direction = direction
        self.guardrail_result = guardrail_result
