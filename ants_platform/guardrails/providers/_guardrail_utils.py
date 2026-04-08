from __future__ import annotations

from typing import Optional

from ..types import GuardrailResult


def overall_guardrail_result(
    guardrail_active: bool,
    input_check: Optional[GuardrailResult],
    output_check: Optional[GuardrailResult],
) -> str:
    if not guardrail_active:
        return "NOT_CONFIGURED"
    if _is_sanitized(input_check) or _is_sanitized(output_check):
        return "SANITIZED"
    return "PASS"


def effective_text(original_text: str, check: Optional[GuardrailResult]) -> str:
    if check and check.result == "SANITIZED" and check.sanitized_text is not None:
        return check.sanitized_text
    return original_text


def _is_sanitized(check: Optional[GuardrailResult]) -> bool:
    return bool(check and check.result == "SANITIZED")
