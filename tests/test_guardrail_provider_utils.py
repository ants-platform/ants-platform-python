from ants_platform.guardrails.providers._guardrail_utils import (
    effective_text,
    overall_guardrail_result,
)
from ants_platform.guardrails.types import GuardrailResult


def test_overall_guardrail_result_is_not_configured_when_guardrails_disabled() -> None:
    assert overall_guardrail_result(False, None, None) == "NOT_CONFIGURED"


def test_overall_guardrail_result_is_sanitized_when_input_is_sanitized() -> None:
    input_check = GuardrailResult(result="SANITIZED", sanitized_text="[REDACTED]")

    assert overall_guardrail_result(True, input_check, None) == "SANITIZED"


def test_overall_guardrail_result_is_sanitized_when_output_is_sanitized() -> None:
    output_check = GuardrailResult(result="SANITIZED", sanitized_text="[REDACTED]")

    assert overall_guardrail_result(True, None, output_check) == "SANITIZED"


def test_effective_text_prefers_sanitized_text_even_when_empty() -> None:
    check = GuardrailResult(result="SANITIZED", sanitized_text="")

    assert effective_text("secret", check) == ""


def test_effective_text_returns_original_for_pass_result() -> None:
    check = GuardrailResult(result="PASS")

    assert effective_text("safe", check) == "safe"
