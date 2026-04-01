"""LLM provider wrappers with guardrail enforcement.

Each provider wrapper intercepts LLM calls to run guardrail checks on
input and output, raising ``GuardrailViolationError`` when policy is
violated.  Providers are optional imports — only installed dependencies
are loaded.
"""
