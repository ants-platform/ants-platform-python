# [PROJECT_NAME] Constitution
<!-- Example: Ants Platform Python SDK Constitution -->

<!-- Optional: one-line scope note + pointer to the canonical cross-repo constitution. -->
> [SCOPE_NOTE]
<!-- Example: The canonical cross-repo constitution lives in agentic-ants-lf-fork/.specify/memory/constitution.md. This document adapts it to the Python SDK. SDK versioning is per-repo and independent of the backend. -->

## Core Principles

### I. Own What You Ship
<!-- Accountability for every merged change, including AI-generated code. State the blast radius (this SDK runs in customer production). -->
[PRINCIPLE_OWN_WHAT_YOU_SHIP]
<!-- Use MUST / MUST NOT bullets. Example: MUST understand every change before merging; MUST be able to explain any committed code in review; MUST NOT merge AI output without verifying logic, types, and backward compatibility. -->

### II. Test-First (NON-NEGOTIABLE)
<!-- pytest tests precede implementation; regression test per bug fix; framework integrations covered by tests, not manual checks. -->
[PRINCIPLE_TEST_FIRST]
<!-- Example: MUST add pytest tests for new behavior (poetry run pytest -s -v); MUST add a regression test per bug fix; MUST NOT reduce coverage of touched modules. -->

### III. Generated Code Is Sacred
<!-- The Fern-generated API client tree is off-limits to hand edits. Name the directory and the regeneration source of truth. -->
[PRINCIPLE_GENERATED_CODE]
<!-- Example: ants_platform/api/ is Fern-generated from the backend OpenAPI spec — MUST NOT hand-edit; API changes start in agentic-ants-lf-fork/fern/, regenerate, then land here. -->

### IV. No Dead Weight
<!-- No unused imports/exports/modules; no untracked TODOs; deprecations carry a DeprecationWarning + removal version. -->
[PRINCIPLE_NO_DEAD_WEIGHT]
<!-- Example: MUST NOT leave unused imports/exports — ruff check . must be clean; MUST NOT leave # TODO without a tracked ticket. -->

### V. Quality Gates Are Mandatory
<!-- Enumerate the exact gate commands that MUST pass before merge. Hooks MUST NOT be bypassed. -->
[PRINCIPLE_QUALITY_GATES]
<!-- Example: poetry run ruff format . / ruff check . / mypy . / pytest must all pass; MUST NOT bypass pre-commit hooks. -->

### VI. Backward Compatibility and Documentation
<!-- SemVer discipline, deprecation window for breaking public-API changes, docstring + README/CONTRIBUTING updates. -->
[PRINCIPLE_BACKWARD_COMPAT_DOCS]
<!-- Example: MUST follow SemVer — breaking public-API changes require a MAJOR bump + deprecation window; MUST update Google-format docstrings (surfaced via pdoc); MUST update CONTRIBUTING.md / README.md when developer workflow changes. -->

## Governance
<!-- What needs approval, what does not, and how this document is amended. -->

[GOVERNANCE_RULES]
<!-- Example: Edits, builds, pytest, lint, type-check — no approval needed. git commit/push and poetry run release (publishes to PyPI) — require explicit approval. Amendments: PR + reviewer + rationale, semantic versioning of this document. -->

**Version**: [CONSTITUTION_VERSION] | **Ratified**: [RATIFICATION_DATE] | **Last Amended**: [LAST_AMENDED_DATE]
<!-- Example: Version: 1.0.0 | Ratified: 2026-06-03 | Last Amended: 2026-06-03 -->
