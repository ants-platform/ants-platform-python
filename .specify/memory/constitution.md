# Ants Platform Python SDK Constitution

> The Python client SDK for the Agentic Ants platform. The canonical cross-repo constitution lives in `agentic-ants-lf-fork/.specify/memory/constitution.md`. This document adapts the shared principles to the Python SDK. SDK versioning is per-repo and independent of the backend.

## Core Principles

### I. Own What You Ship

This SDK runs inside customers' production code. A bug here breaks their observability, not just ours.

- MUST understand every change before merging, including AI-generated code
- MUST be able to explain any committed code in review
- MUST NOT merge AI output without verifying logic, types, and backward compatibility

### II. Test-First (NON-NEGOTIABLE)

- MUST add `pytest` tests for new SDK behavior (`poetry run pytest -s -v`)
- MUST add a regression test for every bug fix, named after the issue
- MUST NOT merge changes that reduce coverage of touched modules
- Integration with instrumented frameworks (OpenAI, LangChain) MUST be covered by tests, not just manual checks

### III. Generated Code Is Sacred

- `ants_platform/api/` is **Fern-generated** from the backend OpenAPI spec — MUST NOT hand-edit these files
- API changes start in `agentic-ants-lf-fork/fern/`, regenerate, then land here
- Hand-written SDK ergonomics (`_client/`, integrations) live outside the generated tree and are owned here

### IV. No Dead Weight

- MUST NOT leave unused imports, exports, or modules — `ruff check .` must be clean
- MUST NOT leave `# TODO` without a tracked ticket
- Deprecations require a `DeprecationWarning` + a removal version, honoring SemVer

### V. Quality Gates Are Mandatory

Every change MUST pass the full pre-commit / CI gate before merge:

- `poetry run ruff format .` (formatting)
- `poetry run ruff check .` (lint)
- `poetry run mypy .` (type checking — type safety is the first defense)
- `poetry run pytest` (tests)
- MUST NOT bypass pre-commit hooks

### VI. Backward Compatibility and Documentation

- MUST follow SemVer: breaking public-API changes require a MAJOR bump and a deprecation window — never silently break a backend deploy's worth of users
- MUST update docstrings (Google format, surfaced via `pdoc`) for public API changes
- MUST update `CONTRIBUTING.md` / `README.md` when developer workflow changes

## Governance

- Edits, builds, `pytest`, lint, type-check — no approval needed
- `git commit/push`, `poetry run release` (publishes to PyPI) — require explicit approval
- Amendments: PR + reviewer + rationale. Semantic versioning of this document

**Version**: 1.0.0 | **Ratified**: 2026-06-03 | **Last Amended**: 2026-06-03
