# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]

**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  Pre-filled with this repo's real stack. Override only the fields a specific
  feature actually changes; mark anything genuinely unknown as NEEDS CLARIFICATION.
-->

**Language/Version**: Python 3.x (Poetry-managed; see `pyproject.toml` for the supported range)

**Primary Dependencies**: OpenTelemetry (tracing core), `httpx`/`respx` (transport + mocking), Fern-generated API client; framework integrations for OpenAI (`ants_platform/openai.py`) and LangChain (`ants_platform/langchain/`)

**Storage**: N/A — stateless client SDK; talks to `api.agenticants.ai`. Background batching/flush via `ants_platform/_task_manager/`

**Testing**: `pytest` (`poetry run pytest -s -v`, supports `-n auto`); `respx` for HTTP mocking, `pytest-httpserver` for test servers

**Target Platform**: Library — installed into customer Python applications (PyPI distribution)

**Project Type**: Client SDK (single package)

**Performance Goals**: [domain-specific, e.g., flush latency, batching throughput, negligible per-span overhead, or NEEDS CLARIFICATION]

**Constraints**: Async-first with background flushing; MUST NOT block the host application's hot path; MUST NOT hand-edit generated `ants_platform/api/`

**Scale/Scope**: [domain-specific, e.g., number of new public methods / integration surfaces touched, or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [ ] **Test-First**: pytest tests written before implementation; a regression test exists for each bug being fixed; framework-integration changes (OpenAI/LangChain) are covered by tests, not manual checks
- [ ] **Generated Code Is Sacred**: no hand-edits to `ants_platform/api/`; any API-shape change originates in `agentic-ants-lf-fork/fern/` and is regenerated, not patched here
- [ ] **Quality Gates**: `ruff format .`, `ruff check .`, `mypy .`, and `pytest` all clean; pre-commit hooks not bypassed
- [ ] **Backward Compatibility**: public-API changes respect SemVer; breaking changes carry a MAJOR bump + `DeprecationWarning` and a documented removal version; Google-format docstrings updated
- [ ] **No Dead Weight**: no unused imports/exports/modules; no untracked `# TODO`s introduced

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)
<!--
  This is the actual ants-platform-python layout. Most features add to
  ants_platform/_client/ or an integration module and land tests under tests/.
  Expand with the concrete files this feature touches; do NOT add a src/ tree.
-->

```text
ants_platform/
├── _client/                # Hand-written SDK core (OTel-based): client.py, span.py,
│                           #   observe.py, datasets.py, environment_variables.py
├── api/                    # Fern-GENERATED API client — DO NOT EDIT by hand.
│                           #   Regenerate from agentic-ants-lf-fork/fern/.
├── _task_manager/          # Background processing: media upload, score ingestion,
│                           #   queue management, batching/flush
├── openai.py               # OpenAI instrumentation (integration module)
├── langchain/              # LangChain integration (CallbackHandler)
└── version.py              # Version string (updated by `poetry run release`)

tests/
└── test_<name>.py          # pytest; respx for HTTP mocking, pytest-httpserver for servers
```

**Structure Decision**: Single Python package (`ants_platform`). New hand-written
behavior belongs in `_client/`, `_task_manager/`, or an integration module
(`openai.py`, `langchain/`). `ants_platform/api/` is off-limits — API-shape changes
flow from the backend Fern spec and are regenerated. Tests live flat under `tests/`
as `test_<name>.py`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
