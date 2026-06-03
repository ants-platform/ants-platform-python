---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: REQUIRED (Test-First, NON-NEGOTIABLE). Per the constitution, every new SDK behavior ships with pytest tests written FIRST, every bug fix ships with a regression test named after the issue, and framework-integration work (OpenAI/LangChain) is covered by tests, not manual checks. Test tasks below are not optional.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This is the `ants-platform-python` SDK (single Poetry package). Use the real layout:

- **SDK core (hand-written)**: `ants_platform/_client/<module>.py` (client, span, observe, datasets, environment_variables)
- **Background processing**: `ants_platform/_task_manager/<module>.py`
- **Integrations**: `ants_platform/openai.py`, `ants_platform/langchain/<module>.py`
- **Tests**: `tests/test_<name>.py` (pytest; `respx` for HTTP mocking, `pytest-httpserver` for servers)
- **OFF-LIMITS**: `ants_platform/api/` is Fern-GENERATED. Never add hand-edit tasks here — API-shape changes originate in `agentic-ants-lf-fork/fern/` and are regenerated.

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /speckit-tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Contracts/ (note: API contracts drive Fern regeneration upstream, not hand edits here)

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Confirm Poetry env + extras: `poetry install --all-extras`
- [ ] T002 Confirm pre-commit installed: `poetry run pre-commit install`
- [ ] T003 [P] Verify baseline gates are clean: `poetry run ruff check .` and `poetry run mypy .`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**[CRITICAL]**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your feature):

- [ ] T004 Add/extend shared types or config in `ants_platform/_client/environment_variables.py`
- [ ] T005 [P] Establish base span/attribute helpers in `ants_platform/_client/span.py` that all stories depend on
- [ ] T006 If the feature consumes new backend endpoints, confirm `ants_platform/api/` was REGENERATED from `agentic-ants-lf-fork/fern/` (do not hand-edit)
- [ ] T007 Set up shared test fixtures (respx mocks / pytest-httpserver) in `tests/`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) [MVP]

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 1 (REQUIRED - write FIRST, must FAIL before implementation)

- [ ] T010 [P] [US1] Behavior test for [SDK capability] in `tests/test_[name].py`
- [ ] T011 [P] [US1] Integration test for [OpenAI/LangChain path] in `tests/test_[name]_integration.py`

### Implementation for User Story 1

- [ ] T012 [P] [US1] Implement [helper/type] in `ants_platform/_client/[module].py`
- [ ] T013 [US1] Implement [client method/feature] in `ants_platform/_client/client.py` (depends on T012)
- [ ] T014 [US1] Wire background handling in `ants_platform/_task_manager/[module].py` (if batched/flushed)
- [ ] T015 [US1] Add Google-format docstrings to new public API surface
- [ ] T016 [US1] Run gates: `poetry run ruff format . && poetry run ruff check . && poetry run mypy . && poetry run pytest -s -v`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 2 (REQUIRED - write FIRST, must FAIL before implementation)

- [ ] T018 [P] [US2] Behavior test for [SDK capability] in `tests/test_[name].py`
- [ ] T019 [P] [US2] Integration test for [framework path] in `tests/test_[name]_integration.py`

### Implementation for User Story 2

- [ ] T020 [P] [US2] Implement [helper/type] in `ants_platform/[module].py`
- [ ] T021 [US2] Implement [feature] in `ants_platform/langchain/[module].py` or `ants_platform/openai.py`
- [ ] T022 [US2] Integrate with User Story 1 components (if needed)
- [ ] T023 [US2] Run gates (ruff format/check, mypy, pytest)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - [Title] (Priority: P3)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 3 (REQUIRED - write FIRST, must FAIL before implementation)

- [ ] T024 [P] [US3] Behavior test for [SDK capability] in `tests/test_[name].py`
- [ ] T025 [P] [US3] Integration test for [framework path] in `tests/test_[name]_integration.py`

### Implementation for User Story 3

- [ ] T026 [P] [US3] Implement [helper/type] in `ants_platform/[module].py`
- [ ] T027 [US3] Implement [feature] in `ants_platform/_client/[module].py`
- [ ] T028 [US3] Run gates (ruff format/check, mypy, pytest)

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Update docstrings + regenerate docs (`poetry run pdoc -o docs/ --docformat google ... ants_platform`)
- [ ] TXXX Update README.md / CONTRIBUTING.md if developer workflow changed
- [ ] TXXX Confirm SemVer impact + add DeprecationWarning + removal version for any deprecated public API
- [ ] TXXX Code cleanup: remove dead imports/exports so `ruff check .` is clean
- [ ] TXXX [P] Additional pytest cases for edge conditions in `tests/`
- [ ] TXXX Final full gate run: ruff format/check, mypy, pytest -n auto

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 -> P2 -> P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests MUST be written and FAIL before implementation (Test-First, NON-NEGOTIABLE)
- Types/helpers before client methods
- Client methods before task-manager/background wiring
- Core implementation before integration
- Story complete (gates green) before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (write them first, expect failures):
Task: "Behavior test for SDK capability in tests/test_[name].py"
Task: "Integration test for OpenAI/LangChain path in tests/test_[name]_integration.py"

# Then implement helpers/types in parallel where files differ:
Task: "Implement helper in ants_platform/_client/[module].py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational ([CRITICAL] - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently (gates green)
5. Release if ready (`poetry run release` — requires approval)

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 -> Test independently -> Release ([MVP]!)
3. Add User Story 2 -> Test independently -> Release
4. Add User Story 3 -> Test independently -> Release
5. Each story adds value without breaking previous stories (SemVer-respecting)

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Tests are REQUIRED and MUST fail before implementing (Test-First)
- `ants_platform/api/` is Fern-generated and off-limits — never add hand-edit tasks there
- Every public-API change: SemVer impact assessed, Google-format docstrings updated
- Commit after each task or logical group (commit/push require approval)
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
