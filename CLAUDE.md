# ants-platform-python — Claude Code orientation

Python SDK (`ants-platform` on PyPI) for the Agentic Ants platform. Talks to `api.agenticants.ai`. OpenTelemetry-first tracing, auto-instrumentation for OpenAI / LangChain / CrewAI, plus a Fern-generated REST client. Currently `3.6.0` (the `3.x` is inherited from the Langfuse SDK lineage; backend is `0.1.x`).

## Layout

```
ants_platform/
  _client/                 Core SDK on top of OTel.
    client.py              AntsPlatform client + lifecycle.
    span.py                AntsPlatformSpan / Generation / Event wrappers.
    observe.py             @observe decorator.
    datasets.py            Dataset operations.
    environment_variables.py
  api/                     Fern-generated REST client. DO NOT hand-edit.
  _task_manager/           Background batching, media upload, score ingestion.
  _utils/, _crewai_bootstrap.py
  openai.py                OpenAI auto-instrumentation.
  langchain/               LangChain CallbackHandler integration.
  crewai/                  CrewAI integration (gated by py>=3.10,<3.14).
  guardrails/              Guardrails client.
  cli/                     CLI entrypoints.
  media.py, model.py, types.py, logger.py, version.py
tests/                     pytest, respx (HTTP mocking), pytest-httpserver.
scripts/                   Release / housekeeping scripts.
pyproject.toml             Poetry. Python >=3.9,<4.0.
ruff.toml / ci.ruff.toml   Local (strict) vs CI (permissive) lint configs.
```

## Commands

```sh
poetry self add poetry-dotenv-plugin poetry-bumpversion        # one-time
poetry install --all-extras
poetry run pre-commit install

poetry run pytest -s -v --log-cli-level=INFO                   # all tests
poetry run pytest -s -v tests/test_core_sdk.py::test_flush     # single test
poetry run pytest -n auto                                      # parallel

poetry run ruff format .
poetry run ruff check .
poetry run mypy .

poetry build
poetry run release                                             # versioning + build + tag + publish
poetry run pdoc -o docs/ --docformat google --logo "static/Ants_Platform_Blended.png" ants_platform
```

E2E tests against real OpenAI / SERP keys are decorated `@pytest.mark.skip` by default — remove the marker locally to run them. Create `.env` from `.env.template` for integration tests.

## Things to know

- **Fern-generated `ants_platform/api/`.** Regenerate by running Fern in `agentic-ants-lf-fork`, copying `generated/python` over `ants_platform/api/`, then `poetry run ruff format .`. Hand edits get clobbered.
- **Default host is stale.** `_client/environment_variables.py` and `client.py` hard-code `https://cloud.ants-platform.com` as the default. The real production host is `https://api.agenticants.ai`. Until the default is updated, every consumer must set `ANTS_PLATFORM_HOST` explicitly. (TODO: fix the default to point at `api.agenticants.ai`.)
- **Env var prefix.** `ANTS_PLATFORM_PUBLIC_KEY`, `ANTS_PLATFORM_SECRET_KEY`, `ANTS_PLATFORM_HOST`, `ANTS_PLATFORM_DEBUG`, `ANTS_PLATFORM_TRACING_ENABLED`, `ANTS_PLATFORM_SAMPLE_RATE`. Don't introduce other prefixes (e.g. `LANGFUSE_*`, `AGENTICANTS_*`).
- **Async-first batching.** Spans flush via background workers in `_task_manager/`. Tests that assert on emitted spans must call `client.flush()` first.
- **Pydantic 1 + 2 supported** (`pydantic = ">=1.10.7, <3.0"`). Don't use Pydantic-2-only syntax in shared code.
- **Optional extras**: `openai`, `langchain`, `crewai`. Don't move these into the base dependency block — installs would explode.
- **Exception messages**: don't put f-strings directly in `raise SomeException(f"...")`; assign to a variable first (existing repo convention; flagged by lint).
- **Don't remove unit-test cases just to make them pass.** Adjust the test only when underlying behavior intentionally changed.

## Compatibility

| | |
|---|---|
| Python | 3.9 – 3.13 (CrewAI extra: 3.10 – 3.13) |
| Backend | `agentic-ants-lf-fork` v0.1.x (`api.agenticants.ai`) |
| OTel | `opentelemetry-api/sdk/exporter-otlp-proto-http ^1.33.1` |

## Approval boundaries

`poetry install`, `pytest`, `ruff`, `mypy`, `poetry build`, file edits — fine. `poetry run release` (publishes to PyPI), `git commit/push`, version bumps — ask first.
