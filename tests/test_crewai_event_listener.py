"""Tests for CrewAI EventListener integration.

Follows the same InMemorySpanExporter + monkeypatch pattern as test_otel.py.
Constructs the EventListener manually (bypassing crewai imports) to test
event handler logic with a mock event bus.
"""

import json
import threading
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

import pytest
from opentelemetry import context, trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from ants_platform._client.attributes import AntsPlatformOtelSpanAttributes
from ants_platform._client.client import AntsPlatform
from ants_platform._client.resource_manager import LangfuseResourceManager


# ── In-Memory Span Exporter (same as test_otel.py) ────────────────────


class InMemorySpanExporter(SpanExporter):
    """Thread-safe in-memory exporter to collect spans for testing."""

    def __init__(self):
        self._finished_spans: List[ReadableSpan] = []
        self._stopped = False
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._stopped:
            return SpanExportResult.FAILURE
        with self._lock:
            self._finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self._stopped = True

    def get_finished_spans(self) -> List[ReadableSpan]:
        with self._lock:
            return list(self._finished_spans)

    def clear(self):
        with self._lock:
            self._finished_spans.clear()


# ── Mock CrewAI Event Bus ──────────────────────────────────────────────


class MockEventBus:
    """Simulates CrewAI's event bus for testing."""

    def __init__(self):
        self._handlers: Dict[type, list] = {}

    def on(self, event_type):
        """Decorator to register an event handler."""

        def decorator(fn):
            self._handlers.setdefault(event_type, []).append(fn)
            return fn

        return decorator

    def emit(self, event_type, source: Any, event: Any) -> None:
        """Fire all registered handlers for an event type."""
        for handler in self._handlers.get(event_type, []):
            handler(source, event)


# ── Mock CrewAI Event Classes ─────────────────────────────────────────
# Lightweight stand-ins keyed by class name — the EventListener registers
# handlers by event *type*, so these must be distinct classes.


class _MockEvent:
    """Base mock event with arbitrary attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CrewKickoffStartedEvent(_MockEvent):
    pass


class CrewKickoffCompletedEvent(_MockEvent):
    pass


class CrewKickoffFailedEvent(_MockEvent):
    pass


class AgentExecutionStartedEvent(_MockEvent):
    pass


class AgentExecutionCompletedEvent(_MockEvent):
    pass


class AgentExecutionErrorEvent(_MockEvent):
    pass


class TaskStartedEvent(_MockEvent):
    pass


class TaskCompletedEvent(_MockEvent):
    pass


class TaskFailedEvent(_MockEvent):
    pass


class LLMCallStartedEvent(_MockEvent):
    pass


class LLMCallCompletedEvent(_MockEvent):
    pass


class LLMCallFailedEvent(_MockEvent):
    pass


class ToolUsageStartedEvent(_MockEvent):
    pass


class ToolUsageFinishedEvent(_MockEvent):
    pass


class ToolUsageErrorEvent(_MockEvent):
    pass


# ── Lightweight EventListener (no crewai import) ─────────────────────


def _build_event_listener(client: AntsPlatform, bus: MockEventBus):
    """Construct an EventListener-like object and wire it to the mock bus.

    This avoids importing crewai entirely. We import the real event_listener
    module's EventListener class at runtime after injecting mock crewai
    modules into sys.modules, then manually initialise it.
    """
    import sys

    # Map our local mock event classes so the bus.on() decorators register
    # against the same types we emit in tests.
    # A real class for BaseEventListener — EventListener inherits from it,
    # so it can't be a MagicMock (object.__new__ would fail).
    class _FakeBaseEventListener:
        def __init__(self):
            pass

        def setup_listeners(self, bus):
            pass

        def validate_dependencies(self):
            pass

    base_module = _make_module_mock(BaseEventListener=_FakeBaseEventListener)

    _MOCK_CREWAI_MODULES = {
        "crewai": MagicMock(),
        "crewai.events": MagicMock(),
        "crewai.events.base_event_listener": base_module,
        "crewai.events.event_bus": MagicMock(crewai_event_bus=bus),
        "crewai.events.types": MagicMock(),
        "crewai.events.types.crew_events": _make_module_mock(
            CrewKickoffStartedEvent=CrewKickoffStartedEvent,
            CrewKickoffCompletedEvent=CrewKickoffCompletedEvent,
            CrewKickoffFailedEvent=CrewKickoffFailedEvent,
        ),
        "crewai.events.types.agent_events": _make_module_mock(
            AgentExecutionStartedEvent=AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent=AgentExecutionCompletedEvent,
            AgentExecutionErrorEvent=AgentExecutionErrorEvent,
        ),
        "crewai.events.types.task_events": _make_module_mock(
            TaskStartedEvent=TaskStartedEvent,
            TaskCompletedEvent=TaskCompletedEvent,
            TaskFailedEvent=TaskFailedEvent,
        ),
        "crewai.events.types.llm_events": _make_module_mock(
            LLMCallStartedEvent=LLMCallStartedEvent,
            LLMCallCompletedEvent=LLMCallCompletedEvent,
            LLMCallFailedEvent=LLMCallFailedEvent,
        ),
        "crewai.events.types.tool_usage_events": _make_module_mock(
            ToolUsageStartedEvent=ToolUsageStartedEvent,
            ToolUsageFinishedEvent=ToolUsageFinishedEvent,
            ToolUsageErrorEvent=ToolUsageErrorEvent,
        ),
    }

    saved = {}
    for name, mock_mod in _MOCK_CREWAI_MODULES.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock_mod

    # Also remove phase-2 modules so the optional imports raise ImportError
    for phase2 in [
        "crewai.events.types.llm_guardrail_events",
        "crewai.events.types.reasoning_events",
    ]:
        saved[phase2] = sys.modules.pop(phase2, None)

    # Remove cached event_listener module so it re-imports with our mocks
    el_key = "ants_platform.crewai.event_listener"
    crewai_init_key = "ants_platform.crewai"
    saved[el_key] = sys.modules.pop(el_key, None)
    saved[crewai_init_key] = sys.modules.pop(crewai_init_key, None)

    # Now construct the listener manually — skip BaseEventListener.__init__
    from ants_platform.crewai.event_listener import EventListener

    # Create instance without calling __init__ (avoids BaseEventListener)
    listener = object.__new__(EventListener)
    listener.client = client
    listener._lock = threading.Lock()
    listener._crew_spans = {}
    listener._agent_spans = {}
    listener._task_spans = {}
    listener._llm_spans = {}
    listener._tool_spans = {}
    listener._guardrail_spans = {}
    listener._reasoning_spans = {}
    listener._context_tokens = {}
    listener._thread_local = threading.local()
    listener._user_tags = {}
    listener._explicit_agent_name = None
    listener._explicit_agent_display_name = None

    # Register event handlers on our mock bus
    listener.setup_listeners(bus)

    # Restore sys.modules
    for name, original in saved.items():
        if original is not None:
            sys.modules[name] = original
        else:
            sys.modules.pop(name, None)

    return listener


def _make_module_mock(**attrs):
    """Create a MagicMock that exposes specific attributes (event classes)."""
    m = MagicMock()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ── Test Base Class ───────────────────────────────────────────────────


class TestCrewAIBase:
    """Base class providing OTel fixtures and CrewAI mock infrastructure."""

    @pytest.fixture(scope="function", autouse=True)
    def cleanup_otel(self):
        """Reset OpenTelemetry state between tests."""
        original_provider = trace_api.get_tracer_provider()
        yield
        # Clear any leftover OTel baggage from this test
        from opentelemetry import baggage as otel_baggage

        ctx = context.get_current()
        for key in ["agent_name", "agent_id", "agent_display_name"]:
            ctx = otel_baggage.remove_baggage(key, ctx)
        context.attach(ctx)
        trace_api.set_tracer_provider(original_provider)
        LangfuseResourceManager.reset()

    @pytest.fixture
    def memory_exporter(self):
        """Create an in-memory span exporter for testing."""
        exporter = InMemorySpanExporter()
        yield exporter
        exporter.shutdown()

    @pytest.fixture
    def tracer_provider(self, memory_exporter):
        """Create a fresh tracer provider with our memory exporter."""
        resource = Resource.create({"service.name": "ants_platform-crewai-test"})
        provider = TracerProvider(resource=resource)
        processor = SimpleSpanProcessor(memory_exporter)
        provider.add_span_processor(processor)
        trace_api.set_tracer_provider(provider)
        return provider

    @pytest.fixture
    def mock_processor_init(self, monkeypatch, memory_exporter):
        """Mock the AntsPlatformSpanProcessor to avoid HTTP traffic."""

        def mock_init(self, **kwargs):
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self.public_key = kwargs.get("public_key", "test-key")
            blocked_scopes = kwargs.get("blocked_instrumentation_scopes")
            self.blocked_instrumentation_scopes = (
                blocked_scopes if blocked_scopes is not None else []
            )
            BatchSpanProcessor.__init__(
                self,
                span_exporter=memory_exporter,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )

        monkeypatch.setattr(
            "ants_platform._client.span_processor.AntsPlatformSpanProcessor.__init__",
            mock_init,
        )

    @pytest.fixture
    def ants_platform_client(
        self, monkeypatch, tracer_provider, mock_processor_init
    ):
        """Create a mocked AntsPlatform client for testing."""
        monkeypatch.setenv("ANTS_PLATFORM_PUBLIC_KEY", "test-public-key")
        monkeypatch.setenv("ANTS_PLATFORM_SECRET_KEY", "test-secret-key")

        client = AntsPlatform(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="http://test-host",
            tracing_enabled=True,
        )
        client._otel_tracer = tracer_provider.get_tracer("ants_platform-crewai-test")
        # Pre-set project_id to prevent API call to resolve agent context
        client._project_id = "test-project-id"
        yield client

    @pytest.fixture
    def mock_bus(self):
        """Create a mock event bus."""
        return MockEventBus()

    @pytest.fixture
    def event_listener(self, ants_platform_client, mock_bus):
        """Build an EventListener wired to mock bus and test client."""
        return _build_event_listener(ants_platform_client, mock_bus)

    # ── Helper Methods ────────────────────────────────────────────────

    def get_span_data(self, span: ReadableSpan) -> dict:
        """Extract important data from a span for testing."""
        return {
            "name": span.name,
            "attributes": dict(span.attributes) if span.attributes else {},
            "span_id": format(span.context.span_id, "016x"),
            "trace_id": format(span.context.trace_id, "032x"),
            "parent_span_id": (
                format(span.parent.span_id, "016x") if span.parent else None
            ),
        }

    def get_all_spans(self, memory_exporter) -> List[dict]:
        """Get all finished spans as dicts."""
        return [
            self.get_span_data(s) for s in memory_exporter.get_finished_spans()
        ]

    def get_spans_by_name(self, memory_exporter, name: str) -> List[dict]:
        """Get all spans matching a name."""
        return [
            s for s in self.get_all_spans(memory_exporter) if s["name"] == name
        ]

    def find_span(self, memory_exporter, name: str) -> Optional[dict]:
        """Find a single span by name, or None."""
        spans = self.get_spans_by_name(memory_exporter, name)
        return spans[0] if spans else None

    def assert_parent_child(self, parent: dict, child: dict):
        """Verify parent-child relationship between two spans."""
        assert child["parent_span_id"] == parent["span_id"], (
            f"Expected child '{child['name']}' parent_span_id "
            f"({child['parent_span_id']}) to match parent '{parent['name']}' "
            f"span_id ({parent['span_id']})"
        )
        assert child["trace_id"] == parent["trace_id"]

    def get_attr(self, span_data: dict, attr_key: str) -> Any:
        """Get an attribute value from span data."""
        return span_data["attributes"].get(attr_key)

    def get_json_attr(self, span_data: dict, attr_key: str) -> Any:
        """Get a JSON-serialized attribute value, parsed."""
        raw = span_data["attributes"].get(attr_key)
        if raw is None:
            return None
        return json.loads(raw)


# ── Crew Event Tests ──────────────────────────────────────────────────


class TestCrewEvents(TestCrewAIBase):
    """Tests for crew kickoff start/complete/fail events."""

    def test_crew_started_and_completed(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Crew kickoff creates a root span; completion ends it with output."""
        source = MagicMock(id="crew-456")
        mock_bus.emit(
            CrewKickoffStartedEvent,
            source,
            CrewKickoffStartedEvent(
                crew_name="blog_crew", inputs={"topic": "testing"}
            ),
        )
        mock_bus.emit(
            CrewKickoffCompletedEvent,
            source,
            CrewKickoffCompletedEvent(output="Done", total_tokens=500),
        )

        spans = self.get_all_spans(memory_exporter)
        assert len(spans) == 1

        crew_span = spans[0]
        assert crew_span["name"] == "blog_crew"
        assert (
            self.get_attr(
                crew_span, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE
            )
            == "span"
        )
        output = self.get_attr(
            crew_span, AntsPlatformOtelSpanAttributes.OBSERVATION_OUTPUT
        )
        assert "Done" in output

    def test_crew_failed_sets_error_level(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Crew failure ends the span with level=ERROR and status_message."""
        source = MagicMock(id="crew-fail")
        mock_bus.emit(
            CrewKickoffStartedEvent,
            source,
            CrewKickoffStartedEvent(crew_name="failing_crew"),
        )
        mock_bus.emit(
            CrewKickoffFailedEvent,
            source,
            CrewKickoffFailedEvent(error="Out of memory"),
        )

        spans = self.get_all_spans(memory_exporter)
        assert len(spans) == 1
        span = spans[0]
        assert (
            self.get_attr(span, AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL)
            == "ERROR"
        )
        assert (
            self.get_attr(
                span, AntsPlatformOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE
            )
            == "Out of memory"
        )


# ── Agent Event Tests ─────────────────────────────────────────────────


class TestAgentEvents(TestCrewAIBase):
    """Tests for agent execution start/complete/error events."""

    def _start_crew(self, mock_bus):
        """Helper to create a crew span as parent."""
        source = MagicMock(id="crew-parent")
        mock_bus.emit(
            CrewKickoffStartedEvent,
            source,
            CrewKickoffStartedEvent(crew_name="parent_crew"),
        )
        return source

    def test_agent_nested_under_crew(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Agent execution creates an agent observation nested under crew."""
        crew_source = self._start_crew(mock_bus)

        agent_obj = MagicMock(role="Research Specialist", goal="Find papers")
        mock_bus.emit(
            AgentExecutionStartedEvent,
            MagicMock(),
            AgentExecutionStartedEvent(
                agent=agent_obj,
                agent_id="agent-abc",
                tools=[],
                task_prompt="Research AI trends",
            ),
        )
        mock_bus.emit(
            AgentExecutionCompletedEvent,
            MagicMock(),
            AgentExecutionCompletedEvent(
                agent_id="agent-abc", output="Found 5 papers"
            ),
        )
        mock_bus.emit(
            CrewKickoffCompletedEvent,
            crew_source,
            CrewKickoffCompletedEvent(output="Done"),
        )

        crew_span = self.find_span(memory_exporter, "parent_crew")
        agent_span = self.find_span(memory_exporter, "Research Specialist")

        assert crew_span is not None
        assert agent_span is not None
        assert (
            self.get_attr(
                agent_span, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE
            )
            == "span"
        )
        self.assert_parent_child(crew_span, agent_span)

    def test_agent_error_sets_error_level(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Agent error ends the span with level=ERROR."""
        self._start_crew(mock_bus)

        mock_bus.emit(
            AgentExecutionStartedEvent,
            MagicMock(),
            AgentExecutionStartedEvent(
                agent=MagicMock(role="Broken Agent", goal="Fail"),
                agent_id="agent-err",
                tools=[],
            ),
        )
        mock_bus.emit(
            AgentExecutionErrorEvent,
            MagicMock(),
            AgentExecutionErrorEvent(
                agent_id="agent-err", error="Tool crashed"
            ),
        )

        agent_span = self.find_span(memory_exporter, "Broken Agent")
        assert agent_span is not None
        assert (
            self.get_attr(
                agent_span, AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL
            )
            == "ERROR"
        )


# ── Task Event Tests ──────────────────────────────────────────────────


class TestTaskEvents(TestCrewAIBase):
    """Tests for task start/complete/fail events."""

    def _start_crew_and_agent(self, mock_bus):
        """Helper: crew + agent context."""
        crew_source = MagicMock(id="crew-t")
        mock_bus.emit(
            CrewKickoffStartedEvent,
            crew_source,
            CrewKickoffStartedEvent(crew_name="task_crew"),
        )
        mock_bus.emit(
            AgentExecutionStartedEvent,
            MagicMock(),
            AgentExecutionStartedEvent(
                agent=MagicMock(role="Worker", goal="Do tasks"),
                agent_id="agent-t",
                tools=[],
            ),
        )
        return crew_source

    def test_task_creates_chain_span(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Task creates a chain observation."""
        self._start_crew_and_agent(mock_bus)

        task_obj = MagicMock(
            description="Write a blog post about AI",
            expected_output="A 500-word blog",
        )
        task_source = MagicMock(id="task-1")
        mock_bus.emit(
            TaskStartedEvent,
            task_source,
            TaskStartedEvent(task=task_obj, task_id="task-1", context=None),
        )

        output_obj = MagicMock(raw="Blog post content here")
        mock_bus.emit(
            TaskCompletedEvent,
            task_source,
            TaskCompletedEvent(task_id="task-1", output=output_obj),
        )

        task_span = self.find_span(memory_exporter, "Write a blog post about AI")
        assert task_span is not None
        assert (
            self.get_attr(
                task_span, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE
            )
            == "chain"
        )

    def test_task_long_description_truncated(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Task descriptions longer than 100 chars are truncated in span name."""
        self._start_crew_and_agent(mock_bus)

        long_desc = "A" * 150
        task_source = MagicMock(id="task-long")
        mock_bus.emit(
            TaskStartedEvent,
            task_source,
            TaskStartedEvent(
                task=MagicMock(description=long_desc, expected_output="result"),
                task_id="task-long",
            ),
        )
        mock_bus.emit(
            TaskCompletedEvent,
            task_source,
            TaskCompletedEvent(task_id="task-long", output="done"),
        )

        spans = self.get_all_spans(memory_exporter)
        task_spans = [s for s in spans if s["name"].startswith("AAA")]
        assert len(task_spans) == 1
        assert task_spans[0]["name"] == "A" * 80 + "..."

    def test_task_failed_sets_error(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Task failure ends the span with level=ERROR."""
        self._start_crew_and_agent(mock_bus)

        task_source = MagicMock(id="task-fail")
        mock_bus.emit(
            TaskStartedEvent,
            task_source,
            TaskStartedEvent(
                task=MagicMock(description="Failing task", expected_output="x"),
                task_id="task-fail",
            ),
        )
        mock_bus.emit(
            TaskFailedEvent,
            task_source,
            TaskFailedEvent(task_id="task-fail", error="Timeout"),
        )

        span = self.find_span(memory_exporter, "Failing task")
        assert span is not None
        assert (
            self.get_attr(span, AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL)
            == "ERROR"
        )


# ── LLM Event Tests ──────────────────────────────────────────────────


class TestLLMEvents(TestCrewAIBase):
    """Tests for LLM call start/complete/fail events."""

    def _start_context(self, mock_bus):
        """Create crew + agent context."""
        crew_source = MagicMock(id="crew-llm")
        mock_bus.emit(
            CrewKickoffStartedEvent,
            crew_source,
            CrewKickoffStartedEvent(crew_name="llm_crew"),
        )
        mock_bus.emit(
            AgentExecutionStartedEvent,
            MagicMock(),
            AgentExecutionStartedEvent(
                agent=MagicMock(role="LLM Agent", goal="Call LLM"),
                agent_id="agent-llm",
                tools=[],
            ),
        )
        return crew_source

    def test_llm_call_creates_generation_span(
        self, event_listener, mock_bus, memory_exporter
    ):
        """LLM call creates a generation observation with model name."""
        self._start_context(mock_bus)

        mock_bus.emit(
            LLMCallStartedEvent,
            MagicMock(),
            LLMCallStartedEvent(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                tools=None,
                agent_id="agent-llm",
            ),
        )

        usage = MagicMock(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        message = MagicMock(content="Hello! How can I help?")
        choice = MagicMock(message=message)
        response = MagicMock(usage=usage, choices=[choice])

        mock_bus.emit(
            LLMCallCompletedEvent,
            MagicMock(),
            LLMCallCompletedEvent(response=response, model="gpt-4o-mini"),
        )

        span = self.find_span(memory_exporter, "LLM Call (gpt-4o-mini)")
        assert span is not None
        assert (
            self.get_attr(
                span, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE
            )
            == "generation"
        )
        assert (
            self.get_attr(span, AntsPlatformOtelSpanAttributes.OBSERVATION_MODEL)
            == "gpt-4o-mini"
        )

    def test_llm_call_failed_sets_error(
        self, event_listener, mock_bus, memory_exporter
    ):
        """LLM call failure ends the span with level=ERROR."""
        self._start_context(mock_bus)

        mock_bus.emit(
            LLMCallStartedEvent,
            MagicMock(),
            LLMCallStartedEvent(
                model="gpt-4o", messages=[], tools=None, agent_id="agent-llm"
            ),
        )
        mock_bus.emit(
            LLMCallFailedEvent,
            MagicMock(),
            LLMCallFailedEvent(error="Rate limit exceeded"),
        )

        span = self.find_span(memory_exporter, "LLM Call (gpt-4o)")
        assert span is not None
        assert (
            self.get_attr(span, AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL)
            == "ERROR"
        )

    def test_llm_sequential_matching(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Two sequential LLM calls are matched correctly via thread-local stack."""
        self._start_context(mock_bus)

        for i in range(2):
            model = f"model-{i}"
            mock_bus.emit(
                LLMCallStartedEvent,
                MagicMock(),
                LLMCallStartedEvent(
                    model=model,
                    messages=[],
                    tools=None,
                    agent_id="agent-llm",
                ),
            )
            usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            response = MagicMock(
                usage=usage,
                choices=[MagicMock(message=MagicMock(content=f"r{i}"))],
            )
            mock_bus.emit(
                LLMCallCompletedEvent,
                MagicMock(),
                LLMCallCompletedEvent(response=response, model=model),
            )

        span_0 = self.find_span(memory_exporter, "LLM Call (model-0)")
        span_1 = self.find_span(memory_exporter, "LLM Call (model-1)")

        assert span_0 is not None, "First LLM span should exist"
        assert span_1 is not None, "Second LLM span should exist"


# ── Tool Event Tests ──────────────────────────────────────────────────


class TestToolEvents(TestCrewAIBase):
    """Tests for tool usage start/finish/error events."""

    def _start_context(self, mock_bus):
        crew_source = MagicMock(id="crew-tool")
        mock_bus.emit(
            CrewKickoffStartedEvent,
            crew_source,
            CrewKickoffStartedEvent(crew_name="tool_crew"),
        )
        mock_bus.emit(
            AgentExecutionStartedEvent,
            MagicMock(),
            AgentExecutionStartedEvent(
                agent=MagicMock(role="Tool Agent", goal="Use tools"),
                agent_id="agent-tool",
                tools=[],
            ),
        )
        return crew_source

    def test_tool_creates_tool_span(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Tool usage creates a tool observation."""
        self._start_context(mock_bus)

        mock_bus.emit(
            ToolUsageStartedEvent,
            MagicMock(),
            ToolUsageStartedEvent(
                tool_name="SerperDevTool",
                tool_args={"query": "AI trends"},
                agent_id="agent-tool",
                tool_class="SerperDevTool",
                run_attempts=1,
            ),
        )
        mock_bus.emit(
            ToolUsageFinishedEvent,
            MagicMock(),
            ToolUsageFinishedEvent(
                tool_name="SerperDevTool",
                agent_id="agent-tool",
                output="Search results...",
                from_cache=False,
                started_at="2026-04-05T10:00:00",
                finished_at="2026-04-05T10:00:01",
            ),
        )

        span = self.find_span(memory_exporter, "SerperDevTool")
        assert span is not None
        assert (
            self.get_attr(span, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE)
            == "tool"
        )

    def test_tool_error_sets_error_level(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Tool error ends the span with level=ERROR."""
        self._start_context(mock_bus)

        mock_bus.emit(
            ToolUsageStartedEvent,
            MagicMock(),
            ToolUsageStartedEvent(
                tool_name="BrokenTool",
                tool_args={},
                agent_id="agent-tool",
            ),
        )
        mock_bus.emit(
            ToolUsageErrorEvent,
            MagicMock(),
            ToolUsageErrorEvent(
                tool_name="BrokenTool",
                agent_id="agent-tool",
                error="Connection refused",
            ),
        )

        span = self.find_span(memory_exporter, "BrokenTool")
        assert span is not None
        assert (
            self.get_attr(span, AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL)
            == "ERROR"
        )

    def test_tool_no_leak_after_cleanup(
        self, event_listener, mock_bus, memory_exporter
    ):
        """After tool finish, internal registries should be clean."""
        self._start_context(mock_bus)

        mock_bus.emit(
            ToolUsageStartedEvent,
            MagicMock(),
            ToolUsageStartedEvent(
                tool_name="CleanTool",
                tool_args={},
                agent_id="agent-tool",
            ),
        )
        mock_bus.emit(
            ToolUsageFinishedEvent,
            MagicMock(),
            ToolUsageFinishedEvent(
                tool_name="CleanTool",
                agent_id="agent-tool",
                output="done",
            ),
        )

        # Check latest alias is cleaned up
        alias_key = "latest:agent-tool:CleanTool"
        assert alias_key not in event_listener._tool_spans

        # Check no tool-related context tokens remain
        tool_tokens = [
            k
            for k in event_listener._context_tokens
            if k.startswith("tool:agent-tool:CleanTool:")
        ]
        assert len(tool_tokens) == 0, (
            f"Expected no leftover context tokens, found: {tool_tokens}"
        )


# ── Full Trace Hierarchy Test ─────────────────────────────────────────


class TestFullTraceHierarchy(TestCrewAIBase):
    """Integration test: full Crew -> Agent -> Task -> LLM hierarchy."""

    def test_full_crew_execution_hierarchy(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Simulate a full crew run and verify nested trace hierarchy."""
        crew_source = MagicMock(id="crew-full")

        # 1. Crew starts
        mock_bus.emit(
            CrewKickoffStartedEvent,
            crew_source,
            CrewKickoffStartedEvent(crew_name="full_crew", inputs={"topic": "AI"}),
        )

        # 2. Agent starts
        mock_bus.emit(
            AgentExecutionStartedEvent,
            MagicMock(),
            AgentExecutionStartedEvent(
                agent=MagicMock(role="Researcher", goal="Research"),
                agent_id="agent-full",
                tools=["SearchTool"],
            ),
        )

        # 3. Task starts
        task_source = MagicMock(id="task-full")
        mock_bus.emit(
            TaskStartedEvent,
            task_source,
            TaskStartedEvent(
                task=MagicMock(
                    description="Research AI impact",
                    expected_output="Report",
                ),
                task_id="task-full",
                agent_id="agent-full",
            ),
        )

        # 4. LLM call
        mock_bus.emit(
            LLMCallStartedEvent,
            MagicMock(),
            LLMCallStartedEvent(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Tell me about AI"}],
                tools=None,
                agent_id="agent-full",
            ),
        )

        usage = MagicMock(
            prompt_tokens=200, completion_tokens=100, total_tokens=300
        )
        response = MagicMock(
            usage=usage,
            choices=[MagicMock(message=MagicMock(content="AI is transforming..."))],
        )
        mock_bus.emit(
            LLMCallCompletedEvent,
            MagicMock(),
            LLMCallCompletedEvent(response=response, model="gpt-4o-mini"),
        )

        # 5. End: task, agent, crew
        mock_bus.emit(
            TaskCompletedEvent,
            task_source,
            TaskCompletedEvent(
                task_id="task-full",
                output=MagicMock(raw="AI impact report"),
            ),
        )
        mock_bus.emit(
            AgentExecutionCompletedEvent,
            MagicMock(),
            AgentExecutionCompletedEvent(
                agent_id="agent-full", output="Research complete"
            ),
        )
        mock_bus.emit(
            CrewKickoffCompletedEvent,
            crew_source,
            CrewKickoffCompletedEvent(output="Full report", total_tokens=300),
        )

        # Verify all spans exist
        crew = self.find_span(memory_exporter, "full_crew")
        agent = self.find_span(memory_exporter, "Researcher")
        task = self.find_span(memory_exporter, "Research AI impact")
        llm = self.find_span(memory_exporter, "LLM Call (gpt-4o-mini)")

        assert crew is not None, "Crew span should exist"
        assert agent is not None, "Agent span should exist"
        assert task is not None, "Task span should exist"
        assert llm is not None, "LLM span should exist"

        # Verify parent-child: Agent -> Crew
        self.assert_parent_child(crew, agent)

        # Verify observation types
        assert self.get_attr(crew, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE) == "span"
        assert self.get_attr(agent, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE) == "span"
        assert self.get_attr(task, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE) == "chain"
        assert self.get_attr(llm, AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE) == "generation"

        # All spans share the same trace ID
        trace_ids = {crew["trace_id"], agent["trace_id"], task["trace_id"], llm["trace_id"]}
        assert len(trace_ids) == 1, f"All spans should share one trace ID, got {trace_ids}"


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases(TestCrewAIBase):
    """Tests for error handling and edge cases."""

    def test_completion_without_start_does_not_crash(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Completing a crew/agent/task that was never started should not raise."""
        mock_bus.emit(
            CrewKickoffCompletedEvent,
            MagicMock(id="ghost"),
            CrewKickoffCompletedEvent(output="phantom"),
        )
        mock_bus.emit(
            AgentExecutionCompletedEvent,
            MagicMock(),
            AgentExecutionCompletedEvent(agent_id="ghost-agent", output="x"),
        )
        mock_bus.emit(
            TaskCompletedEvent,
            MagicMock(id="ghost-task"),
            TaskCompletedEvent(task_id="ghost-task", output="x"),
        )
        mock_bus.emit(
            LLMCallCompletedEvent,
            MagicMock(),
            LLMCallCompletedEvent(response=None, model="x"),
        )
        mock_bus.emit(
            ToolUsageFinishedEvent,
            MagicMock(),
            ToolUsageFinishedEvent(
                tool_name="ghost", agent_id="ghost", output="x"
            ),
        )

        # No spans should be created
        spans = self.get_all_spans(memory_exporter)
        assert len(spans) == 0

    def test_multiple_crews_sequential(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Running two crews sequentially produces two separate root spans."""
        for i in range(2):
            source = MagicMock(id=f"crew-seq-{i}")
            mock_bus.emit(
                CrewKickoffStartedEvent,
                source,
                CrewKickoffStartedEvent(crew_name=f"crew_{i}"),
            )
            mock_bus.emit(
                CrewKickoffCompletedEvent,
                source,
                CrewKickoffCompletedEvent(output=f"result_{i}"),
            )

        crew_spans = [
            s
            for s in self.get_all_spans(memory_exporter)
            if s["name"].startswith("crew_")
        ]
        assert len(crew_spans) == 2
        assert crew_spans[0]["name"] == "crew_0"
        assert crew_spans[1]["name"] == "crew_1"

    def test_handler_exception_does_not_propagate(
        self, event_listener, mock_bus, memory_exporter
    ):
        """Internal SDK error in handler is swallowed, not raised."""
        original = event_listener.client.start_observation

        def broken_start(*args, **kwargs):
            raise RuntimeError("Simulated SDK failure")

        event_listener.client.start_observation = broken_start

        # Should not raise
        mock_bus.emit(
            CrewKickoffStartedEvent,
            MagicMock(id="boom"),
            CrewKickoffStartedEvent(crew_name="boom_crew"),
        )

        event_listener.client.start_observation = original

        spans = self.get_all_spans(memory_exporter)
        assert len(spans) == 0
