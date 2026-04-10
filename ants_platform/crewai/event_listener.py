"""CrewAI EventListener for Ants Platform.

Hooks into CrewAI's event bus to capture crew execution, agent steps,
LLM calls, tool usage, guardrails, and reasoning as Ants Platform observations.

Usage:
    from ants_platform import AntsPlatform
    from ants_platform.crewai import EventListener

    ants_platform = AntsPlatform()
    listener = EventListener()   # auto-registers with CrewAI event bus

    crew = Crew(agents=[...], tasks=[...])
    result = crew.kickoff()
    ants_platform.flush()
"""

import threading
import re
from contextvars import Token
from typing import Any, Dict, Optional, Union, cast

from opentelemetry import baggage as otel_baggage
from opentelemetry import context, trace
from opentelemetry.context import _RUNTIME_CONTEXT

from ants_platform._client.get_client import get_client
from ants_platform._client.span import (
    AntsPlatformAgent,
    AntsPlatformGeneration,
    AntsPlatformGuardrail,
    AntsPlatformSpan,
    AntsPlatformTool,
)
from ants_platform.logger import ants_platform_logger

try:
    from crewai.events.base_event_listener import BaseEventListener
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.agent_events import (
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        AgentExecutionStartedEvent,
    )
    from crewai.events.types.crew_events import (
        CrewKickoffCompletedEvent,
        CrewKickoffFailedEvent,
        CrewKickoffStartedEvent,
    )
    from crewai.events.types.llm_events import (
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        LLMCallStartedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )
except ImportError:
    raise ModuleNotFoundError(
        "Please install crewai to use the Ants Platform CrewAI integration: "
        "'pip install ants-platform[crewai]' or 'pip install crewai>=0.80.0'"
    )

# Phase 2 imports — optional, won't break if not available in older CrewAI versions
try:
    from crewai.events.types.llm_guardrail_events import (
        LLMGuardrailCompletedEvent,
        LLMGuardrailFailedEvent,
        LLMGuardrailStartedEvent,
    )

    _HAS_GUARDRAIL_EVENTS = True
except ImportError:
    _HAS_GUARDRAIL_EVENTS = False

try:
    from crewai.events.types.reasoning_events import (
        AgentReasoningCompletedEvent,
        AgentReasoningFailedEvent,
        AgentReasoningStartedEvent,
    )

    _HAS_REASONING_EVENTS = True
except ImportError:
    _HAS_REASONING_EVENTS = False


# Type alias for any Ants Platform observation
_Observation = Union[
    AntsPlatformSpan,
    AntsPlatformAgent,
    AntsPlatformGeneration,
    AntsPlatformTool,
    AntsPlatformGuardrail,
]


class EventListener(BaseEventListener):
    """Ants Platform event listener for CrewAI.

    Automatically registers with CrewAI's event bus on instantiation.
    Captures crew, agent, task, LLM, and tool events as Ants Platform
    observations with full parent-child hierarchy.

    Args:
        public_key: Optional Ants Platform public key. If not provided,
            uses the default client configuration / environment variables.
    """

    # CrewAI's built-in telemetry scope — must be blocked from our exporter
    # to prevent duplicate/orphan traces in the user's Ants Platform instance.
    _CREWAI_TELEMETRY_SCOPE = "crewai.telemetry"

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_display_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.client = get_client(public_key=public_key)

        # Explicit agent_name — overrides auto-detected crew class name.
        # Without this, all CrewAI agents default to "crew".
        self._explicit_agent_name: Optional[str] = agent_name
        self._explicit_agent_display_name: Optional[str] = agent_display_name

        # User-provided tags (key:value pairs) merged with auto-detected defaults.
        # Stored as dict, converted to List[str] ("key:value") when applied to traces.
        self._user_tags: Dict[str, str] = tags or {}

        # Block CrewAI's own telemetry spans from being exported via our SDK.
        # CrewAI sends its own OTel spans (scope: "crewai.telemetry") to
        # telemetry.crewai.com. Without blocking, these bleed into the user's
        # Ants Platform as orphan traces with no parent-child relationships.
        self._block_crewai_telemetry_scope()

        self._lock = threading.Lock()

        # Observation registries — keyed by CrewAI entity IDs (all string keys)
        self._crew_spans: Dict[str, AntsPlatformSpan] = {}
        self._agent_spans: Dict[str, AntsPlatformAgent] = {}
        self._llm_spans: Dict[str, AntsPlatformGeneration] = {}
        self._tool_spans: Dict[str, AntsPlatformTool] = {}
        self._guardrail_spans: Dict[str, AntsPlatformGuardrail] = {}
        self._reasoning_spans: Dict[str, AntsPlatformSpan] = {}

        # OTel context tokens for proper cleanup
        self._context_tokens: Dict[str, Token] = {}

        # Current active agent span (global — CrewAI fires events across threads)
        self._current_agent: Optional[_Observation] = None

        # The crew's agent_name — stored once, passed explicitly to ALL
        # child spans. OTel baggage doesn't propagate across threads,
        # so we must pass agent_name explicitly on every start_observation.
        self._crew_agent_name: Optional[str] = None

        # Global LLM key stack — CrewAI fires start/complete events on
        # different threads, so thread-local doesn't work.
        self._llm_key_stack: list = []

        # Thread-local storage for guardrail keys
        self._thread_local = threading.local()

        # Pre-warm project_id to avoid blocking HTTP call on first span creation.
        # _get_project_id() fetches once and caches, so doing it here prevents
        # timeouts inside event handlers running on CrewAI's ThreadPoolExecutor.
        try:
            self.client._get_project_id()
        except Exception:
            pass  # Non-fatal — SDK will retry lazily if needed

        # Suppress scary OTel export timeout tracebacks — replace with a
        # friendly one-line warning so users don't panic.
        self._install_export_error_filter()

        # Calls setup_listeners() + validate_dependencies()
        super().__init__()

        ants_platform_logger.info(
            "Ants Platform CrewAI EventListener registered successfully."
        )

    @staticmethod
    def _install_export_error_filter() -> None:
        """Replace OTel's scary export-timeout traceback with a friendly warning.

        The OpenTelemetry BatchSpanProcessor logs at ERROR level with a full
        traceback when a batch export times out. This is normal for large
        crews (many LLM spans) — spans are queued and retried automatically.
        But the traceback panics users who think something is broken.

        This filter intercepts those specific log records, emits a clean
        one-liner via our logger, and suppresses the original.
        """
        import logging

        class _ExportErrorFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                if "Exception while exporting Span" in msg:
                    ants_platform_logger.warning(
                        "Ants Platform: Span export timed out — this is "
                        "normal for large crews. Spans are queued and will "
                        "be retried automatically. No data is lost."
                    )
                    return False  # Suppress the original ERROR + traceback
                return True

        otel_logger = logging.getLogger("opentelemetry.sdk._shared_internal")
        # Avoid adding duplicate filters across multiple EventListener instances
        if not any(isinstance(f, _ExportErrorFilter) for f in otel_logger.filters):
            otel_logger.addFilter(_ExportErrorFilter())

    def _block_crewai_telemetry_scope(self) -> None:
        """Add crewai.telemetry to the blocked scopes on all active span processors."""
        try:
            from ants_platform._client.span_processor import (
                AntsPlatformSpanProcessor,
            )

            provider = trace.get_tracer_provider()
            # The provider may be a ProxyTracerProvider wrapping the real one
            real_provider = getattr(provider, "_real_provider", provider)
            processors = getattr(real_provider, "_active_span_processor", None)

            # Walk the composite processor to find our AntsPlatformSpanProcessor(s)
            span_processors = getattr(processors, "_span_processors", [])
            for proc in span_processors:
                if isinstance(proc, AntsPlatformSpanProcessor):
                    if self._CREWAI_TELEMETRY_SCOPE not in proc.blocked_instrumentation_scopes:
                        proc.blocked_instrumentation_scopes.append(
                            self._CREWAI_TELEMETRY_SCOPE
                        )
        except Exception:
            pass  # Best-effort — won't break if internals change

    def _resolve_crew_agent_name(self, source: Any, event: Any) -> str:
        """Resolve the Ants Platform agent_name for this crew.

        Priority order:
        1. Explicit agent_name passed to EventListener constructor
        2. crew_name from the CrewKickoffStartedEvent
        3. Crew class name from source (e.g. "MarketResearchCrew" → "market_research_crew")
        4. Crew.name attribute if set
        5. Fallback: "crew"
        """
        # 1. Explicit from constructor
        if self._explicit_agent_name:
            return self._explicit_agent_name

        # 2. From event
        event_name = getattr(event, "crew_name", None)
        if event_name and isinstance(event_name, str) and event_name.strip():
            return event_name.strip()

        # 3. From source class name (e.g. MarketResearchCrew → market_research_crew)
        class_name = type(source).__name__
        if class_name and class_name != "Crew":
            # Convert CamelCase to snake_case
            import re

            snake = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", class_name).lower()
            # Remove trailing _crew if present to keep it clean, then add back
            # Actually keep as-is for clarity
            return snake

        # 4. From source.name attribute
        source_name = getattr(source, "name", None)
        if source_name and isinstance(source_name, str) and source_name.strip():
            return source_name.strip()

        # 5. Fallback
        return "crew"

    def _build_trace_tags(self, crew_name: str, source: Any) -> list:
        """Build trace tags list from auto-detected CrewAI info + user tags.

        Returns a List[str] in "key:value" format, matching the Ants Platform
        convention used by other agents (e.g. smart-office-assistant).

        Auto-detected tags:
            - framework:crewai
            - agent:<crew_name>
            - process_type:<sequential|hierarchical|...>
            - crewai_version:<version>
            - num_agents:<count>

        User-provided tags (from constructor) override auto-detected ones
        when keys collide.
        """
        auto_tags: Dict[str, str] = {"framework": "crewai", "agent": crew_name}

        # Detect process type from the crew source object
        process = getattr(source, "process", None)
        if process is not None:
            # CrewAI Process enum → "sequential", "hierarchical", etc.
            process_name = getattr(process, "value", str(process))
            auto_tags["process_type"] = str(process_name).lower()

        # Detect CrewAI version
        try:
            import crewai

            crewai_version = getattr(crewai, "__version__", None)
            if crewai_version:
                auto_tags["crewai_version"] = str(crewai_version)
        except Exception:
            pass

        # Count agents in the crew
        agents = getattr(source, "agents", None)
        if agents and hasattr(agents, "__len__"):
            auto_tags["num_agents"] = str(len(agents))

        # Count tasks in the crew
        tasks = getattr(source, "tasks", None)
        if tasks and hasattr(tasks, "__len__"):
            auto_tags["num_tasks"] = str(len(tasks))

        # User tags override auto-detected tags on key collision
        merged = {**auto_tags, **self._user_tags}

        return [f"{k}:{v}" for k, v in merged.items()]

    def _clear_agent_baggage(self) -> None:
        """Remove agent_name/agent_id/agent_display_name from OTel baggage.

        CrewAI uses a ThreadPoolExecutor — threads are reused across agents.
        Without clearing, a previous agent's baggage persists and the SDK's
        '_resolve_agent_context' refuses to let the next agent set its own name.
        """
        try:
            ctx = context.get_current()
            ctx = otel_baggage.remove_baggage("agent_name", ctx)
            ctx = otel_baggage.remove_baggage("agent_id", ctx)
            ctx = otel_baggage.remove_baggage("agent_display_name", ctx)
            context.attach(ctx)
        except Exception:
            pass

    # ── Context Management ──────────────────────────────────────────────

    def _attach(self, key: str, observation: _Observation, registry: Dict) -> None:
        """Register observation and set it as current OTel context."""
        with self._lock:
            ctx = trace.set_span_in_context(observation._otel_span)
            token = context.attach(ctx)
            registry[key] = observation
            self._context_tokens[key] = token

    def _detach(self, key: str, registry: Dict) -> Optional[_Observation]:
        """Remove observation from registry and restore OTel context."""
        with self._lock:
            token = self._context_tokens.pop(key, None)
            if token:
                try:
                    _RUNTIME_CONTEXT.detach(token)
                except Exception:
                    # Expected in cross-thread scenarios — safe to ignore
                    pass
            return cast(Optional[_Observation], registry.pop(key, None))

    def _wait_for_crew_span(self, timeout: float = 60.0) -> Any:
        """Wait for the crew span AND crew_agent_name to be set.

        CrewAI fires events concurrently on a ThreadPoolExecutor. Task and
        agent events may arrive before the crew event handler finishes creating
        the root span. This method polls briefly to avoid orphan traces and
        ensures _crew_agent_name is available for child spans.
        """
        import time

        deadline = time.monotonic() + timeout
        poll_interval = 0.05  # 50ms
        while time.monotonic() < deadline:
            with self._lock:
                if self._crew_spans and self._crew_agent_name:
                    return next(reversed(self._crew_spans.values()))
            time.sleep(poll_interval)

        # Timeout — fall back to client (creates a new root trace)
        return self.client

    def _get_crew_agent_name(self) -> str:
        """Get the crew's agent_name, waiting if needed.

        Called by child event handlers that need to pass agent_name
        explicitly. Waits up to 15s for the crew span to be created.
        """
        import time

        if self._crew_agent_name:
            return self._crew_agent_name

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if self._crew_agent_name:
                return self._crew_agent_name
            time.sleep(0.05)

        return "crew"  # fallback

    def _wait_for_agent_span(self, agent_id: str, timeout: float = 15.0) -> Any:
        """Wait for an agent span to be registered.

        CrewAI fires LLM/tool events concurrently with agent events.
        The LLM event may arrive before the agent span is created.
        This method polls briefly so child spans attach to the correct agent.
        """
        import time

        deadline = time.monotonic() + timeout
        poll_interval = 0.05  # 50ms
        while time.monotonic() < deadline:
            with self._lock:
                if agent_id in self._agent_spans:
                    return self._agent_spans[agent_id]
                # Also check if _current_agent was set in the meantime
                if self._current_agent is not None:
                    return self._current_agent
            time.sleep(poll_interval)
        return None

    def _get_parent_for_event(self, event: Any) -> Any:
        """Find the correct parent observation for a child event.

        Priority:
        1. Current agent on this thread (set by _set_current_agent)
        2. Agent by event.agent_id in registry (wait if not yet registered)
        3. Most recent crew span
        4. Wait for crew span (handles event ordering race)
        """
        with self._lock:
            # Current active agent is the most reliable parent
            if self._current_agent is not None:
                return self._current_agent

            agent_id = getattr(event, "agent_id", None)
            if agent_id and agent_id in self._agent_spans:
                return self._agent_spans[agent_id]

        # Agent ID present but span not yet registered — wait for it
        agent_id = getattr(event, "agent_id", None)
        if agent_id:
            agent_span = self._wait_for_agent_span(agent_id)
            if agent_span:
                return agent_span

        with self._lock:
            if self._crew_spans:
                return next(reversed(self._crew_spans.values()))

        return self._wait_for_crew_span()

    def _resolve_agent_name(self, event: Any) -> str:
        """Resolve the agent_name for a child observation.

        Uses the thread-local current agent first (most reliable),
        then falls back to registry lookups.
        """
        with self._lock:
            # Current active agent
            if self._current_agent is not None:
                name = getattr(self._current_agent, "_agent_name", None)
                if name:
                    return name

            agent_id = getattr(event, "agent_id", None)
            if agent_id and agent_id in self._agent_spans:
                agent_span = self._agent_spans[agent_id]
                name = getattr(agent_span, "_agent_name", None)
                if name:
                    return name

            if self._agent_spans:
                last_agent = next(reversed(self._agent_spans.values()))
                name = getattr(last_agent, "_agent_name", None)
                if name:
                    return name

        return "crewai"

    def _set_current_agent(self, span: Any) -> None:
        """Set the current active agent (global, not thread-local).

        CrewAI fires events across different threads from its
        ThreadPoolExecutor, so thread-local won't work. Since CrewAI
        runs agents sequentially (Process.sequential), a global
        current agent protected by the existing lock is correct.
        """
        with self._lock:
            self._current_agent = span

    def _clear_current_agent(self) -> None:
        """Clear the current active agent."""
        with self._lock:
            self._current_agent = None

    def _get_crew_span(self) -> Any:
        """Get the most recent crew span, waiting briefly if needed."""
        with self._lock:
            if self._crew_spans:
                return next(reversed(self._crew_spans.values()))
        # Crew span not yet created — wait for it
        return self._wait_for_crew_span()

    def _detach_tool_span(
        self, agent_id: Optional[str], tool_name: str
    ) -> Optional[AntsPlatformTool]:
        """Atomically find and remove a tool span + its context token.

        Uses the latest-alias key for lookup, then cleans up both the alias
        and the primary key with its context token — all under a single lock.
        """
        with self._lock:
            lookup_key = f"latest:{agent_id}:{tool_name}"
            span = self._tool_spans.pop(lookup_key, None)
            if span is None:
                return None

            # Find and clean up the primary key + context token
            primary_prefix = f"tool:{agent_id}:{tool_name}:"
            for k in list(self._context_tokens.keys()):
                if k.startswith(primary_prefix):
                    token = self._context_tokens.pop(k, None)
                    self._tool_spans.pop(k, None)
                    break
            else:
                token = None

        # Detach OTel context outside the lock (may block)
        if token:
            try:
                _RUNTIME_CONTEXT.detach(token)
            except Exception:
                pass

        return span

    # ── Thread-local LLM key management ────────────────────────────────

    def _push_llm_key(self, key: str) -> None:
        """Push an LLM span key onto the global stack.

        CrewAI fires start/complete events on DIFFERENT threads,
        so thread-local won't work. Using a lock-protected global
        stack since CrewAI runs LLM calls sequentially per agent.
        """
        with self._lock:
            if not hasattr(self, "_llm_key_stack"):
                self._llm_key_stack: list = []
            self._llm_key_stack.append(key)

    def _pop_llm_key(self) -> Optional[str]:
        """Pop the most recent LLM span key from the global stack."""
        with self._lock:
            if hasattr(self, "_llm_key_stack") and self._llm_key_stack:
                return self._llm_key_stack.pop()
        return None

    # ── Event Handler Registration ─────────────────────────────────────

    def setup_listeners(self, crewai_event_bus) -> None:
        """Register all event handlers on CrewAI's event bus."""
        self._register_crew_events(crewai_event_bus)
        self._register_agent_events(crewai_event_bus)
        self._register_llm_events(crewai_event_bus)
        self._register_tool_events(crewai_event_bus)

        if _HAS_GUARDRAIL_EVENTS:
            self._register_guardrail_events(crewai_event_bus)

        if _HAS_REASONING_EVENTS:
            self._register_reasoning_events(crewai_event_bus)

    # ── Crew Events ────────────────────────────────────────────────────

    def _register_crew_events(self, bus: Any) -> None:
        @bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source: Any, event: CrewKickoffStartedEvent) -> None:
            try:
                crew_name = self._resolve_crew_agent_name(source, event)
                crew_id = str(getattr(source, "id", None) or id(source))

                # Build display name — explicit > crew_name
                display_name = self._explicit_agent_display_name or crew_name

                # The crew IS the agent in Ants Platform's model.
                # All child spans (agents, tasks, LLM calls) inherit
                # this agent_name via OTel baggage — no conflicts.
                span = self.client.start_observation(
                    name=crew_name,
                    as_type="span",
                    agent_name=crew_name,
                    agent_display_name=display_name,
                    input=getattr(event, "inputs", None),
                    metadata={
                        "framework": "crewai",
                        "crewai.crew_name": crew_name,
                    },
                )
                self._crew_agent_name = crew_name
                # Attach crew span to OTel context so child spans
                # created via parent.start_observation() are properly
                # registered with the span processor for export.
                self._attach(f"crew:{crew_id}", span, self._crew_spans)

                # Apply trace-level tags (framework:crewai, process_type, version, etc.)
                # These become agent tags in the Postgres agents table via
                # the backend's key:value → JSONB conversion during ingestion.
                trace_tags = self._build_trace_tags(crew_name, source)
                span.update_trace(tags=trace_tags)
                ants_platform_logger.debug(
                    "CrewAI: Crew '%s' started with tags: %s", crew_name, trace_tags
                )
            except Exception as e:
                msg = f"CrewAI event handler error (crew_started): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source: Any, event: CrewKickoffCompletedEvent) -> None:
            try:
                crew_id = str(getattr(source, "id", None) or id(source))
                span = self._detach(f"crew:{crew_id}", self._crew_spans)
                if span:
                    span.update(
                        output=str(getattr(event, "output", None)),
                        metadata={
                            "crewai.total_tokens": getattr(
                                event, "total_tokens", 0
                            ),
                        },
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (crew_completed): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source: Any, event: CrewKickoffFailedEvent) -> None:
            try:
                crew_id = str(getattr(source, "id", None) or id(source))
                span = self._detach(f"crew:{crew_id}", self._crew_spans)
                if span:
                    error_msg = getattr(event, "error", "Crew execution failed")
                    span.update(
                        level="ERROR",
                        status_message=error_msg,
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (crew_failed): {e}"
                ants_platform_logger.warning(msg)

    # ── Agent Events ───────────────────────────────────────────────────

    def _register_agent_events(self, bus: Any) -> None:
        @bus.on(AgentExecutionStartedEvent)
        def on_agent_started(
            source: Any, event: AgentExecutionStartedEvent
        ) -> None:
            try:
                agent = getattr(event, "agent", None)
                agent_role = getattr(agent, "role", None) or "agent"
                agent_crewai_id = getattr(event, "agent_id", None) or str(
                    id(source)
                )
                agent_role_str = str(agent_role).strip()
                # Prefer the original role template (before CrewAI
                # interpolates {topic} etc.) for a cleaner display name.
                original_role = getattr(agent, "_original_role", None)
                if original_role and isinstance(original_role, str):
                    # Strip template variables: "Expert Writer on {topic}" → "Expert Writer"
                    cleaned = re.sub(r"\s*\{[^}]+\}\s*", " ", original_role).strip()
                    # Remove trailing connectors like "on", "about", "for"
                    cleaned = re.sub(r"\s+(?:on|about|for|in)\s*$", "", cleaned).strip()
                    # Fall back to interpolated role if template was entirely variables
                    display_name = cleaned if cleaned else agent_role_str
                else:
                    display_name = agent_role_str

                tools = getattr(event, "tools", [])
                tool_names = [
                    getattr(t, "name", None) or str(t) for t in tools
                ]

                parent = self._get_crew_span()
                span = parent.start_observation(
                    name=display_name,
                    as_type="span",
                    agent_name=self._get_crew_agent_name(),
                    input={
                        "role": agent_role_str,
                        "goal": getattr(agent, "goal", None),
                        "tools": tool_names,
                        "task_prompt": getattr(event, "task_prompt", None),
                    },
                    metadata={
                        "crewai.agent_id": agent_crewai_id,
                        "crewai.agent_role": agent_role_str,
                    },
                )
                self._attach(agent_crewai_id, span, self._agent_spans)
                self._set_current_agent(span)

                ants_platform_logger.debug(
                    "CrewAI: Agent '%s' started", agent_role
                )
            except Exception as e:
                msg = f"CrewAI event handler error (agent_started): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(
            source: Any, event: AgentExecutionCompletedEvent
        ) -> None:
            try:
                agent_crewai_id = getattr(event, "agent_id", None) or str(
                    id(source)
                )
                self._clear_current_agent()
                span = self._detach(agent_crewai_id, self._agent_spans)
                if span:
                    span.update(
                        output=getattr(event, "output", None),
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (agent_completed): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(AgentExecutionErrorEvent)
        def on_agent_error(
            source: Any, event: AgentExecutionErrorEvent
        ) -> None:
            try:
                agent_crewai_id = getattr(event, "agent_id", None) or str(
                    id(source)
                )
                self._clear_current_agent()
                span = self._detach(agent_crewai_id, self._agent_spans)
                if span:
                    error_msg = getattr(
                        event, "error", "Agent execution error"
                    )
                    span.update(
                        level="ERROR",
                        status_message=error_msg,
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (agent_error): {e}"
                ants_platform_logger.warning(msg)

    # ── LLM Events ─────────────────────────────────────────────────────

    def _register_llm_events(self, bus: Any) -> None:
        @bus.on(LLMCallStartedEvent)
        def on_llm_started(source: Any, event: LLMCallStartedEvent) -> None:
            try:
                # Build a thread-unique key using thread ID + object ID
                thread_id = threading.get_ident()
                llm_key = f"llm:{thread_id}:{id(event)}"

                model = getattr(event, "model", None)
                messages = getattr(event, "messages", None)
                tools = getattr(event, "tools", None)

                parent = self._get_parent_for_event(event)
                resolved_name = self._get_crew_agent_name()

                gen = parent.start_observation(
                    name=f"LLM Call ({model or 'unknown'})",
                    as_type="generation",
                    agent_name=resolved_name,
                    model=model,
                    input={"messages": messages} if messages else None,
                    metadata={
                        "crewai.tools_available": len(tools)
                        if tools
                        else 0,
                    },
                )
                self._attach(llm_key, gen, self._llm_spans)

                # Store key on thread-local stack so completion can find it
                self._push_llm_key(llm_key)
            except Exception as e:
                msg = f"CrewAI event handler error (llm_started): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(LLMCallCompletedEvent)
        def on_llm_completed(
            source: Any, event: LLMCallCompletedEvent
        ) -> None:
            try:
                # Pop the LLM key from the current thread's stack
                llm_key = self._pop_llm_key()
                if not llm_key:
                    return

                gen = self._detach(llm_key, self._llm_spans)
                if not gen:
                    return

                response = getattr(event, "response", None)
                usage = getattr(response, "usage", None)

                # Extract output text — try multiple paths since
                # LiteLLM response format varies by provider
                output_text = None
                if response is not None:
                    # Path 1: ModelResponse.choices[0].message.content
                    choices = getattr(response, "choices", None)
                    if choices and len(choices) > 0:
                        choice = choices[0]
                        message = getattr(choice, "message", None)
                        if message:
                            output_text = getattr(message, "content", None)
                        # Path 2: choice might be a dict
                        if not output_text and isinstance(choice, dict):
                            msg = choice.get("message", {})
                            output_text = msg.get("content") if isinstance(msg, dict) else None

                    # Path 3: response might have text directly
                    if not output_text:
                        output_text = getattr(response, "text", None)

                    # Path 4: str(response) as last resort
                    if not output_text:
                        resp_str = str(response)
                        if len(resp_str) < 10000:
                            output_text = resp_str

                update_kwargs: Dict[str, Any] = {
                    "output": output_text,
                    "model": getattr(event, "model", None),
                }

                if usage:
                    update_kwargs["usage_details"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0)
                        or 0,
                        "completion_tokens": getattr(
                            usage, "completion_tokens", 0
                        )
                        or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0)
                        or 0,
                    }

                gen.update(**update_kwargs).end()
            except Exception as e:
                msg = f"CrewAI event handler error (llm_completed): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(LLMCallFailedEvent)
        def on_llm_failed(source: Any, event: LLMCallFailedEvent) -> None:
            try:
                llm_key = self._pop_llm_key()
                if not llm_key:
                    return

                gen = self._detach(llm_key, self._llm_spans)
                if gen:
                    error_msg = getattr(event, "error", "LLM call failed")
                    gen.update(
                        level="ERROR",
                        status_message=error_msg,
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (llm_failed): {e}"
                ants_platform_logger.warning(msg)

    # ── Tool Events ────────────────────────────────────────────────────

    def _register_tool_events(self, bus: Any) -> None:
        @bus.on(ToolUsageStartedEvent)
        def on_tool_started(source: Any, event: ToolUsageStartedEvent) -> None:
            try:
                tool_name = getattr(event, "tool_name", None) or "tool"
                tool_args = getattr(event, "tool_args", None)
                agent_id = getattr(event, "agent_id", None)

                # Build a unique key for this tool invocation
                tool_key = f"tool:{agent_id}:{tool_name}:{id(event)}"

                parent = self._get_parent_for_event(event)

                span = parent.start_observation(
                    name=tool_name,
                    as_type="tool",
                    agent_name=self._get_crew_agent_name(),
                    input={"tool_args": tool_args} if tool_args else None,
                    metadata={
                        "crewai.tool_class": getattr(
                            event, "tool_class", None
                        ),
                        "crewai.run_attempts": getattr(
                            event, "run_attempts", None
                        ),
                    },
                )

                # Atomically attach span AND set latest alias
                with self._lock:
                    ctx = trace.set_span_in_context(span._otel_span)
                    token = context.attach(ctx)
                    self._tool_spans[tool_key] = span
                    self._context_tokens[tool_key] = token
                    self._tool_spans[
                        f"latest:{agent_id}:{tool_name}"
                    ] = span
            except Exception as e:
                msg = f"CrewAI event handler error (tool_started): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(ToolUsageFinishedEvent)
        def on_tool_finished(
            source: Any, event: ToolUsageFinishedEvent
        ) -> None:
            try:
                tool_name = getattr(event, "tool_name", None) or "tool"
                agent_id = getattr(event, "agent_id", None)

                span = self._detach_tool_span(agent_id, tool_name)
                if span:
                    span.update(
                        output=getattr(event, "output", None),
                        metadata={
                            "crewai.from_cache": getattr(
                                event, "from_cache", None
                            ),
                            "crewai.started_at": str(
                                getattr(event, "started_at", None)
                            ),
                            "crewai.finished_at": str(
                                getattr(event, "finished_at", None)
                            ),
                        },
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (tool_finished): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(ToolUsageErrorEvent)
        def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            try:
                tool_name = getattr(event, "tool_name", None) or "tool"
                agent_id = getattr(event, "agent_id", None)

                span = self._detach_tool_span(agent_id, tool_name)
                if span:
                    error_msg = str(
                        getattr(event, "error", "Tool error")
                    )
                    span.update(
                        level="ERROR",
                        status_message=error_msg,
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (tool_error): {e}"
                ants_platform_logger.warning(msg)

    # ── Phase 2: Guardrail Events ──────────────────────────────────────

    def _register_guardrail_events(self, bus: Any) -> None:
        @bus.on(LLMGuardrailStartedEvent)
        def on_guardrail_started(
            source: Any, event: LLMGuardrailStartedEvent
        ) -> None:
            try:
                guardrail_name = getattr(event, "guardrail", None) or "guardrail"
                if callable(guardrail_name):
                    guardrail_name = getattr(
                        guardrail_name, "__name__", str(guardrail_name)
                    )
                guardrail_name = str(guardrail_name)

                # Use thread ID + event ID for unique correlation
                thread_id = threading.get_ident()
                key = f"guardrail:{thread_id}:{id(event)}"

                parent = self._get_parent_for_event(event)

                span = parent.start_observation(
                    name=f"Guardrail: {guardrail_name}",
                    as_type="guardrail",
                    agent_name=self._get_crew_agent_name(),
                    input={
                        "guardrail": guardrail_name,
                        "retry_count": getattr(event, "retry_count", 0),
                    },
                )
                self._attach(key, span, self._guardrail_spans)

                # Store key on thread-local for completion lookup
                self._push_guardrail_key(key)
            except Exception as e:
                msg = f"CrewAI event handler error (guardrail_started): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(LLMGuardrailCompletedEvent)
        def on_guardrail_completed(
            source: Any, event: LLMGuardrailCompletedEvent
        ) -> None:
            try:
                key = self._pop_guardrail_key()
                if not key:
                    return

                span = self._detach(key, self._guardrail_spans)
                if span:
                    span.update(
                        output={
                            "success": getattr(event, "success", None),
                            "result": str(getattr(event, "result", None)),
                            "error": getattr(event, "error", None),
                        },
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (guardrail_completed): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(LLMGuardrailFailedEvent)
        def on_guardrail_failed(
            source: Any, event: LLMGuardrailFailedEvent
        ) -> None:
            try:
                key = self._pop_guardrail_key()
                if not key:
                    return

                span = self._detach(key, self._guardrail_spans)
                if span:
                    error_msg = getattr(event, "error", "Guardrail failed")
                    span.update(
                        level="ERROR",
                        status_message=error_msg,
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (guardrail_failed): {e}"
                ants_platform_logger.warning(msg)

    # ── Phase 2: Reasoning Events ──────────────────────────────────────

    def _register_reasoning_events(self, bus: Any) -> None:
        @bus.on(AgentReasoningStartedEvent)
        def on_reasoning_started(
            source: Any, event: AgentReasoningStartedEvent
        ) -> None:
            try:
                agent_role = getattr(event, "agent_role", None) or "agent"
                task_id = getattr(event, "task_id", "")
                attempt = getattr(event, "attempt", 1)
                key = f"reasoning:{task_id}:{attempt}"

                parent = self._get_parent_for_event(event)

                span = parent.start_observation(
                    name=f"Reasoning ({agent_role}, attempt {attempt})",
                    as_type="span",
                    agent_name=self._get_crew_agent_name(),
                    input={
                        "agent_role": agent_role,
                        "task_id": task_id,
                        "attempt": attempt,
                    },
                )
                self._attach(key, span, self._reasoning_spans)
            except Exception as e:
                msg = f"CrewAI event handler error (reasoning_started): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(AgentReasoningCompletedEvent)
        def on_reasoning_completed(
            source: Any, event: AgentReasoningCompletedEvent
        ) -> None:
            try:
                task_id = getattr(event, "task_id", "")
                attempt = getattr(event, "attempt", 1)
                key = f"reasoning:{task_id}:{attempt}"

                span = self._detach(key, self._reasoning_spans)
                if span:
                    span.update(
                        output={
                            "plan": getattr(event, "plan", None),
                            "ready": getattr(event, "ready", None),
                        },
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (reasoning_completed): {e}"
                ants_platform_logger.warning(msg)

        @bus.on(AgentReasoningFailedEvent)
        def on_reasoning_failed(
            source: Any, event: AgentReasoningFailedEvent
        ) -> None:
            try:
                task_id = getattr(event, "task_id", "")
                attempt = getattr(event, "attempt", 1)
                key = f"reasoning:{task_id}:{attempt}"

                span = self._detach(key, self._reasoning_spans)
                if span:
                    error_msg = getattr(event, "error", "Reasoning failed")
                    span.update(
                        level="ERROR",
                        status_message=error_msg,
                    ).end()
            except Exception as e:
                msg = f"CrewAI event handler error (reasoning_failed): {e}"
                ants_platform_logger.warning(msg)

    # ── Thread-local guardrail key management ──────────────────────────

    def _push_guardrail_key(self, key: str) -> None:
        """Push a guardrail span key onto the current thread's stack."""
        stack = getattr(self._thread_local, "guardrail_stack", None)
        if stack is None:
            stack = []
            self._thread_local.guardrail_stack = stack
        stack.append(key)

    def _pop_guardrail_key(self) -> Optional[str]:
        """Pop the most recent guardrail span key from the current thread."""
        stack = getattr(self._thread_local, "guardrail_stack", None)
        if stack:
            return stack.pop()
        return None
