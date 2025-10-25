"""Tracer implementation for AntsPlatform OpenTelemetry integration.

This module provides the LangfuseTracer class, a thread-safe singleton that manages OpenTelemetry
tracing infrastructure for AntsPlatform. It handles tracer initialization, span processors,
API clients, and coordinates background tasks for efficient data processing and media handling.

Key features:
- Thread-safe OpenTelemetry tracer with AntsPlatform-specific span processors and sampling
- Configurable batch processing of spans and scores with intelligent flushing behavior
- Asynchronous background media upload processing with dedicated worker threads
- Concurrent score ingestion with batching and retry mechanisms
- Automatic project ID discovery and caching
- Graceful shutdown handling with proper resource cleanup
- Fault tolerance with detailed error logging and recovery mechanisms
"""

import atexit
import os
import threading
from queue import Full, Queue
from typing import Any, Dict, List, Optional, cast

import httpx
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import Decision, TraceIdRatioBased
from opentelemetry.trace import Tracer

from ants_platform._client.attributes import AntsPlatformOtelSpanAttributes
from ants_platform._client.constants import ANTS_PLATFORM_TRACER_NAME
from ants_platform._client.environment_variables import (
    ANTS_PLATFORM_MEDIA_UPLOAD_ENABLED,
    ANTS_PLATFORM_MEDIA_UPLOAD_THREAD_COUNT,
    ANTS_PLATFORM_RELEASE,
    ANTS_PLATFORM_TRACING_ENVIRONMENT,
)
from ants_platform._client.span_processor import AntsPlatformSpanProcessor
from ants_platform._task_manager.media_manager import MediaManager
from ants_platform._task_manager.media_upload_consumer import MediaUploadConsumer
from ants_platform._task_manager.score_ingestion_consumer import ScoreIngestionConsumer
from ants_platform._utils.environment import get_common_release_envs
from ants_platform._utils.prompt_cache import PromptCache
from ants_platform._utils.request import LangfuseClient
from ants_platform.api.client import AsyncFernLangfuse, FernLangfuse
from ants_platform.logger import ants_platform_logger
from ants_platform.types import MaskFunction

from ..version import __version__ as ants_platform_version


class LangfuseResourceManager:
    """Thread-safe singleton that provides access to the OpenTelemetry tracer and processors.

    This class implements a thread-safe singleton pattern keyed by the public API key,
    ensuring that only one tracer instance exists per API key combination. It manages
    the lifecycle of the OpenTelemetry tracer provider, span processors, and resource
    attributes, as well as background threads for media uploads and score ingestion.

    The tracer is responsible for:
    1. Setting up the OpenTelemetry tracer with appropriate sampling and configuration
    2. Managing the span processor for exporting spans to the AntsPlatform API
    3. Creating and managing AntsPlatform API clients (both synchronous and asynchronous)
    4. Handling background media upload processing via dedicated worker threads
    5. Processing and batching score ingestion events with configurable flush settings
    6. Retrieving and caching project information for URL generation and media handling
    7. Coordinating graceful shutdown of all background processes with proper resource cleanup

    This implementation follows best practices for resource management in long-running
    applications, including thread-safe singleton pattern, bounded queues to prevent memory
    exhaustion, proper resource cleanup on shutdown, and fault-tolerant error handling with
    detailed logging.

    Thread safety is ensured through the use of locks, thread-safe queues, and atomic operations,
    making this implementation suitable for multi-threaded and asyncio applications.
    """

    _instances: Dict[str, "LangfuseResourceManager"] = {}
    _lock = threading.RLock()

    def __new__(
        cls,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        httpx_client: Optional[httpx.Client] = None,
        media_upload_thread_count: Optional[int] = None,
        sample_rate: Optional[float] = None,
        mask: Optional[MaskFunction] = None,
        tracing_enabled: Optional[bool] = None,
        blocked_instrumentation_scopes: Optional[List[str]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        tracer_provider: Optional[TracerProvider] = None,
    ) -> "LangfuseResourceManager":
        if public_key in cls._instances:
            return cls._instances[public_key]

        with cls._lock:
            if public_key not in cls._instances:
                instance = super(LangfuseResourceManager, cls).__new__(cls)

                # Initialize tracer (will be noop until init instance)
                instance._otel_tracer = otel_trace_api.get_tracer(
                    ANTS_PLATFORM_TRACER_NAME,
                    ants_platform_version,
                    attributes={"public_key": public_key},
                )

                instance._initialize_instance(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    timeout=timeout,
                    environment=environment,
                    release=release,
                    flush_at=flush_at,
                    flush_interval=flush_interval,
                    httpx_client=httpx_client,
                    media_upload_thread_count=media_upload_thread_count,
                    sample_rate=sample_rate,
                    mask=mask,
                    tracing_enabled=tracing_enabled
                    if tracing_enabled is not None
                    else True,
                    blocked_instrumentation_scopes=blocked_instrumentation_scopes,
                    additional_headers=additional_headers,
                    tracer_provider=tracer_provider,
                )

                cls._instances[public_key] = instance

            return cls._instances[public_key]

    def _initialize_instance(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        media_upload_thread_count: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
        sample_rate: Optional[float] = None,
        mask: Optional[MaskFunction] = None,
        tracing_enabled: bool = True,
        blocked_instrumentation_scopes: Optional[List[str]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        tracer_provider: Optional[TracerProvider] = None,
    ) -> None:
        self.public_key = public_key
        self.secret_key = secret_key
        self.tracing_enabled = tracing_enabled
        self.host = host
        self.mask = mask

        # OTEL Tracer
        if tracing_enabled:
            tracer_provider = tracer_provider or _init_tracer_provider(
                environment=environment, release=release, sample_rate=sample_rate
            )

            ants_platform_processor = AntsPlatformSpanProcessor(
                public_key=self.public_key,
                secret_key=secret_key,
                host=host,
                timeout=timeout,
                flush_at=flush_at,
                flush_interval=flush_interval,
                blocked_instrumentation_scopes=blocked_instrumentation_scopes,
                additional_headers=additional_headers,
            )
            tracer_provider.add_span_processor(ants_platform_processor)

            self._otel_tracer = tracer_provider.get_tracer(
                ANTS_PLATFORM_TRACER_NAME,
                ants_platform_version,
                attributes={"public_key": self.public_key},
            )

        # API Clients

        ## API clients must be singletons because the underlying HTTPX clients
        ## use connection pools with limited capacity. Creating multiple instances
        ## could exhaust the OS's maximum number of available TCP sockets (file descriptors),
        ## leading to connection errors.
        if httpx_client is not None:
            self.httpx_client = httpx_client
        else:
            # Create a new httpx client with additional_headers if provided
            client_headers = additional_headers if additional_headers else {}
            self.httpx_client = httpx.Client(timeout=timeout, headers=client_headers)

        self.api = FernLangfuse(
            base_url=host,
            username=self.public_key,
            password=secret_key,
            x_ants_platform_sdk_name="python",
            x_ants_platform_sdk_version=ants_platform_version,
            x_ants_platform_public_key=self.public_key,
            httpx_client=self.httpx_client,
            timeout=timeout,
        )
        self.async_api = AsyncFernLangfuse(
            base_url=host,
            username=self.public_key,
            password=secret_key,
            x_ants_platform_sdk_name="python",
            x_ants_platform_sdk_version=ants_platform_version,
            x_ants_platform_public_key=self.public_key,
            timeout=timeout,
        )
        score_ingestion_client = LangfuseClient(
            public_key=self.public_key,
            secret_key=secret_key,
            base_url=host,
            version=ants_platform_version,
            timeout=timeout or 20,
            session=self.httpx_client,
        )

        # Media
        self._media_upload_enabled = os.environ.get(
            ANTS_PLATFORM_MEDIA_UPLOAD_ENABLED, "True"
        ).lower() not in ("false", "0")

        self._media_upload_queue: Queue[Any] = Queue(100_000)
        self._media_manager = MediaManager(
            api_client=self.api,
            media_upload_queue=self._media_upload_queue,
            max_retries=3,
        )
        self._media_upload_consumers = []

        media_upload_thread_count = media_upload_thread_count or max(
            int(os.getenv(ANTS_PLATFORM_MEDIA_UPLOAD_THREAD_COUNT, 1)), 1
        )

        if self._media_upload_enabled:
            for i in range(media_upload_thread_count):
                media_upload_consumer = MediaUploadConsumer(
                    identifier=i,
                    media_manager=self._media_manager,
                )
                media_upload_consumer.start()
                self._media_upload_consumers.append(media_upload_consumer)

        # Prompt cache
        self.prompt_cache = PromptCache()

        # Score ingestion
        self._score_ingestion_queue: Queue[Any] = Queue(100_000)
        self._ingestion_consumers = []

        ingestion_consumer = ScoreIngestionConsumer(
            ingestion_queue=self._score_ingestion_queue,
            identifier=0,
            client=score_ingestion_client,
            flush_at=flush_at,
            flush_interval=flush_interval,
            max_retries=3,
            public_key=self.public_key,
        )
        ingestion_consumer.start()
        self._ingestion_consumers.append(ingestion_consumer)

        # Register shutdown handler
        atexit.register(self.shutdown)

        ants_platform_logger.info(
            f"Startup: AntsPlatform tracer successfully initialized | "
            f"public_key={self.public_key} | "
            f"host={host} | "
            f"environment={environment or 'default'} | "
            f"sample_rate={sample_rate if sample_rate is not None else 1.0} | "
            f"media_threads={media_upload_thread_count or 1}"
        )

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            for key in cls._instances:
                cls._instances[key].shutdown()

            cls._instances.clear()

    def add_score_task(self, event: dict, *, force_sample: bool = False) -> None:
        try:
            # Sample scores with the same sampler that is used for tracing
            tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
            should_sample = (
                force_sample
                or isinstance(
                    tracer_provider, otel_trace_api.ProxyTracerProvider
                )  # default to in-sample if otel sampler is not available
                or (
                    (
                        tracer_provider.sampler.should_sample(
                            parent_context=None,
                            trace_id=int(event["body"].trace_id, 16),
                            name="score",
                        ).decision
                        == Decision.RECORD_AND_SAMPLE
                        if hasattr(event["body"], "trace_id")
                        else True
                    )
                    if event["body"].trace_id
                    is not None  # do not sample out session / dataset run scores
                    else True
                )
            )

            if should_sample:
                ants_platform_logger.debug(
                    f"Score: Enqueuing event type={event['type']} for trace_id={event['body'].trace_id} name={event['body'].name} value={event['body'].value}"
                )
                self._score_ingestion_queue.put(event, block=False)

        except Full:
            ants_platform_logger.warning(
                "System overload: Score ingestion queue has reached capacity (100,000 items). Score will be dropped. Consider increasing flush frequency or decreasing event volume."
            )

            return
        except Exception as e:
            ants_platform_logger.error(
                f"Unexpected error: Failed to process score event. The score will be dropped. Error details: {e}"
            )

            return

    @property
    def tracer(self) -> Optional[Tracer]:
        return self._otel_tracer

    @staticmethod
    def get_current_span() -> Any:
        return otel_trace_api.get_current_span()

    def _stop_and_join_consumer_threads(self) -> None:
        """End the consumer threads once the queue is empty.

        Blocks execution until finished
        """
        ants_platform_logger.debug(
            f"Shutdown: Waiting for {len(self._media_upload_consumers)} media upload thread(s) to complete processing"
        )
        for media_upload_consumer in self._media_upload_consumers:
            media_upload_consumer.pause()

        for media_upload_consumer in self._media_upload_consumers:
            try:
                media_upload_consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            ants_platform_logger.debug(
                f"Shutdown: Media upload thread #{media_upload_consumer._identifier} successfully terminated"
            )

        ants_platform_logger.debug(
            f"Shutdown: Waiting for {len(self._ingestion_consumers)} score ingestion thread(s) to complete processing"
        )
        for score_ingestion_consumer in self._ingestion_consumers:
            score_ingestion_consumer.pause()

        for score_ingestion_consumer in self._ingestion_consumers:
            try:
                score_ingestion_consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            ants_platform_logger.debug(
                f"Shutdown: Score ingestion thread #{score_ingestion_consumer._identifier} successfully terminated"
            )

    def flush(self) -> None:
        tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
        if isinstance(tracer_provider, otel_trace_api.ProxyTracerProvider):
            return

        tracer_provider.force_flush()
        ants_platform_logger.debug("Successfully flushed OTEL tracer provider")

        self._score_ingestion_queue.join()
        ants_platform_logger.debug("Successfully flushed score ingestion queue")

        self._media_upload_queue.join()
        ants_platform_logger.debug("Successfully flushed media upload queue")

    def shutdown(self) -> None:
        # Unregister the atexit handler first
        atexit.unregister(self.shutdown)

        tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
        if isinstance(tracer_provider, otel_trace_api.ProxyTracerProvider):
            return

        tracer_provider.force_flush()

        self._stop_and_join_consumer_threads()


def _init_tracer_provider(
    *,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    sample_rate: Optional[float] = None,
) -> TracerProvider:
    environment = environment or os.environ.get(ANTS_PLATFORM_TRACING_ENVIRONMENT)
    release = release or os.environ.get(ANTS_PLATFORM_RELEASE) or get_common_release_envs()

    resource_attributes = {
        AntsPlatformOtelSpanAttributes.ENVIRONMENT: environment,
        AntsPlatformOtelSpanAttributes.RELEASE: release,
    }

    resource = Resource.create(
        {k: v for k, v in resource_attributes.items() if v is not None}
    )

    provider = None
    default_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())

    if isinstance(default_provider, otel_trace_api.ProxyTracerProvider):
        provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(sample_rate)
            if sample_rate is not None and sample_rate < 1
            else None,
        )
        otel_trace_api.set_tracer_provider(provider)

    else:
        provider = default_provider

    return provider
