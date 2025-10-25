"""Span processor for AntsPlatform OpenTelemetry integration.

This module defines the AntsPlatformSpanProcessor class, which extends OpenTelemetry's
BatchSpanProcessor with AntsPlatform-specific functionality. It handles exporting
spans to the AntsPlatform API with proper authentication and filtering.

Key features:
- HTTP-based span export to AntsPlatform API
- Basic authentication with AntsPlatform API keys
- Configurable batch processing behavior
- Project-scoped span filtering to prevent cross-project data leakage
"""

import base64
import os
from typing import Dict, List, Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ants_platform._client.constants import ANTS_PLATFORM_TRACER_NAME
from ants_platform._client.environment_variables import (
    ANTS_PLATFORM_FLUSH_AT,
    ANTS_PLATFORM_FLUSH_INTERVAL,
)
from ants_platform._client.utils import span_formatter
from ants_platform.logger import ants_platform_logger
from ants_platform.version import __version__ as ants_platform_version


class AntsPlatformSpanProcessor(BatchSpanProcessor):
    """OpenTelemetry span processor that exports spans to the AntsPlatform API.

    This processor extends OpenTelemetry's BatchSpanProcessor with AntsPlatform-specific functionality:
    1. Project-scoped span filtering to prevent cross-project data leakage
    2. Instrumentation scope filtering to block spans from specific libraries/frameworks
    3. Configurable batch processing parameters for optimal performance
    4. HTTP-based span export to the AntsPlatform OTLP endpoint
    5. Debug logging for span processing operations
    6. Authentication with AntsPlatform API using Basic Auth

    The processor is designed to efficiently handle large volumes of spans with
    minimal overhead, while ensuring spans are only sent to the correct project.
    It integrates with OpenTelemetry's standard span lifecycle, adding AntsPlatform-specific
    filtering and export capabilities.
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        blocked_instrumentation_scopes: Optional[List[str]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ):
        self.public_key = public_key
        self.blocked_instrumentation_scopes = (
            blocked_instrumentation_scopes
            if blocked_instrumentation_scopes is not None
            else []
        )

        env_flush_at = os.environ.get(ANTS_PLATFORM_FLUSH_AT, None)
        flush_at = flush_at or int(env_flush_at) if env_flush_at is not None else None

        env_flush_interval = os.environ.get(ANTS_PLATFORM_FLUSH_INTERVAL, None)
        flush_interval = (
            flush_interval or float(env_flush_interval)
            if env_flush_interval is not None
            else None
        )

        basic_auth_header = "Basic " + base64.b64encode(
            f"{public_key}:{secret_key}".encode("utf-8")
        ).decode("ascii")

        # Prepare default headers
        default_headers = {
            "Authorization": basic_auth_header,
            "x_ants_platform_sdk_name": "python",
            "x_ants_platform_sdk_version": ants_platform_version,
            "x_ants_platform_public_key": public_key,
        }

        # Merge additional headers if provided
        headers = {**default_headers, **(additional_headers or {})}

        ants_platform_span_exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers=headers,
            timeout=timeout,
        )

        super().__init__(
            span_exporter=ants_platform_span_exporter,
            export_timeout_millis=timeout * 1_000 if timeout else None,
            max_export_batch_size=flush_at,
            schedule_delay_millis=flush_interval * 1_000
            if flush_interval is not None
            else None,
        )

    def on_end(self, span: ReadableSpan) -> None:
        # Only export spans that belong to the scoped project
        # This is important to not send spans to wrong project in multi-project setups
        if self._is_ants_platform_span(span) and not self._is_ants_platform_project_span(span):
            ants_platform_logger.debug(
                f"Security: Span rejected - belongs to project '{span.instrumentation_scope.attributes.get('public_key') if span.instrumentation_scope and span.instrumentation_scope.attributes else None}' but processor is for '{self.public_key}'. "
                f"This prevents cross-project data leakage in multi-project environments."
            )
            return

        # Do not export spans from blocked instrumentation scopes
        if self._is_blocked_instrumentation_scope(span):
            return

        ants_platform_logger.debug(
            f"Trace: Processing span name='{span._name}' | Full details:\n{span_formatter(span)}"
        )

        super().on_end(span)

    @staticmethod
    def _is_ants_platform_span(span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None
            and span.instrumentation_scope.name == ANTS_PLATFORM_TRACER_NAME
        )

    def _is_blocked_instrumentation_scope(self, span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None
            and span.instrumentation_scope.name in self.blocked_instrumentation_scopes
        )

    def _is_ants_platform_project_span(self, span: ReadableSpan) -> bool:
        if not AntsPlatformSpanProcessor._is_ants_platform_span(span):
            return False

        if span.instrumentation_scope is not None:
            public_key_on_span = (
                span.instrumentation_scope.attributes.get("public_key", None)
                if span.instrumentation_scope.attributes
                else None
            )

            return public_key_on_span == self.public_key

        return False
