"""Environment variable definitions for Ants Platform OpenTelemetry integration.

This module defines environment variables used to configure the Ants Platform OpenTelemetry integration.
Each environment variable includes documentation on its purpose, expected values, and defaults.
"""

ANTS_PLATFORM_TRACING_ENVIRONMENT = "ANTS_PLATFORM_TRACING_ENVIRONMENT"
"""
.. envvar:: ANTS_PLATFORM_TRACING_ENVIRONMENT

The tracing environment. Can be any lowercase alphanumeric string with hyphens and underscores that does not start with 'ants-platform'.

**Default value:** ``"default"``
"""

ANTS_PLATFORM_RELEASE = "ANTS_PLATFORM_RELEASE"
"""
.. envvar:: ANTS_PLATFORM_RELEASE

Release number/hash of the application to provide analytics grouped by release.
"""


ANTS_PLATFORM_PUBLIC_KEY = "ANTS_PLATFORM_PUBLIC_KEY"
"""
.. envvar:: ANTS_PLATFORM_PUBLIC_KEY

Public API key of Ants Platform project
"""

ANTS_PLATFORM_SECRET_KEY = "ANTS_PLATFORM_SECRET_KEY"
"""
.. envvar:: ANTS_PLATFORM_SECRET_KEY

Secret API key of Ants Platform project
"""

ANTS_PLATFORM_HOST = "ANTS_PLATFORM_HOST"
"""
.. envvar:: ANTS_PLATFORM_HOST

Host of Ants Platform API. Can be set via `ANTS_PLATFORM_HOST` environment variable.

**Default value:** ``"https://cloud.ants-platform.com"``
"""

ANTS_PLATFORM_DEBUG = "ANTS_PLATFORM_DEBUG"
"""
.. envvar:: ANTS_PLATFORM_DEBUG

Enables debug mode for more verbose logging.

**Default value:** ``"False"``
"""

ANTS_PLATFORM_TRACING_ENABLED = "ANTS_PLATFORM_TRACING_ENABLED"
"""
.. envvar:: ANTS_PLATFORM_TRACING_ENABLED

Enables or disables the Ants Platform client. If disabled, all observability calls to the backend will be no-ops. Default is True. Set to `False` to disable tracing.

**Default value:** ``"True"``
"""

ANTS_PLATFORM_MEDIA_UPLOAD_THREAD_COUNT = "ANTS_PLATFORM_MEDIA_UPLOAD_THREAD_COUNT"
"""
.. envvar:: ANTS_PLATFORM_MEDIA_UPLOAD_THREAD_COUNT 

Number of background threads to handle media uploads from trace ingestion.

**Default value:** ``1``
"""

ANTS_PLATFORM_FLUSH_AT = "ANTS_PLATFORM_FLUSH_AT"
"""
.. envvar:: ANTS_PLATFORM_FLUSH_AT

Max batch size until a new ingestion batch is sent to the API.
**Default value:** same as OTEL ``OTEL_BSP_MAX_EXPORT_BATCH_SIZE``
"""

ANTS_PLATFORM_FLUSH_INTERVAL = "ANTS_PLATFORM_FLUSH_INTERVAL"
"""
.. envvar:: ANTS_PLATFORM_FLUSH_INTERVAL

Max delay in seconds until a new ingestion batch is sent to the API.
**Default value:** same as OTEL ``OTEL_BSP_SCHEDULE_DELAY``
"""

ANTS_PLATFORM_SAMPLE_RATE = "ANTS_PLATFORM_SAMPLE_RATE"
"""
.. envvar: ANTS_PLATFORM_SAMPLE_RATE

Float between 0 and 1 indicating the sample rate of traces to bet sent to Ants Platform servers.

**Default value**: ``1.0``

"""
ANTS_PLATFORM_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED = (
    "ANTS_PLATFORM_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED"
)
"""
.. envvar: ANTS_PLATFORM_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED

Default capture of function args, kwargs and return value when using the @observe decorator.

Having default IO capture enabled for observe decorated function may have a performance impact on your application
if large or deeply nested objects are attempted to be serialized. Set this value to `False` and use manual
input/output setting on your observation to avoid this.

**Default value**: ``True``
"""

ANTS_PLATFORM_MEDIA_UPLOAD_ENABLED = "ANTS_PLATFORM_MEDIA_UPLOAD_ENABLED"
"""
.. envvar: ANTS_PLATFORM_MEDIA_UPLOAD_ENABLED

Controls whether media detection and upload is attempted by the SDK.

**Default value**: ``True``
"""

ANTS_PLATFORM_TIMEOUT = "ANTS_PLATFORM_TIMEOUT"
"""
.. envvar: ANTS_PLATFORM_TIMEOUT

Controls the timeout for all API requests in seconds

**Default value**: ``5``
"""

ANTS_PLATFORM_PROMPT_CACHE_DEFAULT_TTL_SECONDS = "ANTS_PLATFORM_PROMPT_CACHE_DEFAULT_TTL_SECONDS"
"""
.. envvar: ANTS_PLATFORM_PROMPT_CACHE_DEFAULT_TTL_SECONDS

Controls the default time-to-live (TTL) in seconds for cached prompts.
This setting determines how long prompt responses are cached before they expire.

**Default value**: ``60``
"""
