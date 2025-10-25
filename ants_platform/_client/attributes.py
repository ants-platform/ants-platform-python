"""Span attribute management for AntsPlatform OpenTelemetry integration.

This module defines constants and functions for managing OpenTelemetry span attributes
used by AntsPlatform. It provides a structured approach to creating and manipulating
attributes for different span types (trace, span, generation) while ensuring consistency.

The module includes:
- Attribute name constants organized by category
- Functions to create attribute dictionaries for different entity types
- Utilities for serializing and processing attribute values
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from ants_platform._client.constants import (
    ObservationTypeGenerationLike,
    ObservationTypeSpanLike,
)

from ants_platform._utils.serializer import EventSerializer
from ants_platform.model import PromptClient
from ants_platform.types import MapValue, SpanLevel


class AntsPlatformOtelSpanAttributes:
    # AntsPlatform-Trace attributes
    TRACE_NAME = "ants_platform.trace.name"
    TRACE_USER_ID = "user.id"
    TRACE_SESSION_ID = "session.id"
    TRACE_TAGS = "ants_platform.trace.tags"
    TRACE_PUBLIC = "ants_platform.trace.public"
    TRACE_METADATA = "ants_platform.trace.metadata"
    TRACE_INPUT = "ants_platform.trace.input"
    TRACE_OUTPUT = "ants_platform.trace.output"

    # AntsPlatform-observation attributes
    OBSERVATION_TYPE = "ants_platform.observation.type"
    OBSERVATION_METADATA = "ants_platform.observation.metadata"
    OBSERVATION_LEVEL = "ants_platform.observation.level"
    OBSERVATION_STATUS_MESSAGE = "ants_platform.observation.status_message"
    OBSERVATION_INPUT = "ants_platform.observation.input"
    OBSERVATION_OUTPUT = "ants_platform.observation.output"

    # AntsPlatform-observation of type Generation attributes
    OBSERVATION_COMPLETION_START_TIME = "ants_platform.observation.completion_start_time"
    OBSERVATION_MODEL = "ants_platform.observation.model.name"
    OBSERVATION_MODEL_PARAMETERS = "ants_platform.observation.model.parameters"
    OBSERVATION_USAGE_DETAILS = "ants_platform.observation.usage_details"
    OBSERVATION_COST_DETAILS = "ants_platform.observation.cost_details"
    OBSERVATION_PROMPT_NAME = "ants_platform.observation.prompt.name"
    OBSERVATION_PROMPT_VERSION = "ants_platform.observation.prompt.version"

    # General
    ENVIRONMENT = "ants_platform.environment"
    RELEASE = "ants_platform.release"
    VERSION = "ants_platform.version"

    # Internal
    AS_ROOT = "ants_platform.internal.as_root"


def create_trace_attributes(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Any] = None,
    tags: Optional[List[str]] = None,
    public: Optional[bool] = None,
) -> dict:
    attributes = {
        AntsPlatformOtelSpanAttributes.TRACE_NAME: name,
        AntsPlatformOtelSpanAttributes.TRACE_USER_ID: user_id,
        AntsPlatformOtelSpanAttributes.TRACE_SESSION_ID: session_id,
        AntsPlatformOtelSpanAttributes.VERSION: version,
        AntsPlatformOtelSpanAttributes.RELEASE: release,
        AntsPlatformOtelSpanAttributes.TRACE_INPUT: _serialize(input),
        AntsPlatformOtelSpanAttributes.TRACE_OUTPUT: _serialize(output),
        AntsPlatformOtelSpanAttributes.TRACE_TAGS: tags,
        AntsPlatformOtelSpanAttributes.TRACE_PUBLIC: public,
        **_flatten_and_serialize_metadata(metadata, "trace"),
    }

    return {k: v for k, v in attributes.items() if v is not None}


def create_span_attributes(
    *,
    metadata: Optional[Any] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    level: Optional[SpanLevel] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    observation_type: Optional[
        Union[ObservationTypeSpanLike, Literal["event"]]
    ] = "span",
) -> dict:
    attributes = {
        AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL: level,
        AntsPlatformOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        AntsPlatformOtelSpanAttributes.VERSION: version,
        AntsPlatformOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        AntsPlatformOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        **_flatten_and_serialize_metadata(metadata, "observation"),
    }

    return {k: v for k, v in attributes.items() if v is not None}


def create_generation_attributes(
    *,
    name: Optional[str] = None,
    completion_start_time: Optional[datetime] = None,
    metadata: Optional[Any] = None,
    level: Optional[SpanLevel] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    model: Optional[str] = None,
    model_parameters: Optional[Dict[str, MapValue]] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    usage_details: Optional[Dict[str, int]] = None,
    cost_details: Optional[Dict[str, float]] = None,
    prompt: Optional[PromptClient] = None,
    observation_type: Optional[ObservationTypeGenerationLike] = "generation",
) -> dict:
    attributes = {
        AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL: level,
        AntsPlatformOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        AntsPlatformOtelSpanAttributes.VERSION: version,
        AntsPlatformOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        AntsPlatformOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        AntsPlatformOtelSpanAttributes.OBSERVATION_MODEL: model,
        AntsPlatformOtelSpanAttributes.OBSERVATION_PROMPT_NAME: prompt.name
        if prompt and not prompt.is_fallback
        else None,
        AntsPlatformOtelSpanAttributes.OBSERVATION_PROMPT_VERSION: prompt.version
        if prompt and not prompt.is_fallback
        else None,
        AntsPlatformOtelSpanAttributes.OBSERVATION_USAGE_DETAILS: _serialize(usage_details),
        AntsPlatformOtelSpanAttributes.OBSERVATION_COST_DETAILS: _serialize(cost_details),
        AntsPlatformOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME: _serialize(
            completion_start_time
        ),
        AntsPlatformOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS: _serialize(
            model_parameters
        ),
        **_flatten_and_serialize_metadata(metadata, "observation"),
    }

    return {k: v for k, v in attributes.items() if v is not None}


def _serialize(obj: Any) -> Optional[str]:
    if obj is None or isinstance(obj, str):
        return obj

    return json.dumps(obj, cls=EventSerializer)


def _flatten_and_serialize_metadata(
    metadata: Any, type: Literal["observation", "trace"]
) -> dict:
    prefix = (
        AntsPlatformOtelSpanAttributes.OBSERVATION_METADATA
        if type == "observation"
        else AntsPlatformOtelSpanAttributes.TRACE_METADATA
    )

    metadata_attributes: Dict[str, Union[str, int, None]] = {}

    if not isinstance(metadata, dict):
        metadata_attributes[prefix] = _serialize(metadata)
    else:
        for key, value in metadata.items():
            metadata_attributes[f"{prefix}.{key}"] = (
                value
                if isinstance(value, str) or isinstance(value, int)
                else _serialize(value)
            )

    return metadata_attributes
