"""Span attribute management for AntsPlatform OpenTelemetry integration.

This module defines constants and functions for managing OpenTelemetry span attributes
used by AntsPlatform. It provides a structured approach to creating and manipulating
attributes for different span types (trace, span, generation) while ensuring consistency.

The module includes:
- Attribute name constants organized by category
- Functions to create attribute dictionaries for different entity types
- Utilities for serializing and processing attribute values
"""

import hashlib
import json
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from ants_platform._client.constants import (
    ObservationTypeGenerationLike,
    ObservationTypeSpanLike,
)

from ants_platform._utils.serializer import EventSerializer
from ants_platform.model import PromptClient
from ants_platform.types import MapValue, SpanLevel

# Agent name validation constants
AGENT_NAME_MAX_LENGTH = 255

# Configure logger
logger = logging.getLogger(__name__)


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

    # Agent identification attributes
    AGENT_ID = "ants_platform.agent.id"
    AGENT_NAME = "ants_platform.agent.name"
    AGENT_DISPLAY_NAME = "ants_platform.agent.display_name"

    # Internal
    AS_ROOT = "ants_platform.internal.as_root"


def validate_and_normalize_agent_name(agent_name: Optional[str]) -> Optional[str]:
    """Validate and normalize agent_name to lowercase snake_case.

    Applies the following transformations:
    1. Strip whitespace
    2. Convert to lowercase for consistency (Python PEP 8 snake_case convention)
    3. Validate max length (255 chars), truncate with warning if exceeded
    4. Convert empty strings to None

    Args:
        agent_name: Raw agent name string

    Returns:
        Normalized agent name (lowercase) or None if invalid/empty

    Examples:
        >>> validate_and_normalize_agent_name("QA_Agent")
        "qa_agent"
        >>> validate_and_normalize_agent_name("qa_agent")
        "qa_agent"
        >>> validate_and_normalize_agent_name("  QA_Agent  ")
        "qa_agent"
        >>> validate_and_normalize_agent_name("")
        None
        >>> validate_and_normalize_agent_name("A" * 300)  # Truncates to 255 with warning
        "aaa..."
    """
    if agent_name is None:
        return None

    # Strip whitespace
    agent_name = agent_name.strip()

    # Convert empty strings to None
    if not agent_name:
        return None

    # Normalize to lowercase for consistency (Python PEP 8 snake_case convention)
    agent_name = agent_name.lower()

    # Validate max length and truncate if needed
    if len(agent_name) > AGENT_NAME_MAX_LENGTH:
        original_length = len(agent_name)
        agent_name = agent_name[:AGENT_NAME_MAX_LENGTH]
        warning_message = (
            f"agent_name exceeds maximum length of {AGENT_NAME_MAX_LENGTH} characters "
            f"(was {original_length} chars). Truncated to: '{agent_name}'"
        )
        warnings.warn(warning_message, UserWarning, stacklevel=3)

    return agent_name


def generate_agent_id(agent_name: str, project_id: str) -> str:
    """Generate a stable 16-character hexadecimal agent_id from agent_name and project_id using BLAKE2b-128.

    The agent_id is a deterministic hash of both agent_name and project_id, ensuring:
    1. Same agent_name in same project = same agent_id (deterministic)
    2. Same agent_name in different projects = different agent_id (project-scoped)
    3. Transfer-safe: projectId doesn't change, so agent_id stays stable

    Args:
        agent_name: Normalized agent name (should be already validated and normalized)
        project_id: The project ID this agent belongs to

    Returns:
        str: 16 lowercase hexadecimal characters (64 bits)

    Examples:
        >>> generate_agent_id("qa_agent", "project-123")
        "a1b2c3d4e5f6g7h8"
        >>> generate_agent_id("qa_agent", "project-123")  # Same = same output
        "a1b2c3d4e5f6g7h8"
        >>> generate_agent_id("qa_agent", "project-456")  # Different project
        "x9y8z7w6v5u4t3s2"

    Note:
        Uses BLAKE2b with 8-byte digest (64 bits) to produce 16 hex characters.
        BLAKE2b is faster than SHA-256 and designed for exactly this use case.
    """
    # Validate inputs
    if not agent_name or not isinstance(agent_name, str):
        raise ValueError("agent_name must be a non-empty string")

    if not project_id or not isinstance(project_id, str):
        raise ValueError("project_id must be a non-empty string")

    # Use BLAKE2b with 8-byte digest (64 bits = 16 hex characters)
    # Hash both agent_name and project_id for project-scoped agent IDs
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(agent_name.encode('utf-8'))
    hasher.update(project_id.encode('utf-8'))
    agent_id = hasher.hexdigest()

    logger.info(f"[AGENT_ID] Generated: {agent_id} from agent_name: {agent_name}")

    return agent_id


def create_trace_attributes(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_display_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    project_id: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Any] = None,
    tags: Optional[List[str]] = None,
    public: Optional[bool] = None,
) -> dict:
    """Create trace attributes with agent identification.

    Agent Identity Flow:
    - agent_name (immutable): Stable identifier used for agent_id generation
    - agent_display_name (mutable): Optional user-facing name stored in PostgreSQL
    - agent_id: Auto-generated 16-char hex hash of agent_name + project_id using BLAKE2b-64

    If only agent_name is provided, it serves as both identifier and display name.
    """
    # Validate and normalize agent_name
    agent_name = validate_and_normalize_agent_name(agent_name)

    # Generate agent_id from agent_name if agent_name exists but agent_id doesn't
    # Note: This is a fallback; agent_id should be generated in span.py with project_id
    if agent_name and not agent_id:
        if not project_id:
            raise ValueError(
                "project_id is required for agent_id generation. "
                "This is an internal error - agent_id should be generated in span.py."
            )
        agent_id = generate_agent_id(agent_name, project_id)
        logger.info(f"[AGENT_ID] Auto-generated for trace: agent_id={agent_id}, agent_name={agent_name}, project_id={project_id}, agent_display_name={agent_display_name}")

    attributes = {
        AntsPlatformOtelSpanAttributes.TRACE_NAME: name,
        AntsPlatformOtelSpanAttributes.TRACE_USER_ID: user_id,
        AntsPlatformOtelSpanAttributes.TRACE_SESSION_ID: session_id,
        AntsPlatformOtelSpanAttributes.AGENT_NAME: agent_name,
        AntsPlatformOtelSpanAttributes.AGENT_ID: agent_id,
        AntsPlatformOtelSpanAttributes.AGENT_DISPLAY_NAME: agent_display_name,
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
    agent_name: Optional[str] = None,
    agent_display_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    observation_type: Optional[
        Union[ObservationTypeSpanLike, Literal["event"]]
    ] = "span",
) -> dict:
    # Validate and normalize agent_name
    agent_name = validate_and_normalize_agent_name(agent_name)

    # Note: agent_id should always be provided by span.py when agent_name is provided
    # This function does NOT generate agent_id - that happens in span.py with project_id
    # If agent_name exists but agent_id doesn't, it's an internal error that should be caught in span.py

    attributes = {
        AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL: level,
        AntsPlatformOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        AntsPlatformOtelSpanAttributes.VERSION: version,
        AntsPlatformOtelSpanAttributes.AGENT_NAME: agent_name,
        AntsPlatformOtelSpanAttributes.AGENT_ID: agent_id,
        AntsPlatformOtelSpanAttributes.AGENT_DISPLAY_NAME: agent_display_name,
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
    agent_name: Optional[str] = None,
    agent_display_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    model: Optional[str] = None,
    model_parameters: Optional[Dict[str, MapValue]] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    usage_details: Optional[Dict[str, int]] = None,
    cost_details: Optional[Dict[str, float]] = None,
    prompt: Optional[PromptClient] = None,
    observation_type: Optional[ObservationTypeGenerationLike] = "generation",
) -> dict:
    # Validate and normalize agent_name
    agent_name = validate_and_normalize_agent_name(agent_name)

    # Note: agent_id should always be provided by span.py when agent_name is provided
    # This function does NOT generate agent_id - that happens in span.py with project_id
    # If agent_name exists but agent_id doesn't, it's an internal error that should be caught in span.py

    attributes = {
        AntsPlatformOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        AntsPlatformOtelSpanAttributes.OBSERVATION_LEVEL: level,
        AntsPlatformOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        AntsPlatformOtelSpanAttributes.VERSION: version,
        AntsPlatformOtelSpanAttributes.AGENT_NAME: agent_name,
        AntsPlatformOtelSpanAttributes.AGENT_ID: agent_id,
        AntsPlatformOtelSpanAttributes.AGENT_DISPLAY_NAME: agent_display_name,
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
