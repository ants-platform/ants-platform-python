"""Auto-instrumentation bootstrap for CrewAI.

Loaded at Python startup via ants_platform_crewai.pth.
When ANTS_AUTO_INSTRUMENT is not set, the only cost is a single
os.environ.get() call — no other imports happen.

When enabled, registers a wrapt post-import hook that fires when
``import crewai`` happens anywhere in the process. At that point,
AntsPlatform and EventListener are initialized automatically.

Environment variables:
    ANTS_AUTO_INSTRUMENT: Set to "1" or "true" to enable auto-instrumentation.
    ANTS_AGENT_NAME: Override the agent name (default: auto-detected from crew class).
    ANTS_AGENT_DISPLAY_NAME: Override the display name shown in the UI.
    ANTS_PLATFORM_PUBLIC_KEY: API public key (required).
    ANTS_PLATFORM_SECRET_KEY: API secret key (required).
    ANTS_PLATFORM_HOST: API endpoint URL.
"""
import os as _os

_ENABLED = _os.environ.get("ANTS_AUTO_INSTRUMENT", "").lower() in ("1", "true")

if _ENABLED:
    import atexit as _atexit

    import wrapt as _wrapt

    _ants_platform_client = None
    _event_listener = None

    @_wrapt.when_imported("crewai")
    def _on_crewai_import(module):
        """Called once when ``import crewai`` happens anywhere."""
        global _ants_platform_client, _event_listener

        from ants_platform import AntsPlatform
        from ants_platform.crewai import EventListener

        _ants_platform_client = AntsPlatform(timeout=30)

        _event_listener = EventListener(
            public_key=_ants_platform_client._public_key,
            agent_name=_os.environ.get("ANTS_AGENT_NAME"),
            agent_display_name=_os.environ.get("ANTS_AGENT_DISPLAY_NAME"),
        )

        _atexit.register(_ants_platform_client.flush)
