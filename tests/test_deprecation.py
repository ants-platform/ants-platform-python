"""Tests for deprecation warnings on deprecated functions."""

import warnings
import pytest
from unittest.mock import patch

from ants_platform import AntsPlatform


class TestDeprecationWarnings:
    """Test that deprecated functions emit proper deprecation warnings."""

    # List of deprecated functions and their expected warning messages. Target is the object they are called on.
    DEPRECATED_FUNCTIONS = [
        # on the client:
        {
            "method": "start_generation",
            "target": "client",
            "kwargs": {"name": "test_generation"},
            "expected_message": "start_generation is deprecated and will be removed in a future version. Use start_observation(as_type='generation') instead.",
        },
        {
            "method": "start_as_current_generation",
            "target": "client",
            "kwargs": {"name": "test_generation"},
            "expected_message": "start_as_current_generation is deprecated and will be removed in a future version. Use start_as_current_observation(as_type='generation') instead.",
        },
        # on the span:
        {
            "method": "start_generation",
            "target": "span",
            "kwargs": {"name": "test_generation"},
            "expected_message": "start_generation is deprecated and will be removed in a future version. Use start_observation(as_type='generation') instead.",
        },
        {
            "method": "start_as_current_generation",
            "target": "span",
            "kwargs": {"name": "test_generation"},
            "expected_message": "start_as_current_generation is deprecated and will be removed in a future version. Use start_as_current_observation(as_type='generation') instead.",
        },
        {
            "method": "start_as_current_span",
            "target": "span",
            "kwargs": {"name": "test_span"},
            "expected_message": "start_as_current_span is deprecated and will be removed in a future version. Use start_as_current_observation(as_type='span') instead.",
        },
    ]

    @pytest.fixture
    def ants_platform_client(self):
        """Create a AntsPlatform client for testing."""
        with patch.dict(
            "os.environ",
            {
                "ANTS_PLATFORM_PUBLIC_KEY": "test_key",
                "ANTS_PLATFORM_SECRET_KEY": "test_secret",
                "ANTS_PLATFORM_HOST": "http://localhost:3000",
            },
        ):
            return AntsPlatform()

    @pytest.mark.parametrize("func_info", DEPRECATED_FUNCTIONS)
    def test_deprecated_function_warnings(self, ants_platform_client, func_info):
        """Test that deprecated functions emit proper deprecation warnings."""
        method_name = func_info["method"]
        target = func_info["target"]
        kwargs = func_info["kwargs"]
        expected_message = func_info["expected_message"]

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            try:
                if target == "client":
                    # Test deprecated methods on the client
                    method = getattr(ants_platform_client, method_name)
                    if "current" in method_name:
                        # Context manager methods
                        with method(**kwargs) as obj:
                            if hasattr(obj, "end"):
                                obj.end()
                    else:
                        # Regular methods
                        obj = method(**kwargs)
                        if hasattr(obj, "end"):
                            obj.end()

                elif target == "span":
                    # Test deprecated methods on spans
                    span = ants_platform_client.start_span(name="test_parent")
                    method = getattr(span, method_name)
                    if "current" in method_name:
                        # Context manager methods
                        with method(**kwargs) as obj:
                            if hasattr(obj, "end"):
                                obj.end()
                    else:
                        # Regular methods
                        obj = method(**kwargs)
                        if hasattr(obj, "end"):
                            obj.end()
                    span.end()

            except Exception:
                pass

            # Check that a deprecation warning was emitted
            deprecation_warnings = [
                w for w in warning_list if issubclass(w.category, DeprecationWarning)
            ]
            assert (
                len(deprecation_warnings) > 0
            ), f"No DeprecationWarning emitted for {target}.{method_name}"

            # Check that the warning message matches expected
            warning_messages = [str(w.message) for w in deprecation_warnings]
            assert (
                expected_message in warning_messages
            ), f"Expected warning message not found for {target}.{method_name}. Got: {warning_messages}"
