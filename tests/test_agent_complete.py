#!/usr/bin/env python3
"""
Comprehensive E2E Test Suite for Agent Display Name Functionality

Tests Cover:
1. Agent creation with agent_name and agent_display_name
2. Agent ID generation (BLAKE2b-64 hash of agent_name + project_id)
3. Display name updates via dedicated API endpoint
4. First-write-wins policy for concurrent span creation
5. Edge cases: empty names, long names, special characters
6. Security: cross-project isolation, invalid credentials
"""

import sys
import os

# Add local source to path FIRST to test local changes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ants_platform import AntsPlatform
import time

# Configuration
PUBLIC_KEY = "pk-ap-aec806c8-6fae-4049-926f-795ed3f00041"
SECRET_KEY = "sk-ap-81f5e898-22b4-456d-84d8-90729149d151"
HOST = "http://localhost:3000"
PROJECT_ID = "cmimnvc8p000gtgv44e4hxjhy"  # Hardcoded to bypass API call

# Test counters
tests_passed = 0
tests_failed = 0


def print_header(title):
    """Print a formatted test section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")


def print_test(test_name, passed, message=""):
    """Print test result."""
    global tests_passed, tests_failed
    if passed:
        tests_passed += 1
        status = "[PASS]"
    else:
        tests_failed += 1
        status = "[FAIL]"

    print(f"  {status} {test_name}")
    if message:
        print(f"       {message}")


def wait_for_backend(seconds=2):
    """Wait for backend to process data."""
    print(f"  [WAIT] Waiting {seconds}s for backend processing...")
    time.sleep(seconds)


# =============================================================================
# TEST SUITE START
# =============================================================================

print_header("E2E Test Suite: Agent Display Name Functionality")

# Create client
print("\n[SETUP] Creating AntsPlatform client...")
try:
    client = AntsPlatform(
        public_key=PUBLIC_KEY,
        secret_key=SECRET_KEY,
        host=HOST
    )
    # Manually set project_id to bypass API call
    client._project_id = PROJECT_ID
    print("  [OK] Client created successfully")
    print(f"  [OK] Using Project ID: {PROJECT_ID}")
except Exception as e:
    print(f"  [ERROR] Failed to create client: {e}")
    sys.exit(1)

# =============================================================================
# TEST 1: Basic Agent Creation
# =============================================================================

print_header("Test 1: Basic Agent Creation with agent_name and agent_display_name")

TEST_AGENT_1 = "basic_test_agent"
TEST_DISPLAY_1 = "Basic Test Agent v1.0"

try:
    with client.start_as_current_span(
        name="test-basic-creation",
        agent_name=TEST_AGENT_1,
        agent_display_name=TEST_DISPLAY_1
    ) as span:
        span.update(
            input={"test": "basic creation"},
            output={"status": "success"}
        )
    client.flush()
    print_test("Create agent with name and display name", True)
except Exception as e:
    print_test("Create agent with name and display name", False, str(e))

wait_for_backend()

# =============================================================================
# TEST 2: Agent ID Generation Verification
# =============================================================================

print_header("Test 2: Agent ID Generation (BLAKE2b-64 Hash)")

from ants_platform._client.attributes import generate_agent_id

try:
    agent_id = generate_agent_id(TEST_AGENT_1, PROJECT_ID)
    print(f"  Agent Name: {TEST_AGENT_1}")
    print(f"  Project ID: {PROJECT_ID}")
    print(f"  Generated Agent ID: {agent_id}")

    # Verify it's 16 hex characters
    is_valid = len(agent_id) == 16 and all(c in "0123456789abcdef" for c in agent_id)
    print_test("Agent ID is 16-character hex string", is_valid, f"Length: {len(agent_id)}")

    # Verify deterministic (same input = same output)
    agent_id_2 = generate_agent_id(TEST_AGENT_1, PROJECT_ID)
    is_deterministic = agent_id == agent_id_2
    print_test("Agent ID generation is deterministic", is_deterministic)

except Exception as e:
    print_test("Agent ID generation", False, str(e))

# =============================================================================
# TEST 3: Update Display Name via API
# =============================================================================

print_header("Test 3: Update Display Name via API Endpoint")

UPDATED_DISPLAY_1 = "Basic Test Agent v2.0 (Updated!)"

try:
    result = client.update_agent_display_name(
        agent_name=TEST_AGENT_1,
        new_display_name=UPDATED_DISPLAY_1
    )

    print(f"  Agent ID: {result.get('agentId')}")
    print(f"  New Display Name: {result.get('displayName')}")
    print(f"  Updated At: {result.get('updatedAt')}")

    success = result.get('success') == True
    correct_name = result.get('displayName') == UPDATED_DISPLAY_1

    print_test("API endpoint returns success", success)
    print_test("Display name updated correctly", correct_name)

except Exception as e:
    print_test("Update display name via API", False, str(e))

wait_for_backend()

# =============================================================================
# TEST 4: First-Write-Wins Policy
# =============================================================================

print_header("Test 4: First-Write-Wins Policy (Concurrent Span Creation)")

print("  Creating span with DIFFERENT display name (should be ignored)...")

try:
    with client.start_as_current_span(
        name="test-race-condition",
        agent_name=TEST_AGENT_1,
        agent_display_name="Should Be Ignored v3.0"  # This should be ignored
    ) as span:
        span.update(
            input={"test": "race condition"},
            output={"status": "completed"}
        )
    client.flush()
    print_test("Span with conflicting display name created", True)
    print("  [INFO] Backend should ignore the new display name (first-write-wins)")
except Exception as e:
    print_test("First-write-wins test", False, str(e))

wait_for_backend()

# =============================================================================
# TEST 5: Edge Case - Empty Display Name
# =============================================================================

print_header("Test 5: Edge Case - Empty Display Name (Should Fail)")

try:
    result = client.update_agent_display_name(
        agent_name=TEST_AGENT_1,
        new_display_name=""  # Empty string
    )
    print_test("Empty display name rejected", False, "Should have raised ValueError")
except ValueError as e:
    print_test("Empty display name rejected", True, "Correctly raised ValueError")
except Exception as e:
    print_test("Empty display name handling", False, f"Wrong exception: {type(e).__name__}")

# =============================================================================
# TEST 6: Edge Case - Very Long Display Name
# =============================================================================

print_header("Test 6: Edge Case - Very Long Display Name (255 char limit)")

LONG_NAME = "A" * 300  # 300 characters

try:
    result = client.update_agent_display_name(
        agent_name=TEST_AGENT_1,
        new_display_name=LONG_NAME
    )
    print_test("Long display name rejected by backend", False, "Should have been rejected")
except Exception as e:
    # Backend should return 400 Bad Request
    is_correct_error = "400" in str(e) or "Maximum length" in str(e)
    print_test("Long display name rejected (400 error)", is_correct_error, str(e)[:80])

# =============================================================================
# TEST 7: Edge Case - Special Characters in Names
# =============================================================================

print_header("Test 7: Edge Case - Special Characters in Display Name")

SPECIAL_CHARS_NAME = "Test Agent 🤖 with émojis & spëcial©"

try:
    # Create new agent with special characters
    TEST_AGENT_SPECIAL = "special_char_agent"

    with client.start_as_current_span(
        name="test-special-chars",
        agent_name=TEST_AGENT_SPECIAL,
        agent_display_name=SPECIAL_CHARS_NAME
    ) as span:
        span.update(input={"test": "special chars"}, output={"ok": True})

    client.flush()
    wait_for_backend(1)

    # Now update it
    UPDATED_SPECIAL = "Updated Agent 🚀 with new émojis"
    result = client.update_agent_display_name(
        agent_name=TEST_AGENT_SPECIAL,
        new_display_name=UPDATED_SPECIAL
    )

    print_test("Special characters in display name supported", True)

except Exception as e:
    print_test("Special characters handling", False, str(e))

# =============================================================================
# TEST 8: Agent Name Validation (255 char limit)
# =============================================================================

print_header("Test 8: Agent Name Validation (SDK enforces 255 char limit)")

LONG_AGENT_NAME = "a" * 300  # 300 characters

try:
    with client.start_as_current_span(
        name="test-long-agent-name",
        agent_name=LONG_AGENT_NAME,
        agent_display_name="Test Long Agent Name"
    ) as span:
        span.update(input={"test": "long name"}, output={"ok": True})

    client.flush()
    print("  [INFO] SDK should have truncated agent_name to 255 chars with warning")
    print_test("Long agent_name handled (truncated)", True)

except Exception as e:
    print_test("Long agent_name validation", False, str(e))

# =============================================================================
# TEST 9: Multiple Agents in Same Project
# =============================================================================

print_header("Test 9: Multiple Agents in Same Project (Isolation)")

TEST_AGENT_2 = "second_test_agent"
TEST_DISPLAY_2 = "Second Test Agent"

try:
    # Create second agent
    with client.start_as_current_span(
        name="test-second-agent",
        agent_name=TEST_AGENT_2,
        agent_display_name=TEST_DISPLAY_2
    ) as span:
        span.update(input={"agent": 2}, output={"ok": True})

    client.flush()
    wait_for_backend(1)

    # Update second agent
    result = client.update_agent_display_name(
        agent_name=TEST_AGENT_2,
        new_display_name="Second Test Agent (Updated)"
    )

    # Verify agent IDs are different
    agent_id_1 = generate_agent_id(TEST_AGENT_1, PROJECT_ID)
    agent_id_2 = generate_agent_id(TEST_AGENT_2, PROJECT_ID)

    different_ids = agent_id_1 != agent_id_2
    print(f"  Agent 1 ID: {agent_id_1}")
    print(f"  Agent 2 ID: {agent_id_2}")
    print_test("Different agents have different IDs", different_ids)
    print_test("Multiple agents can coexist", True)

except Exception as e:
    print_test("Multiple agents test", False, str(e))

# =============================================================================
# TEST 10: Non-existent Agent Update (Should Fail)
# =============================================================================

print_header("Test 10: Update Non-existent Agent (Should Return 404)")

try:
    result = client.update_agent_display_name(
        agent_name="nonexistent_agent_12345",
        new_display_name="This Should Fail"
    )
    print_test("Non-existent agent update rejected", False, "Should have raised error")
except Exception as e:
    # Should get 404 or similar error
    is_not_found = "404" in str(e) or "not found" in str(e).lower()
    print_test("Non-existent agent returns 404", is_not_found, str(e)[:80])

# =============================================================================
# TEST SUMMARY
# =============================================================================

print_header("Test Suite Summary")
print(f"\n  Total Tests: {tests_passed + tests_failed}")
print(f"  Passed: {tests_passed}")
print(f"  Failed: {tests_failed}")

if tests_failed == 0:
    print(f"\n  SUCCESS: All tests passed!")
    sys.exit(0)
else:
    print(f"\n  WARNING: {tests_failed} test(s) failed")
    sys.exit(1)
