#!/usr/bin/env python3
"""
E2E Test for all 25 cuba-memorys MCP tools.
Tests each tool against a real PostgreSQL database.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional
from pathlib import Path
import uuid

# Configuration — resolve binary path from env or auto-detect relative to this script
BINARY_PATH = os.environ.get(
    "CUBA_BINARY_PATH",
    str(Path(__file__).resolve().parent.parent / "target" / "release" / "cuba-memorys"),
)
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cuba:memorys2026@127.0.0.1:5488/brain"
)
ENV = {
    "DATABASE_URL": DATABASE_URL,
    "PATH": "/usr/bin:/bin",
}

# Test tracking
tests_run = 0
tests_passed = 0
tests_failed = 0
failed_tests = []


def invoke_mcp(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Invoke the MCP server with a JSON-RPC request.
    Returns the parsed response or None on error.
    """
    payload = json.dumps(request)
    try:
        # Set up environment with all existing env vars plus our overrides
        env = os.environ.copy()
        env.update(ENV)

        result = subprocess.run(
            [BINARY_PATH],
            input=payload,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        if result.returncode != 0:
            stderr_preview = result.stderr[:300] if result.stderr else "(empty)"
            print(
                f"  [ERROR] MCP invocation failed (exit={result.returncode}): {stderr_preview}"
            )
            return None

        if not result.stdout:
            print(f"  [ERROR] Empty stdout from MCP server")
            return None

        response = json.loads(result.stdout)
        return response
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Invalid JSON response: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] MCP invocation timed out")
        return None
    except Exception as e:
        print(f"  [ERROR] MCP invocation error: {e}")
        return None


def extract_tool_result(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract the tool result from the MCP response.
    The actual content is in response["result"]["content"][0]["text"]
    """
    try:
        if "error" in response:
            error = response["error"]
            print(f"  [ERROR] RPC Error: {error.get('message', 'Unknown error')}")
            return None

        if "result" not in response:
            print(f"  [ERROR] No result in response: {response}")
            return None

        result = response["result"]
        if isinstance(result, dict) and "content" in result:
            content_list = result["content"]
            if isinstance(content_list, list) and len(content_list) > 0:
                return content_list[0].get("text", "")

        print(f"  [ERROR] Invalid result structure: {str(result)[:200]}")
        return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"  [ERROR] Failed to extract result: {e}")
        return None


def parse_tool_result(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse the tool result JSON string.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse tool result JSON: {e}")
        return None


def test(
    tool_name: str, action: str, request: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Execute a single test and record result.
    Returns the parsed tool result or None on failure.
    """
    global tests_run, tests_passed, tests_failed, failed_tests

    tests_run += 1
    test_id = f"{tool_name}/{action}"

    print(f"\n[TEST {tests_run}] {test_id}")

    # Set up the request in MCP tools/call format
    full_request = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": request,
        },
    }

    response = invoke_mcp(full_request)
    if response is None:
        print(f"  FAIL: MCP invocation failed")
        tests_failed += 1
        failed_tests.append(test_id)
        return None

    content = extract_tool_result(response)
    if content is None:
        print(f"  FAIL: Could not extract result")
        tests_failed += 1
        failed_tests.append(test_id)
        return None

    result = parse_tool_result(content)
    if result is None:
        print(f"  FAIL: Could not parse tool result")
        tests_failed += 1
        failed_tests.append(test_id)
        return None

    if "error" in result:
        print(f"  FAIL: Tool returned error: {result['error']}")
        tests_failed += 1
        failed_tests.append(test_id)
        return None

    print(f"  PASS: {test_id}")
    tests_passed += 1
    return result


def test_cuba_alma():
    """Test cuba_alma: create, get, update, delete entity."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_alma")
    print("=" * 60)

    entity_name = "_e2e_test"

    # Create
    result = test(
        "cuba_alma",
        "create",
        {
            "action": "create",
            "name": entity_name,
            "entity_type": "concept",
        },
    )
    if result is None:
        return False

    # Get
    result = test(
        "cuba_alma",
        "get",
        {
            "action": "get",
            "name": entity_name,
        },
    )
    if result is None:
        return False

    # Update
    new_name = "_e2e_test_renamed"
    result = test(
        "cuba_alma",
        "update",
        {
            "action": "update",
            "name": entity_name,
            "new_name": new_name,
        },
    )
    if result is None:
        return False

    # Get updated
    result = test(
        "cuba_alma",
        "get_renamed",
        {
            "action": "get",
            "name": new_name,
        },
    )
    if result is None:
        return False

    # Delete
    result = test(
        "cuba_alma",
        "delete",
        {
            "action": "delete",
            "name": new_name,
        },
    )
    if result is None:
        return False

    return True


def test_cuba_cronica():
    """Test cuba_cronica: add, list, batch_add, delete observations."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_cronica")
    print("=" * 60)

    entity_name = "_e2e_cronica"

    # Create entity first (via alma for setup)
    test(
        "cuba_alma",
        "create_cronica_entity",
        {
            "action": "create",
            "name": entity_name,
            "entity_type": "concept",
        },
    )

    # Add observation
    result = test(
        "cuba_cronica",
        "add",
        {
            "action": "add",
            "entity_name": entity_name,
            "observation_type": "fact",
            "content": "First observation for E2E",
            "source": "agent",
        },
    )
    if result is None:
        return False

    # Add another
    result = test(
        "cuba_cronica",
        "add_second",
        {
            "action": "add",
            "entity_name": entity_name,
            "observation_type": "lesson",
            "content": "Second observation for E2E",
            "source": "user",
        },
    )
    if result is None:
        return False

    # List observations
    result = test(
        "cuba_cronica",
        "list",
        {
            "action": "list",
            "entity_name": entity_name,
        },
    )
    if result is None:
        return False

    # Batch add 3 more (each obs needs its own entity_name)
    result = test(
        "cuba_cronica",
        "batch_add",
        {
            "action": "batch_add",
            "entity_name": entity_name,
            "observations": [
                {
                    "entity_name": entity_name,
                    "observation_type": "fact",
                    "content": "Batch observation about PostgreSQL performance tuning",
                    "source": "agent",
                },
                {
                    "entity_name": entity_name,
                    "observation_type": "decision",
                    "content": "Decided to use exponential decay instead of FSRS-6 model",
                    "source": "user",
                },
                {
                    "entity_name": entity_name,
                    "observation_type": "lesson",
                    "content": "Lesson learned about Rust borrow checker with mutable references",
                    "source": "error_detection",
                },
            ],
        },
    )
    if result is None:
        return False

    # List again to see all
    result = test(
        "cuba_cronica",
        "list_all",
        {
            "action": "list",
            "entity_name": entity_name,
        },
    )
    if result is None:
        return False

    # Extract an observation_id to delete (from the previous list result)
    if result and isinstance(result, dict):
        observations = result.get("observations", [])
        if observations and len(observations) > 0:
            obs_id = observations[0].get("id")
            if obs_id:
                # Delete one
                test(
                    "cuba_cronica",
                    "delete",
                    {
                        "action": "delete",
                        "entity_name": entity_name,
                        "observation_id": obs_id,
                    },
                )

    # List final
    result = test(
        "cuba_cronica",
        "list_final",
        {
            "action": "list",
            "entity_name": entity_name,
        },
    )
    if result is None:
        return False

    return True


def test_cuba_faro():
    """Test cuba_faro: search with hybrid and verify modes."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_faro")
    print("=" * 60)

    # Search (hybrid mode, default)
    result = test(
        "cuba_faro",
        "search_hybrid",
        {
            "query": "_e2e_cronica",
            "mode": "hybrid",
            "limit": 5,
        },
    )
    if result is None:
        return False

    # Verify mode
    result = test(
        "cuba_faro",
        "search_verify",
        {
            "query": "_e2e_cronica",
            "mode": "verify",
        },
    )
    if result is None:
        return False

    return True


def test_cuba_puente():
    """Test cuba_puente: create relations, traverse, infer."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_puente")
    print("=" * 60)

    entity_a = "_e2e_A"
    entity_b = "_e2e_B"

    # Create entities
    test(
        "cuba_alma",
        "create_A",
        {
            "action": "create",
            "name": entity_a,
            "entity_type": "concept",
        },
    )
    test(
        "cuba_alma",
        "create_B",
        {
            "action": "create",
            "name": entity_b,
            "entity_type": "concept",
        },
    )

    # Create relation
    result = test(
        "cuba_puente",
        "create_relation",
        {
            "action": "create",
            "from_entity": entity_a,
            "to_entity": entity_b,
            "relation_type": "uses",
            "bidirectional": False,
        },
    )
    if result is None:
        return False

    # Traverse from A
    result = test(
        "cuba_puente",
        "traverse",
        {
            "action": "traverse",
            "start_entity": entity_a,
            "max_depth": 2,
        },
    )
    if result is None:
        return False

    # Infer from A
    result = test(
        "cuba_puente",
        "infer",
        {
            "action": "infer",
            "start_entity": entity_a,
            "max_depth": 2,
        },
    )
    if result is None:
        return False

    # Delete relation
    result = test(
        "cuba_puente",
        "delete_relation",
        {
            "action": "delete",
            "from_entity": entity_a,
            "to_entity": entity_b,
        },
    )
    if result is None:
        return False

    return True


def test_cuba_eco():
    """Test cuba_eco: positive, negative, correct feedback."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_eco")
    print("=" * 60)

    entity_name = "_e2e_cronica"

    # Get an observation_id first by listing
    response = invoke_mcp(
        {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "cuba_cronica",
                "arguments": {
                    "action": "list",
                    "entity_name": entity_name,
                },
            },
        }
    )

    content = extract_tool_result(response)
    result = parse_tool_result(content) if content else None

    obs_id = None
    if result and isinstance(result, dict):
        observations = result.get("observations", [])
        if observations:
            obs_id = observations[0].get("id")

    if not obs_id:
        print(f"  SKIP: Could not get observation_id for feedback tests")
        return True

    # Positive feedback
    result = test(
        "cuba_eco",
        "positive",
        {
            "action": "positive",
            "entity_name": entity_name,
            "observation_id": obs_id,
        },
    )
    if result is None:
        return False

    # Negative feedback
    result = test(
        "cuba_eco",
        "negative",
        {
            "action": "negative",
            "entity_name": entity_name,
            "observation_id": obs_id,
        },
    )
    if result is None:
        return False

    # Correct feedback
    result = test(
        "cuba_eco",
        "correct",
        {
            "action": "correct",
            "entity_name": entity_name,
            "observation_id": obs_id,
            "correction": "Corrected observation content",
        },
    )
    if result is None:
        return False

    return True


def test_cuba_alarma():
    """Test cuba_alarma: report error."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_alarma")
    print("=" * 60)

    result = test(
        "cuba_alarma",
        "report",
        {
            "error_type": "E2ETestError",
            "error_message": "This is a test error from E2E suite",
            "project": "default",
            "context": {
                "file": "e2e_all_tools.py",
                "function": "test_cuba_alarma",
                "line": 42,
            },
        },
    )
    if result is None:
        return False

    return True


def test_cuba_remedio():
    """Test cuba_remedio: resolve error."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_remedio")
    print("=" * 60)

    # Get an error_id from the alarma test via expediente search
    response = invoke_mcp(
        {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "cuba_expediente",
                "arguments": {
                    "query": "E2ETestError",
                },
            },
        }
    )

    content = extract_tool_result(response)
    result = parse_tool_result(content) if content else None

    error_id = None
    if result and isinstance(result, dict):
        results_list = result.get("results", [])
        if results_list:
            error_id = results_list[0].get("id")

    if not error_id:
        print(f"  SKIP: Could not get error_id from expediente")
        return True

    # Resolve it
    result = test(
        "cuba_remedio",
        "resolve",
        {
            "error_id": error_id,
            "solution": "This was a test error that has been resolved with this solution.",
        },
    )
    if result is None:
        return False

    return True


def test_cuba_expediente():
    """Test cuba_expediente: search errors, test anti-repetition."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_expediente")
    print("=" * 60)

    # Search for errors
    result = test(
        "cuba_expediente",
        "search",
        {
            "query": "E2ETestError",
        },
    )
    if result is None:
        return False

    # Test proposed_action anti-repetition
    result = test(
        "cuba_expediente",
        "proposed_action",
        {
            "query": "E2ETestError",
            "proposed_action": "Try to replicate the E2E test error again",
        },
    )
    if result is None:
        return False

    # resolved_only
    result = test(
        "cuba_expediente",
        "resolved_only",
        {
            "query": "E2ETestError",
            "resolved_only": True,
        },
    )
    if result is None:
        return False

    return True


def test_cuba_jornada():
    """Test cuba_jornada: start, get current, list, end session."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_jornada")
    print("=" * 60)

    session_name = f"e2e_session_{uuid.uuid4().hex[:8]}"

    # Start session
    result = test(
        "cuba_jornada",
        "start",
        {
            "action": "start",
            "name": session_name,
            "goals": ["Test all MCP tools", "Verify end-to-end functionality"],
        },
    )
    if result is None:
        return False

    # Get current
    result = test(
        "cuba_jornada",
        "current",
        {
            "action": "current",
        },
    )
    if result is None:
        return False

    # List sessions
    result = test(
        "cuba_jornada",
        "list",
        {
            "action": "list",
        },
    )
    if result is None:
        return False

    # End session
    result = test(
        "cuba_jornada",
        "end",
        {
            "action": "end",
            "outcome": "success",
            "summary": "All E2E tests completed successfully",
        },
    )
    if result is None:
        return False

    return True


def test_cuba_decreto():
    """Test cuba_decreto: record, query, list decisions."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_decreto")
    print("=" * 60)

    # Record a decision
    result = test(
        "cuba_decreto",
        "record",
        {
            "action": "record",
            "title": "E2E Test Decision",
            "context": "Testing cuba_decreto functionality",
            "alternatives": ["Option A", "Option B", "Option C"],
            "chosen": "Option B",
            "rationale": "Option B was most appropriate for E2E testing",
        },
    )
    if result is None:
        return False

    # Query for it
    result = test(
        "cuba_decreto",
        "query",
        {
            "action": "query",
            "query": "E2E Test Decision",
        },
    )
    if result is None:
        return False

    # List all
    result = test(
        "cuba_decreto",
        "list",
        {
            "action": "list",
        },
    )
    if result is None:
        return False

    return True


def test_cuba_vigia():
    """Test cuba_vigia: summary, health, drift, communities, bridges."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_vigia")
    print("=" * 60)

    metrics = ["summary", "health", "drift", "communities", "bridges"]

    for metric in metrics:
        result = test(
            "cuba_vigia",
            metric,
            {
                "metric": metric,
            },
        )
        if result is None:
            return False

    return True


def test_cuba_zafra():
    """Test cuba_zafra: stats, decay, pagerank, find_duplicates, export."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_zafra")
    print("=" * 60)

    # Stats
    result = test(
        "cuba_zafra",
        "stats",
        {
            "action": "stats",
        },
    )
    if result is None:
        return False

    # Decay (non-destructive)
    result = test(
        "cuba_zafra",
        "decay",
        {
            "action": "decay",
        },
    )
    if result is None:
        return False

    # PageRank
    result = test(
        "cuba_zafra",
        "pagerank",
        {
            "action": "pagerank",
        },
    )
    if result is None:
        return False

    # Find duplicates
    result = test(
        "cuba_zafra",
        "find_duplicates",
        {
            "action": "find_duplicates",
        },
    )
    if result is None:
        return False

    # Export
    result = test(
        "cuba_zafra",
        "export",
        {
            "action": "export",
        },
    )
    if result is None:
        return False

    return True


def test_cuba_forget():
    """Test cuba_forget: cascading deletion."""
    print("\n" + "=" * 60)
    print("TESTING: cuba_forget")
    print("=" * 60)

    entity_name = "_e2e_forget_me"

    # Create entity
    test(
        "cuba_alma",
        "create_forget",
        {
            "action": "create",
            "name": entity_name,
            "entity_type": "concept",
        },
    )

    # Add observation
    test(
        "cuba_cronica",
        "add_forget",
        {
            "action": "add",
            "entity_name": entity_name,
            "observation_type": "fact",
            "content": "Observation to be forgotten",
            "source": "agent",
        },
    )

    # Create relation with another entity (for relation cascading test)
    helper = "_e2e_forget_helper"
    test(
        "cuba_alma",
        "create_helper",
        {
            "action": "create",
            "name": helper,
            "entity_type": "concept",
        },
    )
    test(
        "cuba_puente",
        "create_forget_relation",
        {
            "action": "create",
            "from_entity": entity_name,
            "to_entity": helper,
            "relation_type": "related_to",
        },
    )

    # Now forget the main entity
    result = test(
        "cuba_forget",
        "forget",
        {
            "action": "forget",
            "entity_name": entity_name,
            "confirm": True,
        },
    )
    if result is None:
        return False

    return True


def test_v08_v09_tools():
    """Smoke-test v0.8–v0.10 tools (read-mostly + light writes)."""
    print("\n" + "=" * 60)
    print("TESTING: v0.8–v0.10 tools (12)")
    print("=" * 60)

    test("cuba_reflexion", "analyze", {"action": "analyze"})
    test("cuba_hipotesis", "explain", {"action": "explain", "effect": "_e2e_cronica"})
    test("cuba_contradiccion", "scan", {"action": "scan"})
    test(
        "cuba_centinela",
        "create",
        {
            "action": "create",
            "entity_pattern": "_e2e_cronica",
            "condition_type": "on_access",
            "message": "E2E centinela trigger",
        },
    )
    test("cuba_centinela", "list", {"action": "list"})
    test("cuba_calibrar", "stats", {"action": "stats"})
    test("cuba_calibrar", "metrics", {"action": "metrics"})
    test(
        "cuba_ingesta",
        "parse",
        {
            "action": "parse",
            "entity_name": "_e2e_cronica",
            "text": "E2E ingesta paragraph one.\n\nSecond paragraph for parse split.",
        },
    )
    test("cuba_proyecto", "list", {"action": "list"})
    test("cuba_proyecto", "current", {"action": "current"})
    test("cuba_pre_compact", "restore", {"action": "restore"})
    test("cuba_sync", "status", {"action": "status"})
    test(
        "cuba_juez",
        "scan_entity",
        {"action": "scan_entity", "entity_name": "_e2e_cronica", "max_pairs": 1},
    )
    test(
        "cuba_pizarra",
        "write",
        {
            "action": "write",
            "content": "E2E working memory note",
            "tag": "e2e",
            "ttl_seconds": 120,
        },
    )
    test("cuba_pizarra", "read", {"action": "read", "tag": "e2e"})
    test(
        "cuba_archivo",
        "append",
        {
            "action": "append",
            "event_action": "e2e_test",
            "payload": {"tool": "cuba_archivo"},
        },
    )
    test("cuba_archivo", "verify", {"action": "verify", "limit": 5})
    test("cuba_archivo", "tail", {"action": "tail", "limit": 3})


def cleanup():
    """Clean up remaining test entities."""
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    entities_to_cleanup = [
        "_e2e_cronica",
        "_e2e_A",
        "_e2e_B",
        "_e2e_forget_helper",
    ]

    for entity in entities_to_cleanup:
        # Try to delete each entity
        test(
            "cuba_alma",
            f"cleanup_{entity}",
            {
                "action": "delete",
                "name": entity,
            },
        )


def main():
    """Run all E2E tests."""
    global tests_run, tests_passed, tests_failed

    print("\n")
    print("=" * 60)
    print("CUBA-MEMORYS E2E TEST SUITE - ALL 25 TOOLS")
    print("=" * 60)

    # Run all tool tests
    test_cuba_alma()
    test_cuba_cronica()
    test_cuba_faro()
    test_cuba_puente()
    test_cuba_eco()
    test_cuba_alarma()
    test_cuba_remedio()
    test_cuba_expediente()
    test_cuba_jornada()
    test_cuba_decreto()
    test_cuba_vigia()
    test_cuba_zafra()
    test_cuba_forget()
    test_v08_v09_tools()

    # Cleanup
    cleanup()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run:     {tests_run}")
    print(f"Passed:              {tests_passed}")
    print(f"Failed:              {tests_failed}")

    if failed_tests:
        print(f"\nFailed Tests:")
        for test_id in failed_tests:
            print(f"  - {test_id}")

    print("\n" + "=" * 60)

    # Exit with appropriate code
    sys.exit(0 if tests_failed == 0 else 1)


if __name__ == "__main__":
    main()
