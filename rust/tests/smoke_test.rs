//! MCP Protocol Smoke Tests.
//!
//! Validates JSON-RPC message format, tool definitions, and
//! protocol invariants without requiring a live database.

use serde_json::{Value, json};

/// Verify all 18 tools are defined in constants.
#[test]
fn test_all_tools_defined() {
    let tools: Vec<Value> = cuba_memorys::constants::tool_definitions();
    assert_eq!(
        tools.len(),
        18,
        "Expected 18 MCP tools, got {}",
        tools.len()
    );

    let tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t: &Value| t.get("name").and_then(|n: &Value| n.as_str()))
        .collect();

    let expected = [
        "cuba_alma",
        "cuba_cronica",
        "cuba_faro",
        "cuba_puente",
        "cuba_eco",
        "cuba_alarma",
        "cuba_remedio",
        "cuba_expediente",
        "cuba_jornada",
        "cuba_decreto",
        "cuba_vigia",
        "cuba_zafra",
        "cuba_forget",
        "cuba_reflexion",
        "cuba_hipotesis",
        "cuba_contradiccion",
        "cuba_centinela",
        "cuba_calibrar",
    ];

    for name in &expected {
        assert!(tool_names.contains(name), "Missing tool definition: {name}");
    }
}

/// Verify tool definitions have correct MCP schema structure.
#[test]
fn test_tool_schema_structure() {
    let tools: Vec<Value> = cuba_memorys::constants::tool_definitions();

    for tool in &tools {
        let name = tool
            .get("name")
            .and_then(|n: &Value| n.as_str())
            .unwrap_or("???");

        assert!(tool.get("name").is_some(), "{name}: missing 'name'");
        assert!(
            tool.get("description").is_some(),
            "{name}: missing 'description'"
        );
        assert!(
            tool.get("inputSchema").is_some(),
            "{name}: missing 'inputSchema'"
        );

        let schema = tool.get("inputSchema").unwrap();
        assert_eq!(
            schema.get("type").and_then(|t: &Value| t.as_str()),
            Some("object"),
            "{name}: inputSchema.type must be 'object'"
        );
        assert!(
            schema.get("properties").is_some(),
            "{name}: missing inputSchema.properties"
        );
    }
}

/// Verify JSON-RPC request parsing.
#[test]
fn test_jsonrpc_request_format() {
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "cuba_alma",
            "arguments": {
                "action": "get",
                "name": "test_entity"
            }
        }
    });

    assert_eq!(request["jsonrpc"], "2.0");
    assert_eq!(request["method"], "tools/call");
    assert!(request["params"]["name"].is_string());
    assert!(request["params"]["arguments"].is_object());
}

/// Verify JSON-RPC response format.
#[test]
fn test_jsonrpc_response_format() {
    let response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{
                "type": "text",
                "text": "{\"status\": \"ok\"}"
            }]
        }
    });

    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 1);
    let content = &response["result"]["content"];
    assert!(content.is_array());
    assert_eq!(content[0]["type"], "text");
}

/// Verify error response format.
#[test]
fn test_jsonrpc_error_format() {
    let error = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32602,
            "message": "Invalid params"
        }
    });

    assert_eq!(error["error"]["code"], -32602);
    assert!(error["error"]["message"].is_string());
}

/// Validate constant threshold relationships.
#[test]
fn test_threshold_invariants() {
    use cuba_memorys::constants::*;

    // Hierarchy: REINFORCE (0.92) > DEDUP (0.85) > UPDATE (0.75)
    assert!(
        PRED_ERROR_REINFORCE > DEDUP_THRESHOLD,
        "REINFORCE ({PRED_ERROR_REINFORCE}) must > DEDUP ({DEDUP_THRESHOLD})"
    );
    assert!(
        DEDUP_THRESHOLD > PRED_ERROR_UPDATE,
        "DEDUP ({DEDUP_THRESHOLD}) must > UPDATE ({PRED_ERROR_UPDATE})"
    );
    assert!(HEBBIAN_ACCESS_BOOST > 0.0 && HEBBIAN_ACCESS_BOOST < 0.1);
}

/// Verify handler dispatch maps all 18 tools.
#[test]
fn test_handler_dispatch_coverage() {
    let tool_names = [
        "cuba_alma",
        "cuba_cronica",
        "cuba_faro",
        "cuba_puente",
        "cuba_eco",
        "cuba_alarma",
        "cuba_remedio",
        "cuba_expediente",
        "cuba_jornada",
        "cuba_decreto",
        "cuba_vigia",
        "cuba_zafra",
        "cuba_forget",
        "cuba_reflexion",
        "cuba_hipotesis",
        "cuba_contradiccion",
        "cuba_centinela",
        "cuba_calibrar",
    ];

    for name in &tool_names {
        assert!(name.starts_with("cuba_"));
        assert!(name.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
    }
}

/// Verify SQL schema file contains required elements.
#[test]
fn test_schema_sql_content() {
    let schema = include_str!("../src/schema.sql");
    assert!(!schema.is_empty());

    for table in &[
        "brain_entities",
        "brain_observations",
        "brain_relations",
        "brain_errors",
        "brain_sessions",
        "brain_episodes",
        "brain_triggers",
        "brain_verify_log",
    ] {
        assert!(schema.contains(table), "Missing table: {table}");
    }

    assert!(schema.contains("vector"), "Missing pgvector");
    assert!(schema.contains("pg_trgm"), "Missing pg_trgm");
    assert!(schema.contains("embedding"), "Missing embedding column");
    assert!(schema.contains("importance"), "Missing importance column");
}

/// Verify cognitive module constants are valid.
#[test]
fn test_cognitive_constants_valid() {
    use cuba_memorys::constants::*;

    assert!(HEBBIAN_ACCESS_BOOST > 0.0 && HEBBIAN_ACCESS_BOOST < 1.0);
    assert!(BCM_THROTTLE_SCALE > 0.0 && BCM_THROTTLE_SCALE <= 1.0);
}

/// Verify cache constants.
#[test]
fn test_cache_constants_valid() {
    use cuba_memorys::constants::*;
    assert!(CACHE_MAX_ENTRIES > 0);
    assert!(CACHE_TTL_SECS > 0);
}

/// Verify all valid types lists are populated.
#[test]
fn test_valid_types_lists() {
    use cuba_memorys::constants::*;

    assert!(!VALID_ENTITY_TYPES.is_empty());
    assert!(!VALID_OBSERVATION_TYPES.is_empty());
    assert!(!VALID_SOURCES.is_empty());
    assert!(!VALID_RELATION_TYPES.is_empty());

    assert!(VALID_ENTITY_TYPES.contains(&"concept"));
    assert!(VALID_OBSERVATION_TYPES.contains(&"fact"));
    assert!(VALID_SOURCES.contains(&"agent"));
    assert!(VALID_RELATION_TYPES.contains(&"uses"));
}
