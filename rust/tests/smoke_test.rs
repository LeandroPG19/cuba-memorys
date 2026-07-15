use serde_json::{Value, json};

#[test]
fn test_all_tools_defined() {
    let tools = cuba_memorys::constants::tool_definitions();

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
        "cuba_ingesta",
        "cuba_proyecto",
        "cuba_pre_compact",
        "cuba_sync",
        "cuba_juez",
        "cuba_pizarra",
        "cuba_archivo",
        "cuba_receta",
    ];

    for name in &expected {
        assert!(tool_names.contains(name), "Missing tool definition: {name}");
    }

    let mut seen = std::collections::HashSet::new();
    for name in &tool_names {
        assert!(seen.insert(*name), "duplicate tool definition: {name}");
    }
}

#[test]
fn test_tool_schema_structure() {
    let tools = cuba_memorys::constants::tool_definitions();

    for tool in tools.iter() {
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

#[test]
fn test_threshold_invariants() {
    use cuba_memorys::constants::*;

    const _: () = assert!(PRED_ERROR_REINFORCE > DEDUP_THRESHOLD);
    const _: () = assert!(DEDUP_THRESHOLD > PRED_ERROR_UPDATE);
    const _: () = assert!(HEBBIAN_ACCESS_BOOST > 0.0 && HEBBIAN_ACCESS_BOOST < 0.1);
}

#[test]
fn advertised_tools_are_all_dispatchable() {
    let tools = cuba_memorys::constants::tool_definitions();

    for tool in tools.iter() {
        let name = tool
            .get("name")
            .and_then(|n: &Value| n.as_str())
            .expect("every tool definition has a name");

        assert!(
            cuba_memorys::handlers::is_known_tool(name),
            "{name} se anuncia en tools/list pero el dispatcher no la conoce: \
             un cliente que la llame recibiría 'unknown tool' en runtime"
        );

        assert!(name.starts_with("cuba_"), "{name}: prefijo inesperado");
        assert!(
            name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "{name}: los nombres de tool son snake_case en minúsculas"
        );
    }
}

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
    assert!(
        schema.contains("embedding_model"),
        "Missing embedding_model column"
    );
    assert!(schema.contains("session_id"), "Missing session_id column");
    assert!(schema.contains("tags TEXT[]"), "Missing tags column");
    assert!(
        schema.contains("idx_obs_high_importance"),
        "Missing partial importance index"
    );
    assert!(
        schema.contains("tags TEXT[]"),
        "Missing tags column definition"
    );
}

#[test]
fn test_cognitive_constants_valid() {
    use cuba_memorys::constants::*;

    const _: () = assert!(HEBBIAN_ACCESS_BOOST > 0.0 && HEBBIAN_ACCESS_BOOST < 1.0);
    const _: () = assert!(BCM_THROTTLE_SCALE > 0.0 && BCM_THROTTLE_SCALE <= 1.0);
}

#[test]
fn test_cache_constants_valid() {
    use cuba_memorys::constants::*;
    const _: () = assert!(CACHE_MAX_ENTRIES > 0);
    const _: () = assert!(CACHE_TTL_SECS > 0);
}

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

#[test]
fn test_importance_priors() {
    use cuba_memorys::constants::importance_prior;

    assert!((importance_prior("decision", 0.5) - 0.8).abs() < f64::EPSILON);
    assert!((importance_prior("lesson", 0.5) - 0.75).abs() < f64::EPSILON);
    assert!((importance_prior("error", 0.5) - 0.7).abs() < f64::EPSILON);
    assert!((importance_prior("solution", 0.5) - 0.7).abs() < f64::EPSILON);

    let fact_high = importance_prior("fact", 1.0);
    let fact_low = importance_prior("fact", 0.2);
    assert!(
        fact_high > fact_low,
        "Higher density should yield higher importance"
    );
    assert!(fact_high <= 0.9, "fact importance capped at 0.9");

    let ctx = importance_prior("context", 1.0);
    assert!(ctx <= 0.7, "context importance capped at 0.7");

    assert!(importance_prior("fact", 0.0) >= 0.1);
    assert!(importance_prior("context", 0.0) >= 0.1);
}

#[test]
fn model_tag_follows_the_environment() {
    unsafe { std::env::remove_var("CUBA_EMBED_MODEL") };
    assert_eq!(
        cuba_memorys::embeddings::onnx::current_model(),
        "multilingual-e5-small",
        "with nothing set, the default stands"
    );

    unsafe { std::env::set_var("CUBA_EMBED_MODEL", "bge-m3") };
    assert_eq!(
        cuba_memorys::embeddings::onnx::current_model(),
        "bge-m3",
        "CUBA_EMBED_MODEL must win — it is the only thing that knows which model ran"
    );

    unsafe { std::env::remove_var("CUBA_EMBED_MODEL") };
}
