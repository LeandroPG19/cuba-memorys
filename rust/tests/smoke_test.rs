use serde_json::Value;

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
fn the_embedded_migrations_build_every_object_the_handlers_query() {
    let migrator = sqlx::migrate!("./migrations");
    let up: String = migrator
        .migrations
        .iter()
        .filter(|m| m.migration_type.is_up_migration())
        .map(|m| m.sql.as_ref())
        .collect::<Vec<&str>>()
        .join("\n");

    assert!(
        !up.is_empty(),
        "sqlx::migrate! resolved zero up-migrations, so db.rs would hand the handlers an \
         empty database and every query would fail at runtime"
    );

    for object in &[
        "brain_entities",
        "brain_observations",
        "brain_relations",
        "brain_errors",
        "brain_sessions",
        "brain_episodes",
        "brain_triggers",
        "brain_verify_log",
        "brain_audit_log",
        "brain_procedures",
        "brain_wm",
        "embedding_model",
        "session_id",
        "importance",
        "tags TEXT[]",
        "idx_obs_high_importance",
        "CREATE EXTENSION IF NOT EXISTS vector",
        "pg_trgm",
    ] {
        assert!(
            up.contains(object),
            "«{object}» is queried by the handlers but no migration creates it. This test \
             reads the set sqlx::migrate! embeds — the same one db.rs applies — precisely \
             because src/schema.sql froze at v0.6.0 and is 20 tables behind"
        );
    }
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

#[test]
fn a_request_without_an_id_parses_as_a_notification() {
    use cuba_memorys::protocol::JsonRpcRequest;

    let notification: JsonRpcRequest = serde_json::from_str(
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    )
    .expect("a notification is a valid envelope: params is #[serde(default)] and id is optional");
    assert!(
        notification.id.is_none(),
        "an absent id is the only thing separating a notification from a call. Parsed with \
         one, the server would answer something nobody is waiting for and the client would \
         see an orphan response"
    );
    assert_eq!(notification.method, "notifications/initialized");

    let call: JsonRpcRequest =
        serde_json::from_str(r#"{"jsonrpc":"2.0","id":7,"method":"tools/list"}"#)
            .expect("a call with no params is valid");
    assert_eq!(call.id, Some(serde_json::json!(7)));

    assert!(
        serde_json::from_str::<JsonRpcRequest>(r#"{"jsonrpc":"2.0","id":1}"#).is_err(),
        "with no method there is nothing to dispatch: it has to die in the parser and come \
         back as -32600, not reach a handler"
    );
}
