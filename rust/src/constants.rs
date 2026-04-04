//! Constants, tool definitions, and threshold configuration.
//!
//! Mirrors Python constants.py — tool definitions for MCP tools/list.

use serde_json::Value;

// ── Thresholds ───────────────────────────────────────────────────

/// Deduplication similarity threshold (cosine similarity).
pub const DEDUP_THRESHOLD: f64 = 0.85;

/// Prediction Error Gating thresholds (V5 — Vestige-inspired).
pub const PRED_ERROR_REINFORCE: f64 = 0.92; // Very similar → reinforce existing
pub const PRED_ERROR_UPDATE: f64 = 0.75; // Somewhat similar → update existing
                                         // Below PRED_ERROR_UPDATE → create new observation

/// Cache configuration.
/// V3: TTL raised 60→300s to prevent thrashing during long tool executions.
/// Configurable via CUBA_CACHE_TTL env var.
pub const CACHE_MAX_ENTRIES: usize = 256;
pub const CACHE_TTL_SECS: u64 = 300;

/// Hebbian boost constants.
pub const HEBBIAN_ACCESS_BOOST: f64 = 0.01;

/// BCM Metaplasticity (Bienenstock-Cooper-Munro, 1982).
/// Throttle scale: how aggressively to reduce boost (0.0=off, 1.0=max throttle).
pub const BCM_THROTTLE_SCALE: f64 = 0.8;

/// Relation types.
pub const VALID_RELATION_TYPES: &[&str] =
    &["uses", "causes", "implements", "depends_on", "related_to"];

/// Entity types.
pub const VALID_ENTITY_TYPES: &[&str] = &[
    "concept",
    "project",
    "technology",
    "person",
    "pattern",
    "config",
];

/// Observation types.
pub const VALID_OBSERVATION_TYPES: &[&str] = &[
    "fact",
    "decision",
    "lesson",
    "preference",
    "error",
    "solution",
    "context",
    "tool_usage",
    "superseded",
];

/// Observation sources.
pub const VALID_SOURCES: &[&str] = &[
    "agent",
    "error_detection",
    "user",
    "consolidation",
    "inference",
];

// ── Importance Priors ────────────────────────────────────────────

/// Importance priors by observation type.
///
/// Decisions and lessons inherently more valuable than transient context.
/// Used by cronica::add to set initial importance based on both the
/// observation type and the information density of the content.
pub fn importance_prior(obs_type: &str, density: f64) -> f64 {
    match obs_type {
        "decision" => 0.8,
        "lesson" => 0.75,
        "error" | "solution" => 0.7,
        "fact" | "preference" => (density * 0.6).clamp(0.1, 0.9),
        "context" | "tool_usage" => (density * 0.4).clamp(0.1, 0.7),
        _ => density.clamp(0.1, 0.8),
    }
}

// ── Tool Definitions ─────────────────────────────────────────────

/// Generate MCP tool definitions for tools/list response.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        tool_def(
            "cuba_alma",
            "CRUD knowledge graph entities (concepts, projects, technologies, patterns, people). Auto-boosts neighbors on access. For transient info use cuba_cronica instead.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "update", "delete", "get"], "description": "Operation to perform"},
                    "name": {"type": "string", "description": "Entity name (unique identifier)"},
                    "entity_type": {"type": "string", "description": "Type: concept, project, technology, person, pattern, config"},
                    "new_name": {"type": "string", "description": "New name for update action"}
                },
                "required": ["action", "name"]
            }),
        ),
        tool_def(
            "cuba_cronica",
            "Attach facts/lessons/decisions to entities. Also manages episodic memories (specific events with actors/artifacts) via episode_add/episode_list. Timeline view shows chronological history. Auto-creates entity if not found. Dedup gate blocks near-duplicates.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "delete", "list", "batch_add", "episode_add", "episode_list", "timeline"], "description": "Operation to perform. episode_add stores a temporal event; episode_list retrieves events. timeline shows chronological observations+episodes."},
                    "entity_name": {"type": "string", "description": "Entity to attach observation/episode to"},
                    "content": {"type": "string", "description": "Observation or episode text"},
                    "observation_type": {"type": "string", "enum": ["fact", "decision", "lesson", "preference", "context", "tool_usage"], "description": "Type of observation (for add action)"},
                    "source": {"type": "string", "enum": ["agent", "user", "error_detection"], "description": "Who/what created this observation"},
                    "actors": {"type": "array", "items": {"type": "string"}, "description": "People/agents involved in episode (for episode_add)"},
                    "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Files/resources affected in episode (for episode_add)"}
                },
                "required": ["action", "entity_name"]
            }),
        ),
        tool_def(
            "cuba_faro",
            "Search memory BEFORE answering to ground responses. Returns grounding scores. Mode 'verify' checks claims against evidence (confidence: verified/partial/weak/unknown). Session-aware: boosts results matching active session goals. Supports temporal filtering.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text"},
                    "mode": {"type": "string", "enum": ["hybrid", "verify"], "description": "Search mode (default: hybrid). 'verify' checks if claim is grounded."},
                    "scope": {"type": "string", "enum": ["all", "entities", "observations", "errors"], "description": "Where to search (default: all)"},
                    "limit": {"type": "integer", "description": "Max results (default 10, max 50)"},
                    "before": {"type": "string", "description": "ISO8601 datetime — return results created before this time"},
                    "after": {"type": "string", "description": "ISO8601 datetime — return results created after this time"},
                    "format": {"type": "string", "enum": ["verbose", "compact"], "description": "Response format: verbose (default, full data) or compact (abbreviated keys, ~35% fewer tokens)"},
                    "tags": {"type": "string", "description": "Filter observations by tag keyword (exact match against auto-extracted tags)"}
                },
                "required": ["query"]
            }),
        ),
        tool_def(
            "cuba_puente",
            "Create edges between entities (uses, causes, implements, depends_on, related_to). 'traverse' explores connections, 'infer' does transitive reasoning (A→B→C), 'predict' suggests missing links via Adamic-Adar. Relations strengthen with use (Hebbian).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "delete", "traverse", "infer", "predict"], "description": "Operation to perform. 'predict' uses Adamic-Adar to suggest missing relations."},
                    "from_entity": {"type": "string", "description": "Source entity name"},
                    "to_entity": {"type": "string", "description": "Target entity name"},
                    "relation_type": {"type": "string", "description": "Relation: uses, causes, implements, depends_on, related_to"},
                    "bidirectional": {"type": "boolean", "description": "If true, relation goes both ways"},
                    "start_entity": {"type": "string", "description": "Start point for traverse/infer"},
                    "max_depth": {"type": "integer", "description": "Max hops for traverse/infer (default 3, max 5)"},
                    "entity_name": {"type": "string", "description": "Entity name for predict action (Adamic-Adar link prediction)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_eco",
            "RLHF feedback: positive boosts importance (Oja's rule), negative decreases, correct updates content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["positive", "negative", "correct"], "description": "Feedback type"},
                    "entity_name": {"type": "string", "description": "Target entity"},
                    "observation_id": {"type": "string", "description": "Target observation UUID"},
                    "correction": {"type": "string", "description": "New content (for correct action)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_alarma",
            "Report errors immediately. Auto-detects patterns (≥3 similar = warning). Hebbian: similar errors get boosted for easier retrieval.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "error_type": {"type": "string", "description": "Error category: TypeError, ConnectionError, etc."},
                    "error_message": {"type": "string", "description": "Full error message"},
                    "context": {"type": "object", "description": "Context: {file, function, stack_trace, line}"},
                    "project": {"type": "string", "description": "Project name (default: 'default')"}
                },
                "required": ["error_type", "error_message"]
            }),
        ),
        tool_def(
            "cuba_remedio",
            "Mark an error as resolved with solution. Cross-references similar unresolved errors.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "error_id": {"type": "string", "description": "UUID of the error to solve"},
                    "solution": {"type": "string", "description": "Solution that fixed the error"}
                },
                "required": ["error_id", "solution"]
            }),
        ),
        tool_def(
            "cuba_expediente",
            "Search past errors/solutions. Use 'proposed_action' as anti-repetition guard: warns if similar approach previously failed.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text for errors"},
                    "project": {"type": "string", "description": "Filter by project"},
                    "resolved_only": {"type": "boolean", "description": "Only return errors with solutions"},
                    "proposed_action": {"type": "string", "description": "Anti-repetition: describe what you plan to do. Returns warning if similar approach failed before."}
                },
                "required": ["query"]
            }),
        ),
        tool_def(
            "cuba_jornada",
            "Track working sessions with goals and outcomes.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "end", "list", "current"], "description": "Session action"},
                    "name": {"type": "string", "description": "Session name (for start)"},
                    "goals": {"type": "array", "items": {"type": "string"}, "description": "Session goals (for start)"},
                    "outcome": {"type": "string", "enum": ["success", "partial", "failed", "abandoned"], "description": "Session outcome (for end)"},
                    "summary": {"type": "string", "description": "What was accomplished (for end)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_decreto",
            "Record and query architecture/design decisions.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["record", "query", "list"], "description": "Decision action"},
                    "title": {"type": "string", "description": "Decision title (for record)"},
                    "context": {"type": "string", "description": "Why this decision was needed"},
                    "alternatives": {"type": "array", "items": {"type": "string"}, "description": "Options considered"},
                    "chosen": {"type": "string", "description": "Option chosen"},
                    "rationale": {"type": "string", "description": "Why this option was chosen"},
                    "query": {"type": "string", "description": "Search text (for query action)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_vigia",
            "Knowledge graph analytics: summary (counts + token estimate), health (staleness, entropy, DB size), drift (chi-squared on errors), communities (Leiden), bridges (betweenness centrality).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["summary", "health", "drift", "communities", "bridges"], "description": "Metric to compute"}
                },
                "required": ["metric"]
            }),
        ),
        tool_def(
            "cuba_zafra",
            "Memory maintenance: decay (stratified exponential by type), prune (remove low-importance), merge (deduplicate), summarize (compress observations), pagerank (personalized importance), find_duplicates, export, stats, reembed (re-encode with current model).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["decay", "prune", "merge", "summarize", "stats", "pagerank", "find_duplicates", "export", "reembed", "decay_episodes"], "description": "Consolidation action. decay_episodes applies power-law decay to brain_episodes."},
                    "entity_name": {"type": "string", "description": "Entity to summarize (for summarize action)"},
                    "compressed_summary": {"type": "string", "description": "Compressed text replacing observations (for summarize)"},
                    "threshold": {"type": "number", "description": "Importance threshold for prune (default 0.1)"},
                    "similarity_threshold": {"type": "number", "description": "Similarity threshold for merge (default 0.8)"},
                    "batch_size": {"type": "integer", "description": "Max observations to re-encode in reembed (default 500)"},
                    "halflife_days": {"type": "number", "description": "Global halflife override for decay (overrides per-type stratification)"},
                    "c": {"type": "number", "description": "Power-law c parameter for decay_episodes (default 0.1)"},
                    "beta": {"type": "number", "description": "Power-law β exponent for decay_episodes (default 0.5)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_forget",
            "GDPR Right to Erasure: cascading hard-delete of an entity and ALL references across observations, relations, errors, and sessions. IRREVERSIBLE. Requires confirm=true.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Entity name to erase completely"},
                    "confirm": {"type": "boolean", "description": "Must be true to proceed (safety gate)"}
                },
                "required": ["entity_name", "confirm"]
            }),
        ),
        tool_def(
            "cuba_reflexion",
            "Analyze knowledge graph for structural gaps: isolated entities, underconnected hubs, type silos, observation gaps (missing decisions/lessons), and statistical density anomalies. Read-only introspection.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["analyze"], "description": "Gap analysis action (only 'analyze' supported)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_hipotesis",
            "Abductive inference: given an observed effect, find plausible causes by traversing causal relations backwards. Returns hypotheses ranked by plausibility (path_strength × importance). Read-only.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["explain"], "description": "Inference action"},
                    "effect": {"type": "string", "description": "Entity name representing the observed effect"},
                    "max_depth": {"type": "integer", "description": "Max causal chain hops (default 3, max 5)"},
                    "limit": {"type": "integer", "description": "Max hypotheses to return (default 10, max 50)"}
                },
                "required": ["action", "effect"]
            }),
        ),
        tool_def(
            "cuba_contradiccion",
            "Detect semantic contradictions between observations of the same entity. Uses embedding cosine distance + negation heuristics. Read-only.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["scan"], "description": "Contradiction detection action"},
                    "entity_name": {"type": "string", "description": "Entity to scan (omit to scan top entities by observation count)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_centinela",
            "Prospective memory: set triggers that fire when entities are accessed, sessions start, or errors match. 'Remember to remind me about X when Y happens.'",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "list", "delete", "check"], "description": "Trigger action"},
                    "entity_pattern": {"type": "string", "description": "Entity name or pattern to match"},
                    "condition_type": {"type": "string", "enum": ["on_access", "on_session_start", "on_error_match"], "description": "When to fire"},
                    "message": {"type": "string", "description": "Reminder message to surface when triggered"},
                    "max_fires": {"type": "integer", "description": "Max times to fire (default 1, -1 for unlimited)"},
                    "expires_at": {"type": "string", "description": "ISO8601 expiration datetime"},
                    "trigger_id": {"type": "string", "description": "Trigger UUID (for delete)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_calibrar",
            "Bayesian confidence calibration: track verify predictions, mark outcomes, compute P(correct|level). Closes the feedback loop between faro verify and eco correct.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["stats", "history", "resolve"], "description": "Calibration action"},
                    "verify_id": {"type": "string", "description": "Verify log UUID (for resolve)"},
                    "outcome": {"type": "string", "enum": ["correct", "incorrect"], "description": "Whether the verify prediction was right (for resolve)"},
                    "limit": {"type": "integer", "description": "Max results for history (default 20)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_ingesta",
            "Bulk knowledge ingestion: 'ingest' accepts an array of {entity_name, content, observation_type} items. 'parse' splits long text by paragraphs and auto-classifies each. Internally uses same dedup/embedding pipeline as cronica.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["ingest", "parse"], "description": "Ingestion mode. 'ingest' for structured items, 'parse' for raw text splitting."},
                    "items": {"type": "array", "items": {"type": "object"}, "description": "Array of {entity_name, content, observation_type?} objects (for ingest action, max 200)"},
                    "entity_name": {"type": "string", "description": "Entity to attach parsed observations to (for parse action)"},
                    "text": {"type": "string", "description": "Long text to split into observations (for parse action)"}
                },
                "required": ["action"]
            }),
        ),
    ]
}

/// Helper to build a tool definition.
fn tool_def(name: &str, description: &str, input_schema: Value) -> Value {
    serde_json::json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema
    })
}
