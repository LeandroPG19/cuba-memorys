//! Constants, tool definitions, and threshold configuration.
//!
//! Mirrors Python constants.py — tool definitions for MCP tools/list.

use serde_json::Value;

// ── Thresholds ───────────────────────────────────────────────────

/// Default FSRS-6 parameters (Ye 2024) — 21 parameters.
pub const FSRS6_DEFAULT_PARAMS: [f64; 21] = [
    0.40255, 1.18385, 3.173, 15.69105,   // w[0..3]: initial stability
    7.1949,  0.5345,  1.4604, 0.0046,     // w[4..7]: difficulty
    1.54575, 0.1192,  1.01925, 1.9395,    // w[8..11]: recall
    0.11,    0.29605, 2.2698, 0.2315,     // w[12..15]: forget
    2.9898,  0.51655, 0.6621,             // w[16..18]: review
    0.0,     0.0,                          // w[19..20]: reserved
];

/// Factor for desired retention → retrievability threshold.
pub const DESIRED_RETENTION: f64 = 0.9;

/// Decay threshold — observations below this retrievability are eligible.
pub const DECAY_THRESHOLD: f64 = 0.3;

/// Deduplication similarity threshold (cosine similarity).
pub const DEDUP_THRESHOLD: f64 = 0.85;

/// Prediction Error Gating thresholds (V5 — Vestige-inspired).
pub const PRED_ERROR_REINFORCE: f64 = 0.92; // Very similar → reinforce existing
pub const PRED_ERROR_UPDATE: f64 = 0.75;     // Somewhat similar → update existing
// Below PRED_ERROR_UPDATE → create new observation

/// Cache configuration.
/// V3: TTL raised 60→300s to prevent thrashing during long tool executions.
/// Configurable via CUBA_CACHE_TTL env var.
pub const CACHE_MAX_ENTRIES: usize = 256;
pub const CACHE_TTL_SECS: u64 = 300;

/// Community summary cap (V9).
pub const COMMUNITY_SUMMARY_CAP: usize = 30;

/// Hebbian boost constants.
pub const HEBBIAN_ACCESS_BOOST: f64 = 0.01;
pub const HEBBIAN_SEARCH_BOOST: f64 = 0.02; // VF2: Testing Effect
pub const HEBBIAN_OJA_RATE: f64 = 0.05;      // Oja's learning rate
pub const HEBBIAN_MAX_IMPORTANCE: f64 = 1.0;

/// BCM Metaplasticity (Bienenstock-Cooper-Munro, 1982).
/// Throttle scale: how aggressively to reduce boost (0.0=off, 1.0=max throttle).
pub const BCM_THROTTLE_SCALE: f64 = 0.8;
/// High activity threshold: access_count above this triggers significant throttling.
pub const BCM_HIGH_ACTIVITY_THRESHOLD: f64 = 50.0;

/// Dual-Strength constants (VF1 — Bjork 1992).
pub const STORAGE_STRENGTH_INCREMENT: f64 = 0.1;
pub const RETRIEVAL_DECAY_FACTOR: f64 = 0.95;
pub const RETRIEVAL_SEARCH_BOOST: f64 = 0.05; // VF2: Testing Effect

/// FSRS-6 w20: Default decay rate (Expertium 2025).
/// Range [0.1, 0.8]. Adaptive per-entity via sigmoid(access_count).
pub const DEFAULT_DECAY_RATE: f64 = 0.5;

// RRF k=60 constant moved to search/rrf.rs and handlers/faro.rs (V4)
// Adaptive RRF_K_MIN/MAX removed per Gemini Deep Research audit 2026-03-14.

/// Relation types.
pub const VALID_RELATION_TYPES: &[&str] = &[
    "uses", "causes", "implements", "depends_on", "related_to",
];

/// Entity types.
pub const VALID_ENTITY_TYPES: &[&str] = &[
    "concept", "project", "technology", "person", "pattern", "config",
];

/// Observation types.
pub const VALID_OBSERVATION_TYPES: &[&str] = &[
    "fact", "decision", "lesson", "preference",
    "error", "solution", "context", "tool_usage", "superseded",
];

/// Observation sources.
pub const VALID_SOURCES: &[&str] = &[
    "agent", "error_detection", "user", "consolidation", "inference",
];

// ── Tool Definitions ─────────────────────────────────────────────

/// Generate MCP tool definitions for tools/list response.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        tool_def("cuba_alma", "CRUD knowledge graph entities (concepts, projects, technologies, patterns, people). Auto-boosts neighbors on access. For transient info use cuba_cronica instead.", serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "update", "delete", "get"], "description": "Operation to perform"},
                "name": {"type": "string", "description": "Entity name (unique identifier)"},
                "entity_type": {"type": "string", "description": "Type: concept, project, technology, person, pattern, config"},
                "new_name": {"type": "string", "description": "New name for update action"}
            },
            "required": ["action", "name"]
        })),
        tool_def("cuba_cronica", "Attach facts/lessons/decisions to entities. Auto-creates entity if not found. Dedup gate blocks near-duplicates. Contradictions auto-supersede old facts. Use batch_add with 'observations' array for bulk writes.", serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "delete", "list", "batch_add"], "description": "Operation to perform. batch_add accepts 'observations' array."},
                "entity_name": {"type": "string", "description": "Entity to attach observation to"},
                "content": {"type": "string", "description": "Observation text"},
                "observation_type": {"type": "string", "enum": ["fact", "decision", "lesson", "preference", "context", "tool_usage"], "description": "Type of observation"},
                "source": {"type": "string", "enum": ["agent", "user", "error_detection"], "description": "Who/what created this observation"}
            },
            "required": ["action", "entity_name"]
        })),
        tool_def("cuba_faro", "Search memory BEFORE answering to ground responses. Returns grounding scores. Mode 'verify' checks claims against evidence (confidence: verified/partial/weak/unknown). Session-aware: boosts results matching active session goals.", serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text"},
                "mode": {"type": "string", "enum": ["hybrid", "verify"], "description": "Search mode (default: hybrid). 'verify' checks if claim is grounded."},
                "scope": {"type": "string", "enum": ["all", "entities", "observations", "errors"], "description": "Where to search (default: all)"},
                "limit": {"type": "integer", "description": "Max results (default 10, max 50)"}
            },
            "required": ["query"]
        })),
        tool_def("cuba_puente", "Create edges between entities (uses, causes, implements, depends_on, related_to). 'traverse' explores connections, 'infer' does transitive reasoning (A→B→C). Relations strengthen with use (Hebbian).", serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "delete", "traverse", "infer"], "description": "Operation to perform"},
                "from_entity": {"type": "string", "description": "Source entity name"},
                "to_entity": {"type": "string", "description": "Target entity name"},
                "relation_type": {"type": "string", "description": "Relation: uses, causes, implements, depends_on, related_to"},
                "bidirectional": {"type": "boolean", "description": "If true, relation goes both ways"},
                "start_entity": {"type": "string", "description": "Start point for traverse/infer"},
                "max_depth": {"type": "integer", "description": "Max hops for traverse/infer (default 3, max 5)"}
            },
            "required": ["action"]
        })),
        tool_def("cuba_eco", "RLHF feedback: positive boosts importance (Oja's rule), negative decreases, correct updates content.", serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["positive", "negative", "correct"], "description": "Feedback type"},
                "entity_name": {"type": "string", "description": "Target entity"},
                "observation_id": {"type": "string", "description": "Target observation UUID"},
                "correction": {"type": "string", "description": "New content (for correct action)"}
            },
            "required": ["action"]
        })),
        tool_def("cuba_alarma", "Report errors immediately. Auto-detects patterns (≥3 similar = warning). Hebbian: similar errors get boosted for easier retrieval.", serde_json::json!({
            "type": "object",
            "properties": {
                "error_type": {"type": "string", "description": "Error category: TypeError, ConnectionError, etc."},
                "error_message": {"type": "string", "description": "Full error message"},
                "context": {"type": "object", "description": "Context: {file, function, stack_trace, line}"},
                "project": {"type": "string", "description": "Project name (default: 'default')"}
            },
            "required": ["error_type", "error_message"]
        })),
        tool_def("cuba_remedio", "Mark an error as resolved with solution. Cross-references similar unresolved errors.", serde_json::json!({
            "type": "object",
            "properties": {
                "error_id": {"type": "string", "description": "UUID of the error to solve"},
                "solution": {"type": "string", "description": "Solution that fixed the error"}
            },
            "required": ["error_id", "solution"]
        })),
        tool_def("cuba_expediente", "Search past errors/solutions. Use 'proposed_action' as anti-repetition guard: warns if similar approach previously failed.", serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text for errors"},
                "project": {"type": "string", "description": "Filter by project"},
                "resolved_only": {"type": "boolean", "description": "Only return errors with solutions"},
                "proposed_action": {"type": "string", "description": "Anti-repetition: describe what you plan to do. Returns warning if similar approach failed before."}
            },
            "required": ["query"]
        })),
        tool_def("cuba_jornada", "Track working sessions with goals and outcomes.", serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "end", "list", "current"], "description": "Session action"},
                "name": {"type": "string", "description": "Session name (for start)"},
                "goals": {"type": "array", "items": {"type": "string"}, "description": "Session goals (for start)"},
                "outcome": {"type": "string", "enum": ["success", "partial", "failed", "abandoned"], "description": "Session outcome (for end)"},
                "summary": {"type": "string", "description": "What was accomplished (for end)"}
            },
            "required": ["action"]
        })),
        tool_def("cuba_decreto", "Record and query architecture/design decisions.", serde_json::json!({
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
        })),
        tool_def("cuba_vigia", "Knowledge graph analytics: summary (counts + token estimate), health (staleness, entropy, DB size), drift (chi-squared on errors), communities (Leiden), bridges (betweenness centrality).", serde_json::json!({
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["summary", "health", "drift", "communities", "bridges"], "description": "Metric to compute"}
            },
            "required": ["metric"]
        })),
        tool_def("cuba_zafra", "Memory maintenance: decay (FSRS adaptive), prune (remove low-importance), merge (deduplicate), summarize (compress observations), pagerank (personalized importance), find_duplicates, export, stats.", serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["decay", "prune", "merge", "summarize", "stats", "pagerank", "find_duplicates", "export", "backfill"], "description": "Consolidation action"},
                "entity_name": {"type": "string", "description": "Entity to summarize (for summarize action)"},
                "compressed_summary": {"type": "string", "description": "Compressed text replacing observations (for summarize)"},
                "threshold": {"type": "number", "description": "Importance threshold for prune (default 0.1)"},
                "similarity_threshold": {"type": "number", "description": "Similarity threshold for merge (default 0.8)"}
            },
            "required": ["action"]
        })),
        tool_def("cuba_forget", "GDPR Right to Erasure: cascading hard-delete of an entity and ALL references across observations, relations, errors, and sessions. IRREVERSIBLE. Requires confirm=true.", serde_json::json!({
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Entity name to erase completely"},
                "confirm": {"type": "boolean", "description": "Must be true to proceed (safety gate)"}
            },
            "required": ["entity_name", "confirm"]
        })),
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
