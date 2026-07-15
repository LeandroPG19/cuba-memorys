use serde_json::Value;
use std::sync::OnceLock;

pub const DEDUP_THRESHOLD: f64 = 0.85;

pub const PRED_ERROR_REINFORCE: f64 = 0.92;
pub const PRED_ERROR_UPDATE: f64 = 0.75;

pub const CACHE_MAX_ENTRIES: usize = 256;
pub const CACHE_TTL_SECS: u64 = 300;

pub const HEBBIAN_ACCESS_BOOST: f64 = 0.01;

pub const BCM_THROTTLE_SCALE: f64 = 0.8;

pub const KILL_SWITCH_ENV: &str = "CUBA_PROJECT_FILTER";

pub const COMPACTION_HINT_HOURS: i64 = 2;
pub const COMPACTION_HINT_OBS_COUNT: i64 = 100;

pub const JUEZ_AMBIGUOUS_LO: f64 = 0.6;
pub const JUEZ_AMBIGUOUS_HI: f64 = 0.8;
pub const JUEZ_DEFAULT_TIMEOUT_SECS: u64 = 30;
pub const JUEZ_DEFAULT_MAX_PAIRS: usize = 5;

pub const VALID_RELATION_TYPES: &[&str] =
    &["uses", "causes", "implements", "depends_on", "related_to"];

pub const VALID_ENTITY_TYPES: &[&str] = &[
    "concept",
    "project",
    "technology",
    "person",
    "pattern",
    "config",
];

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

pub const VALID_SOURCES: &[&str] = &[
    "agent",
    "error_detection",
    "user",
    "consolidation",
    "inference",
];

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

static TOOL_DEFS: OnceLock<Vec<Value>> = OnceLock::new();

pub fn tool_definitions() -> &'static Vec<Value> {
    TOOL_DEFS.get_or_init(|| {
        #[allow(unused_mut)]
        let mut defs = vec![
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
                    "observation_type": {"type": "string", "enum": ["fact", "decision", "lesson", "preference", "context", "tool_usage", "error", "solution"], "description": "Type of observation"},
                    "source": {"type": "string", "enum": ["agent", "user", "error_detection", "consolidation", "inference"], "description": "Who/what created this observation"},
                    "observation_id": {"type": "string", "description": "Observation UUID (for delete action)"},
                    "observations": {"type": "array", "items": {"type": "object"}, "description": "Array of {entity_name, content, observation_type?, source?} objects (for batch_add, max 100)"},
                    "actors": {"type": "array", "items": {"type": "string"}, "description": "People/agents involved in episode (for episode_add)"},
                    "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Files/resources affected in episode (for episode_add)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_faro",
            "Search memory BEFORE answering to ground responses. Returns grounding scores. Mode 'verify' checks claims against evidence (confidence: verified/partial/weak/unknown). Session-aware: boosts results matching active session goals. Supports temporal filtering. v0.9: optional MMR diversification + OOD abstention + exact tiktoken-based budget.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text"},
                    "mode": {"type": "string", "enum": ["hybrid", "verify"], "description": "Search mode (default: hybrid). 'verify' checks if claim is grounded."},
                    "scope": {"type": "string", "enum": ["all", "entities", "observations", "errors"], "description": "Where to search (default: all)"},
                    "limit": {"type": "integer", "description": "Max results (default 10, max 50)"},
                    "before": {"type": "string", "description": "ISO8601 datetime — return results created before this time"},
                    "after": {"type": "string", "description": "ISO8601 datetime — return results created after this time"},
                    "format": {"type": "string", "enum": ["verbose", "compact"], "description": "Response format. compact (DEFAULT): abbreviated keys — e=entity, c=content, t=type, i=importance, s=score. 71% fewer tokens (798 vs 2787 at limit=10, measured). verbose: full key names, only when you need every field."},
                    "tags": {"type": "string", "description": "Filter observations by tag keyword (exact match against auto-extracted tags)"},
                    "max_tokens": {"type": "integer", "description": "Token budget for results (default 5000). Counted exactly via tiktoken cl100k_base."},
                    "diversify": {"type": "boolean", "description": "v0.9: post-RRF MMR pass that penalizes near-duplicates among top-K. Default false."},
                    "mmr_lambda": {"type": "number", "description": "v0.9: MMR balance — 1.0 pure relevance, 0.0 pure diversity. Default 0.7."},
                    "abstain_ood": {"type": "boolean", "description": "v0.9: abstain (return empty results with abstain_reason) when query is out-of-distribution via Mahalanobis distance. Default false."},
                    "ood_threshold": {"type": "number", "description": "v0.9: Mahalanobis distance threshold for abstention. Defaults to sqrt(chi2_0.99(d)), which scales with the embedding dimension (~21.25 for d=384). Override only if you calibrated on your own corpus."},
                    "enable_bm25": {"type": "boolean", "description": "v0.9: enable BM25 (ts_rank_cd) as third RRF signal alongside text + vector. Catches queries with rare terms that dense embeddings miss. Default true."},
                    "rerank": {"type": "boolean", "description": "v0.9.2: cross-encoder rerank top-50 → top-K with bge-reranker-v2-m3 (Xiao 2023). Auto-enabled when CUBA_RERANKER_PATH points to a valid ONNX. Identity fallback otherwise."},
                    "associative": {"type": "boolean", "description": "v0.11: multi-hop expansion (HippoRAG-style). Seeds spreading activation from query-matched entities and pulls in observations on graph-connected entities that no lexical/vector signal surfaced. Additive — never lowers a base hit. Measured +10pts recall@10 on the smoke set. Default false."}
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
            "Track working sessions with goals and outcomes. v0.8: optional 'project' arg binds the session to a named project (upserts in brain_projects); subsequent handlers will scope reads/writes to that project.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "end", "list", "current"], "description": "Session action"},
                    "name": {"type": "string", "description": "Session name (for start)"},
                    "goals": {"type": "array", "items": {"type": "string"}, "description": "Session goals (for start)"},
                    "project": {"type": "string", "description": "v0.8: project name to bind this session to (created on first use). Omit to keep session global."},
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
            "Knowledge graph analytics: summary (counts + token estimate), health (staleness, entropy, DB size), drift (chi-squared on errors), communities (Leiden), bridges (betweenness centrality). v0.9: 'structural' returns harmonic + closeness + k-core ranking for backbone identification.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["summary", "health", "drift", "communities", "bridges", "structural"], "description": "Metric to compute. v0.9: 'structural' adds harmonic + closeness centrality (Boldi-Vigna 2014, Bavelas 1950) + k-core decomposition (Seidman 1983)."}
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
            "Bayesian confidence calibration: track verify predictions, mark outcomes, compute P(correct|level). Closes the feedback loop between faro verify and eco correct. v0.9: action 'trust' returns per-source credibility (Beta posterior updated by resolve outcomes; Yin-Han-Yu IEEE TKDE 2008).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["stats", "history", "resolve", "trust", "metrics"], "description": "Calibration action. v0.9: 'trust' returns per-source Beta(α, β) credibility; 'metrics' returns Brier score (1950) + Expected Calibration Error (Naeini AAAI 2015) + reliability diagram."},
                    "verify_id": {"type": "string", "description": "Verify log UUID (for resolve)"},
                    "outcome": {"type": "string", "enum": ["correct", "incorrect"], "description": "Whether the verify prediction was right (for resolve)"},
                    "limit": {"type": "integer", "description": "Max results for history (default 20)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_ingesta",
            "Bulk knowledge ingestion. 'ingest': array of {entity_name, content, observation_type} items. 'parse': split long text by paragraphs + heuristic classify. 'auto_extract' (v0.11): the calling client's LLM extracts salient durable facts from a turn/conversation via MCP Sampling ($0, no API key) and ingests them — the automatic-extraction that mem0/Zep have. All routes share the dedup/PE-gating/embedding pipeline; none delete.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["ingest", "parse", "auto_extract"], "description": "Ingestion mode. 'ingest' for structured items, 'parse' for raw text splitting, 'auto_extract' for LLM extraction via MCP sampling."},
                    "items": {"type": "array", "items": {"type": "object"}, "description": "Array of {entity_name, content, observation_type?} objects (for ingest action, max 200)"},
                    "entity_name": {"type": "string", "description": "Entity to attach parsed observations to (for parse action)"},
                    "text": {"type": "string", "description": "Raw text: paragraphs to split (parse) or a turn/conversation to extract facts from (auto_extract)"},
                    "entity_hint": {"type": "string", "description": "Optional main-subject hint for auto_extract (biases entity_name)"},
                    "supersede_conflicts": {"type": "boolean", "description": "v0.11 (auto_extract): when a new fact replaces/contradicts an existing related one, ask the judge and mark the old observation superseded (knowledge-update; never deletes). Default false."}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_proyecto",
            "Project scoping (v0.8): isolate memories per project so multiple projects sharing one DB don't bleed into each other. Active project is bound to the current session (cuba_jornada start --project NAME). Legacy rows with NULL project_id remain visible from every scope.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "current", "switch", "stats", "rename", "merge"], "description": "Project action"},
                    "name": {"type": "string", "description": "Project name (for switch/stats/rename source)"},
                    "to": {"type": "string", "description": "Destination name (for rename/merge)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_pre_compact",
            "Compaction-survival protocol (v0.8). Before the agent runs /compact, call action='snapshot' to persist a dense markdown summary of the active session (recent observations, decisions, unresolved errors, pending embeddings, goals). After compaction, call action='restore' to retrieve the latest snapshot for the active session and re-inject it into context.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["snapshot", "restore"], "description": "snapshot persists a session summary; restore returns the latest"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_sync",
            "Git-friendly export/import of the knowledge graph (v0.8). action='export' writes one JSON file per entity (with embedded observations) plus episodes/decisions/errors/relations under CUBA_SYNC_DIR (default ./.cuba-memorys/). 'import' merges files back via INSERT...ON CONFLICT DO NOTHING (idempotent). 'diff' compares disk vs DB. 'status' lists not-yet-imported manifests. Embeddings are omitted by default (set with_embeddings=true to include the binary blob).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["export", "import", "diff", "status"], "description": "Sync mode"},
                    "dir": {"type": "string", "description": "Directory override (default $CUBA_SYNC_DIR or ./.cuba-memorys/)"},
                    "scope": {"type": "string", "enum": ["project", "all"], "description": "Export scope: only the active project (default) or all data"},
                    "with_embeddings": {"type": "boolean", "description": "Include the embeddings.bin.zst blob on export (default false)"},
                    "conflict": {"type": "string", "enum": ["merge", "skip", "overwrite"], "description": "Import conflict policy (default merge)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_archivo",
            "Tamper-evident audit log (v0.9, CFR-21 Part 11 inspired). Append-only with SHA-256 hash chain — every row's current_hash commits to the previous row's, the action and the canonical payload. UPDATE/DELETE blocked at the PostgreSQL trigger level (only `cuba_admin` role can bypass). Use 'verify' to walk the chain and detect tampering, 'tail' to read recent events, 'append' to add a new event.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["append", "verify", "tail"], "description": "Audit operation"},
                    "event_action": {"type": "string", "description": "Event type (for append)"},
                    "payload": {"type": "object", "description": "Arbitrary JSON payload (for append)"},
                    "limit": {"type": "integer", "description": "Limit for verify/tail (default 10000 / 20)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_pizarra",
            "Working memory buffer (v0.9, Baddeley 1992): a TTL-bounded scratchpad orthogonal to episodic and semantic memory. Use for inter-step plan state during long-horizon agent tasks, tentative observations, cross-tool-call reminders inside one session. Auto-expire by ttl_seconds; bulk-purged by cuba_zafra REM cycle.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["write", "read", "clear"], "description": "Working-memory operation"},
                    "content": {"type": "string", "description": "Content to store (for write)"},
                    "tag": {"type": "string", "description": "Optional tag for filtering on read/clear"},
                    "ttl_seconds": {"type": "integer", "description": "Time-to-live in seconds (default 3600)"}
                },
                "required": ["action"]
            }),
        ),
        #[cfg(feature = "docs")]
        tool_def(
            "cuba_docs",
            "Read a library's CURRENT documentation from its official site. Use when you are about to write code against an API you have not verified this session — a renamed function or a changed signature is the most common way generated code fails, and memory cannot save you from it because your memory of the API is the thing that is wrong. `query` filters the page to the paragraphs that mention it. Requires the `docs` Cargo feature; every request is checked against SSRF (private ranges, cloud metadata, redirects).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "library": {"type": "string", "description": "Package name only — `tokio`, `sqlx`, `fastapi`, `react`. Not a URL. Unknown names are assumed to be Rust crates and resolved on docs.rs."},
                    "query": {"type": "string", "description": "What you need to know (e.g. `spawn_blocking`, `Depends`). Filters the page to matching paragraphs; omit for the overview."}
                },
                "required": ["library"]
            }),
        ),
        tool_def(
            "cuba_juez",
            "LLM-judge for semantically-conflicting observations (v0.8). When cosine similarity sits in the ambiguous band (0.6-0.8), heuristic detectors miss vocabulary-different conflicts (e.g. 'Postgres' vs 'MongoDB'). cuba_juez escalates a pair to a real LLM via subprocess (Claude Code CLI, $0 if you have a subscription) or — when feature 'anthropic-api' is built in — the Anthropic API directly. Verdicts are persisted in brain_judgments (UNIQUE per pair = permanent cache).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["judge_pair", "scan_entity"], "description": "judge_pair = decide on two given obs ids; scan_entity = pull ambiguous pairs and judge each"},
                    "observation_a": {"type": "string", "description": "UUID of first observation (for judge_pair)"},
                    "observation_b": {"type": "string", "description": "UUID of second observation (for judge_pair)"},
                    "entity_name": {"type": "string", "description": "Entity to scan (for scan_entity)"},
                    "max_pairs": {"type": "integer", "description": "Max pairs to escalate per call (default 5; controls LLM cost)"}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_receta",
            "PROCEDURAL MEMORY: how things are DONE here — bring up the dev services, run the test suite, deploy, migrate. \
             The other tools remember what is TRUE; this one remembers what to DO, so an agent stops rediscovering it every session. \
             Ranked by reliability, not by how often it is read: report the outcome with action='outcome' after running one, or the \
             memory learns nothing. A recipe that keeps failing is worse than none, because it is trusted.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "get", "search", "outcome", "list", "delete"], "description": "search: find by meaning. get: fetch by exact name. add: store/update (re-adding the same name edits it, keeping its track record). outcome: record success/failure — this is what teaches it."},
                    "name": {"type": "string", "description": "Procedure name, e.g. 'levantar el entorno de desarrollo'"},
                    "trigger": {"type": "string", "description": "WHEN this applies — the IF half. e.g. 'cuando hay que levantar los servicios de mapupita-web'"},
                    "steps": {"type": "array", "items": {"type": "object"}, "description": "Ordered steps: [{do: '...', run: 'comando'?, expect: 'qué debe pasar'?}]"},
                    "preconditions": {"type": "string", "description": "What must already be true before starting"},
                    "verification": {"type": "string", "description": "How you know it actually worked"},
                    "success": {"type": "boolean", "description": "For action=outcome: did it work?"},
                    "query": {"type": "string", "description": "For action=search"},
                    "limit": {"type": "integer", "description": "Max results"}
                },
                "required": ["action"]
            }),
        ),
    ];

        #[cfg(feature = "docs")]
        if !crate::handlers::docs::enabled() {
            defs.retain(|t| t.get("name").and_then(Value::as_str) != Some("cuba_docs"));
        }

        defs
    })
}

fn tool_def(name: &str, description: &str, input_schema: Value) -> Value {
    serde_json::json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema
    })
}

fn meta_tool_defs() -> Vec<Value> {
    vec![
        tool_def(
            "cuba_tools",
            "Find cuba-memorys tools and load their schemas ON DEMAND. The server exposes 25 tools; \
             under CUBA_TOOL_PROFILE=lean only the everyday core is pre-loaded and the rest live here. \
             Search by capability ('audit', 'decay', 'contradiction', 'session'), then call what you \
             find with cuba_call. detail='names' is cheapest, 'full' returns the exact argument schema.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Filter by capability — matches tool names and descriptions. Omit to list everything."},
                    "detail": {"type": "string", "enum": ["names", "summary", "full"], "description": "names: just the names. summary (default): name + description. full: the complete JSON Schema, which is what you need to call the tool correctly."}
                }
            }),
        ),
        tool_def(
            "cuba_call",
            "Invoke any cuba-memorys tool by name — including the ones not pre-loaded in this session. \
             Discover them first with cuba_tools (use detail='full' to see the exact arguments). \
             Goes through the same dispatcher as a direct call, so behaviour is identical.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "Tool name, e.g. cuba_zafra"},
                    "args": {"type": "object", "description": "The tool's own arguments, exactly as its schema declares them"}
                },
                "required": ["tool"]
            }),
        ),
    ]
}

const PROFILE_AGENT: [&str; 14] = [
    "cuba_receta",
    "cuba_faro",
    "cuba_cronica",
    "cuba_alma",
    "cuba_puente",
    "cuba_alarma",
    "cuba_remedio",
    "cuba_expediente",
    "cuba_jornada",
    "cuba_decreto",
    "cuba_ingesta",
    "cuba_proyecto",
    "cuba_pre_compact",
    "cuba_pizarra",
];

const PROFILE_STANDARD_EXTRA: [&str; 6] = [
    "cuba_eco",
    "cuba_reflexion",
    "cuba_hipotesis",
    "cuba_contradiccion",
    "cuba_centinela",
    "cuba_calibrar",
];

const PROFILE_LEAN: [&str; 6] = [
    "cuba_faro",
    "cuba_cronica",
    "cuba_expediente",
    "cuba_receta",
    "cuba_jornada",
    "cuba_alarma",
];

pub fn tools_for_profile() -> Vec<Value> {
    tools_for(&std::env::var("CUBA_TOOL_PROFILE").unwrap_or_else(|_| "full".to_string()))
}

pub fn tools_for(profile: &str) -> Vec<Value> {
    let all = tool_definitions();

    let allowed: Vec<&str> = match profile.to_lowercase().as_str() {
        "lean" => {
            let mut out: Vec<Value> = all
                .iter()
                .filter(|t| {
                    t.get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|n| PROFILE_LEAN.contains(&n))
                })
                .cloned()
                .collect();
            out.extend(meta_tool_defs());
            return out;
        }
        "agent" => PROFILE_AGENT.to_vec(),
        "standard" => PROFILE_AGENT
            .iter()
            .chain(PROFILE_STANDARD_EXTRA.iter())
            .copied()
            .collect(),
        _ => {
            let mut out = all.clone();
            out.extend(meta_tool_defs());
            return out;
        }
    };

    all.iter()
        .filter(|t| {
            t.get("name")
                .and_then(Value::as_str)
                .is_some_and(|n| allowed.contains(&n))
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn every_profiled_tool_actually_exists() {
        let names: Vec<&str> = tool_definitions()
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        for t in PROFILE_AGENT.iter().chain(PROFILE_STANDARD_EXTRA.iter()) {
            assert!(
                names.contains(t),
                "el perfil nombra una tool inexistente: {t}"
            );
        }
    }

    #[test]
    fn the_default_hides_nothing() {
        let full = tools_for("full");
        assert_eq!(full.len(), tool_definitions().len() + 2);
        for t in tool_definitions() {
            let name = t.get("name").and_then(Value::as_str).unwrap();
            assert!(
                full.iter()
                    .any(|f| f.get("name").and_then(Value::as_str) == Some(name)),
                "{name} desapareció del perfil full"
            );
        }
    }

    #[test]
    fn an_unknown_profile_falls_back_to_full() {
        let n = tools_for("full").len();
        assert_eq!(tools_for("typo-de-dedo").len(), n);
        assert_eq!(tools_for("").len(), n);
    }

    #[test]
    fn lean_defers_tools_it_does_not_delete_them() {
        let lean = tools_for("lean");
        assert_eq!(lean.len(), PROFILE_LEAN.len() + 2);
        let names: Vec<&str> = lean
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        assert!(
            names.contains(&"cuba_tools"),
            "lean sin cuba_tools deja las demás inalcanzables"
        );
        assert!(
            names.contains(&"cuba_call"),
            "lean sin cuba_call deja las demás inalcanzables"
        );
        assert!(names.contains(&"cuba_faro"));
    }

    #[test]
    fn narrow_profiles_are_strict_subsets() {
        assert_eq!(tools_for("agent").len(), PROFILE_AGENT.len());
        assert_eq!(
            tools_for("standard").len(),
            PROFILE_AGENT.len() + PROFILE_STANDARD_EXTRA.len()
        );
        let full: Vec<String> = tools_for("full")
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str).map(String::from))
            .collect();
        for t in tools_for("agent") {
            let name = t.get("name").and_then(Value::as_str).unwrap_or_default();
            assert!(full.contains(&name.to_string()), "{name} no está en full");
        }
    }
}
