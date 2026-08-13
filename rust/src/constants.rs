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
        let defs = vec![
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
                    "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Files/resources affected in episode (for episode_add)"},
                    "allow_secret": {"type": "boolean", "description": "Writes are refused when the text carries what looks like a live credential (token, password field, credentials in a URL): stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
                    "relation_type": {"type": "string", "description": "Relation: uses, causes, implements, depends_on, related_to. Also used by predict+persist to pick the type for the persisted edge (default related_to)."},
                    "bidirectional": {"type": "boolean", "description": "If true, relation goes both ways"},
                    "start_entity": {"type": "string", "description": "Start point for traverse/infer"},
                    "max_depth": {"type": "integer", "description": "Max hops for traverse/infer (default 3, max 5)"},
                    "entity_name": {"type": "string", "description": "Entity name for predict action (Adamic-Adar link prediction)"},
                    "persist": {"type": "boolean", "description": "For predict: write the suggestions to brain_relations as provenance='predicted' (relation_type related_to) instead of only returning them. Default false — read-only."}
                },
                "required": ["action"]
            }),
        ),
        tool_def(
            "cuba_eco",
            "RLHF feedback: positive boosts importance (Oja's rule), negative decreases, correct updates content. Also the quarantine gate: 'pending' lists memories withheld from search because they came from untrusted text, 'promote' makes one retrievable, 'quarantine' withdraws one. The gate covers observations, episodes and errors — pick which with 'kind'.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["positive", "negative", "correct", "promote", "quarantine", "pending"], "description": "Feedback type, or a quarantine transition: promote/quarantine flip one memory's retrievability; pending lists everything currently withheld, in three lists (quarantined, quarantined_episodes, quarantined_errors), each row tagged with its kind."},
                    "kind": {"type": "string", "enum": ["observation", "episode", "error"], "description": "Which table promote/quarantine acts on. Default 'observation'. An import quarantines whatever carried a credential, and cuba_sync writes episodes and errors too: without the matching kind those rows would stay stored and permanently unreachable. Ignored by positive/negative/correct, and by pending, which always returns all three."},
                    "entity_name": {"type": "string", "description": "Target entity"},
                    "observation_id": {"type": "string", "description": "Target observation UUID"},
                    "id": {"type": "string", "description": "Target UUID for promote/quarantine when kind is episode or error. For kind=observation use observation_id."},
                    "correction": {"type": "string", "description": "New content (for correct action)"},
                    "limit": {"type": "integer", "description": "Max rows for the 'pending' listing (default 20, max 200)"},
                    "allow_secret": {"type": "boolean", "description": "A correction is refused when it carries what looks like a live credential (token, password field, credentials in a URL) — it overwrites stored content, so without this gate it is a way past the one cuba_cronica applies on the way in. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
                    "project": {"type": "string", "description": "Project name (default: 'default')"},
                    "allow_secret": {"type": "boolean", "description": "Writes are refused when error_message or context carry what looks like a live credential (token, password field, credentials in a URL) — a stack trace with a header in it is the usual case. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
                    "solution": {"type": "string", "description": "Solution that fixed the error"},
                    "allow_secret": {"type": "boolean", "description": "Writes are refused when the solution carries what looks like a live credential (token, password field, credentials in a URL) — 'it was fixed by exporting this key' is the usual case. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
                    "summary": {"type": "string", "description": "What was accomplished (for end)"},
                    "allow_secret": {"type": "boolean", "description": "Ending a session is refused when the summary carries what looks like a live credential (token, password field, credentials in a URL); the summary is replayed to the next session as previous_session.summary. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
                    "query": {"type": "string", "description": "Search text (for query action)"},
                    "allow_secret": {"type": "boolean", "description": "Writes are refused when the recorded decision carries what looks like a live credential (token, password field, credentials in a URL): stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
            "Memory maintenance: decay (stratified exponential by type), prune (remove low-importance), merge (deduplicate), summarize (compress observations), pagerank (personalized importance), find_duplicates, export, stats, reembed (re-encode with current model). prune PLANS by default: it reports how many observations it would delete, broken down by project, and only deletes when you pass confirm=true. Every action is scoped to the active project.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["decay", "prune", "merge", "summarize", "stats", "pagerank", "find_duplicates", "export", "reembed", "decay_episodes"], "description": "Consolidation action. decay_episodes applies power-law decay to brain_episodes."},
                    "confirm": {"type": "boolean", "description": "prune only: actually delete. Without it prune returns a dry-run plan with would_prune and by_project, and deletes nothing. Read the plan before setting this — the default threshold reaches a large share of a mature corpus."},
                    "entity_name": {"type": "string", "description": "Entity to summarize (for summarize action)"},
                    "compressed_summary": {"type": "string", "description": "Compressed text replacing observations (for summarize)"},
                    "threshold": {"type": "number", "description": "Importance threshold for prune (default 0.1)"},
                    "similarity_threshold": {"type": "number", "description": "Similarity threshold for merge (default 0.8)"},
                    "batch_size": {"type": "integer", "description": "Max observations to re-encode in reembed (default 500)"},
                    "halflife_days": {"type": "number", "description": "Global halflife override for decay (overrides per-type stratification)"},
                    "c": {"type": "number", "description": "Power-law c parameter for decay_episodes (default 0.1)"},
                    "beta": {"type": "number", "description": "Power-law β exponent for decay_episodes (default 0.5)"},
                    "allow_secret": {"type": "boolean", "description": "summarize is refused when compressed_summary carries what looks like a live credential (token, password field, credentials in a URL); it supersedes every observation of the entity, so this text is the only survivor. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
            "Prospective memory: set triggers that fire when entities are accessed, sessions start, or errors match. 'Remember to remind me about X when Y happens.' Between two AI sessions on the same daemon this is also the note channel: condition_type='on_session_start' with the other session's name as entity_pattern reaches it the next time it opens, exactly once (max_fires), and carries who left it.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "list", "delete", "check"], "description": "Trigger action"},
                    "entity_pattern": {"type": "string", "description": "Entity name or pattern to match"},
                    "condition_type": {"type": "string", "enum": ["on_access", "on_session_start", "on_error_match"], "description": "When to fire"},
                    "message": {"type": "string", "description": "Reminder message to surface when triggered"},
                    "allow_secret": {"type": "boolean", "description": "Creating a trigger is refused when the message carries what looks like a live credential (token, password field, credentials in a URL); the message is pushed unprompted into the response of whatever matches the pattern. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."},
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
                    "supersede_conflicts": {"type": "boolean", "description": "v0.11 (auto_extract): when a new fact replaces/contradicts an existing related one, ask the judge and mark the old observation superseded (knowledge-update; never deletes). Default false."},
                    "untrusted": {"type": "boolean", "description": "Set when the text came from somewhere you do not control (a fetched page, a pasted document, a third party). Everything extracted lands quarantined — stored and inspectable via cuba_eco action=pending, but withheld from cuba_faro until promoted. Default false."},
                    "allow_secret": {"type": "boolean", "description": "Ingestion is refused when an item's content, or the raw text handed to parse/auto_extract, carries what looks like a live credential (token, password field, credentials in a URL) — pasting a whole work log is the likeliest way one arrives. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
            "Git-friendly export/import of the knowledge graph. action='export' writes one JSON file per entity (observations embedded) plus episodes, errors, relations, projects and tombstones under CUBA_SYNC_DIR (default ./.cuba-memorys/). 'import' applies the bundle: deletions travel as tombstones and are applied before inserts, rows the database would reject are refused before the transaction opens rather than aborting it halfway, and importance/access_count/strength are merged as the maximum of both machines so neither side loses what it learned. 'diff' compares disk vs DB. 'status' lists not-yet-imported manifests. 'pull' returns the bundle in the response instead of writing it anywhere, paged by file, for another machine holding a peer token: it is the only write-free way to hand this node's memory over, and the receiving side writes the pages to its own directory and runs a normal import. 'notify' is the one write a peer token may make: a short message saying the other machine learned something, which surfaces at the next cuba_jornada start and in status, and closes itself when a bundle with its manifest_hash is imported. It never enters the graph. 'conflicts' lists the rows two machines disagree about, with both texts, and 'resolve' closes one with keep=ours|theirs|both — 'both' keeps this machine's text current and files the other in previous_versions, discarding nothing. 'fetch' is the other half and runs on the local machine: it pages a peer's pull over HTTP with CUBA_PEER_TOKEN, lands the files, imports them with the same validation as any bundle, and records the peer's manifest hash so the next fetch stops before opening a transaction when nothing changed. Embeddings are omitted by default; when included, a bundle whose model or dimension does not match this machine is refused rather than silently filling the index with vectors from another space.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["export", "import", "diff", "status", "pull", "notify", "fetch", "conflicts", "resolve"], "description": "Sync mode"},
                    "id": {"type": "string", "description": "resolve only: the conflict id from action=conflicts."},
                    "keep": {"type": "string", "enum": ["ours", "theirs", "both"], "description": "resolve only: which text stays current. Default 'both' — this machine's text stays and the other machine's is filed in previous_versions, which is the only choice that loses nothing. 'theirs' also clears the embedding, because it described text that is no longer there."},
                    "peer": {"type": "string", "description": "fetch only: which peer to pull from (default 'default'). The address is remembered per name after the first fetch."},
                    "url": {"type": "string", "description": "fetch only: base address of the other daemon, e.g. https://brain.example.net. Falls back to the address recorded for this peer, then to CUBA_PEER_URL."},
                    "summary": {"type": "string", "description": "notify only: what the other machine learned, in at most 2000 characters. It is a signal, not a payload — the memory itself travels through pull, which is validated and quarantined."},
                    "node_id": {"type": "string", "description": "notify only: which node is speaking. Self-asserted and unverified: the token is the authentication, this is a label."},
                    "node_name": {"type": "string", "description": "notify only: the readable name of the node sending the notice."},
                    "manifest_hash": {"type": "string", "description": "notify only: the bundle hash this notice refers to. Importing a bundle with that hash closes the notice, which is what stops the same signal being reported forever."},
                    "limit": {"type": "integer", "description": "pull only: at most this many files per page. The response is capped by size regardless; this is for a peer on a slow link that wants smaller pages than the 3 MB budget."},
                    "offset": {"type": "integer", "description": "pull only: index of the first bundle file to return. Page until has_more is false, and abort if manifest_hash changes between pages — that means this node was written to mid-transfer and the pages describe two different states."},
                    "dir": {"type": "string", "description": "Directory override (default $CUBA_SYNC_DIR or ./.cuba-memorys/)"},
                    "scope": {"type": "string", "enum": ["project", "all"], "description": "Export scope: only the active project (default) or all data"},
                    "with_embeddings": {"type": "boolean", "description": "Include the embeddings.bin.zst blob. Default false on export, true on pull: a peer that receives text without vectors cannot search what it just received until it re-embeds, and on a machine without a GPU that is slow and sequential."},
                    "conflict": {"type": "string", "enum": ["merge", "skip", "overwrite"], "description": "How to resolve a row that exists on both sides. merge and skip behave identically for CONTENT — whatever is already here wins and the incoming text is dropped, with the diverging ids reported. overwrite takes the incoming text and pushes the replaced version into previous_versions, so nothing is lost either way. Counters are not content: entity importance and access_count, and relation strength, merge as the maximum under both policies."},
                    "confirm": {"type": "boolean", "description": "Required when a bundle's tombstones would delete more than 10% of this machine's observations, and at least 25 rows. Below that it is not needed — a guard that trips on ordinary curation is one everybody learns to pass through. A bundle that deletes a large share of your memory is either a mistake or a peer worth distrusting, and this is what stands between that and a remote wipe."}
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
                    "ttl_seconds": {"type": "integer", "description": "Time-to-live in seconds (default 3600)"},
                    "allow_secret": {"type": "boolean", "description": "Writes are refused when the content carries what looks like a live credential (token, password field, credentials in a URL); a TTL is not protection — the row is readable, dumped and backed up for as long as it lives. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
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
            "LLM-judge for semantically-conflicting observations (v0.8). When cosine similarity sits in the ambiguous band (0.6-0.8), heuristic detectors miss vocabulary-different conflicts (e.g. 'Postgres' vs 'MongoDB'). cuba_juez escalates a pair to a real LLM via subprocess (Claude Code CLI, $0 if you have a subscription). Verdicts are persisted in brain_judgments (UNIQUE per pair = permanent cache).",
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
                    "limit": {"type": "integer", "description": "Max results"},
                    "allow_secret": {"type": "boolean", "description": "Adding a procedure is refused when any part of it — name, trigger, a step's `run` command, preconditions or verification — carries what looks like a live credential (token, password field, credentials in a URL). A recipe is a list of commands to paste, so `run` is where one usually lands. Stored memory is searchable, exported to JSON inside a git repo, and served to every client. Set true only when the match is not a live credential — the text is stored verbatim, in clear."}
                },
                "required": ["action"]
            }),
        ),
    ];

        #[cfg(feature = "docs")]
        let defs = if crate::handlers::docs::enabled() {
            defs
        } else {
            defs.into_iter()
                .filter(|t| t.get("name").and_then(Value::as_str) != Some("cuba_docs"))
                .collect()
        };

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
            "Find cuba-memorys tools and load their schemas ON DEMAND. The server exposes 28 tools; \
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

    #[test]
    fn a_tool_that_can_destroy_data_declares_its_safety_switch() {
        const GUARDED: [&str; 2] = ["cuba_forget", "cuba_zafra"];

        for name in GUARDED {
            let tool = tool_definitions()
                .iter()
                .find(|t| t.get("name").and_then(Value::as_str) == Some(name))
                .unwrap_or_else(|| panic!("{name} disappeared from the catalogue"));

            let confirm = tool.pointer("/inputSchema/properties/confirm");
            assert!(
                confirm.is_some_and(|c| c.get("type").and_then(Value::as_str) == Some("boolean")),
                "{name} deletes rows and gates that behind `confirm`, but its schema does not \
                 declare the key. A caller cannot discover a switch that is not in the schema, \
                 and clients that strip undeclared properties would make the guarded path \
                 unreachable"
            );

            let described = confirm
                .and_then(|c| c.get("description"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            assert!(
                !described.trim().is_empty(),
                "{name}: a boolean called `confirm` with no description tells the caller \
                 nothing about what it is confirming"
            );
        }
    }

    #[test]
    fn a_tool_that_stores_free_text_declares_its_secret_override() {
        const GATED: [&str; 11] = [
            "cuba_cronica",
            "cuba_decreto",
            "cuba_alarma",
            "cuba_remedio",
            "cuba_ingesta",
            "cuba_eco",
            "cuba_zafra",
            "cuba_receta",
            "cuba_pizarra",
            "cuba_jornada",
            "cuba_centinela",
        ];

        for name in GATED {
            let tool = tool_definitions()
                .iter()
                .find(|t| t.get("name").and_then(Value::as_str) == Some(name))
                .unwrap_or_else(|| panic!("{name} disappeared from the catalogue"));

            let allow = tool.pointer("/inputSchema/properties/allow_secret");
            assert!(
                allow.is_some_and(|a| a.get("type").and_then(Value::as_str) == Some("boolean")),
                "{name} refuses to store text that looks like a live credential, and \
                 `allow_secret` is the only way past that refusal. A caller cannot discover a \
                 switch that is not in the schema, and clients that strip undeclared properties \
                 would leave the legitimate write with no way through at all"
            );

            let described = allow
                .and_then(|a| a.get("description"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            assert!(
                described.contains("clear"),
                "{name}: `allow_secret` must say that the text is then stored verbatim and in \
                 clear. A caller who reads the switch as 'store it safely' is the one who will \
                 flip it on a live credential"
            );
        }
    }
}
