"""Constants, thresholds, and tool definitions for cuba-memorys."""

# ── Input Validation Limits ─────────────────────────────────────────
MAX_NAME_LENGTH: int = 200
MAX_CONTENT_LENGTH: int = 50_000
MAX_ERROR_MESSAGE_LENGTH: int = 10_000

# ── Search & Dedup Thresholds ───────────────────────────────────────
DEDUP_THRESHOLD: float = 0.85
CONTRADICTION_THRESHOLD: float = 0.70
TRIGRAM_SEARCH_THRESHOLD: float = 0.30
SIMILARITY_SEARCH_THRESHOLD: float = 0.40

# ── Importance Constants ────────────────────────────────────────────
IMPORTANCE_ACCESS_BOOST: float = 0.02
IMPORTANCE_NEIGHBOR_BOOST: float = 0.006
IMPORTANCE_TRAVERSE_BOOST: float = 0.05
OBS_OVERLOAD_THRESHOLD: int = 20

# ── Export Safety Limits ────────────────────────────────────────────
EXPORT_MAX_ENTITIES: int = 5_000
EXPORT_MAX_OBSERVATIONS: int = 5_000
EXPORT_MAX_RELATIONS: int = 5_000
EXPORT_MAX_ERRORS: int = 2_000
EXPORT_MAX_SESSIONS: int = 500

# ── REM Sleep Config ────────────────────────────────────────────────
REM_IDLE_SECONDS: float = 900.0

# ── Domain Enums ────────────────────────────────────────────────────
VALID_ENTITY_TYPES: frozenset[str] = frozenset(
    {"concept", "project", "technology", "person", "pattern", "config"}
)

# ── Graph SQL ───────────────────────────────────────────────────────
GRAPH_RELATIONS_SQL: str = (
    "SELECT e1.name AS src, e2.name AS dst, r.strength "
    "FROM brain_relations r "
    "JOIN brain_entities e1 ON r.from_entity = e1.id "
    "JOIN brain_entities e2 ON r.to_entity = e2.id"
)

# ── Tool Definitions (MCP Schema) ──────────────────────────────────
TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "cuba_alma",
        "description": (
            "CRUD knowledge graph entities (concepts, projects, technologies, patterns, people). "
            "Auto-boosts neighbors on access. For transient info use cuba_cronica instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "delete", "get"],
                    "description": "Operation to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Entity name (unique identifier)",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Type: concept, project, technology, person, pattern, config",
                },
                "new_name": {
                    "type": "string",
                    "description": "New name for update action",
                },
            },
            "required": ["action", "name"],
        },
    },
    {
        "name": "cuba_cronica",
        "description": (
            "Attach facts/lessons/decisions to entities. Auto-creates entity if not found. "
            "Dedup gate blocks near-duplicates. Contradictions auto-supersede old facts. "
            "Use batch_add with 'observations' array for bulk writes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "delete", "list", "batch_add"],
                    "description": "Operation to perform. batch_add accepts 'observations' array.",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity to attach observation to",
                },
                "content": {
                    "type": "string",
                    "description": "Observation text",
                },
                "observation_type": {
                    "type": "string",
                    "enum": [
                        "fact", "decision", "lesson", "preference",
                        "context", "tool_usage",
                    ],
                    "description": "Type of observation",
                },
                "source": {
                    "type": "string",
                    "enum": ["agent", "user", "error_detection"],
                    "description": "Who/what created this observation",
                },
            },
            "required": ["action", "entity_name"],
        },
    },
    {
        "name": "cuba_puente",
        "description": (
            "Create edges between entities (uses, causes, implements, depends_on, related_to). "
            "'traverse' explores connections, 'infer' does transitive reasoning (A→B→C). "
            "Relations strengthen with use (Hebbian)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "delete", "traverse", "infer"],
                    "description": "Operation to perform",
                },
                "from_entity": {
                    "type": "string",
                    "description": "Source entity name",
                },
                "to_entity": {
                    "type": "string",
                    "description": "Target entity name",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Relation: uses, causes, implements, depends_on, related_to",
                },
                "bidirectional": {
                    "type": "boolean",
                    "description": "If true, relation goes both ways",
                },
                "start_entity": {
                    "type": "string",
                    "description": "Start point for traverse/infer",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Max hops for traverse/infer (default 3, max 5)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "cuba_faro",
        "description": (
            "Search memory BEFORE answering to ground responses. Returns grounding scores. "
            "Mode 'verify' checks claims against evidence (confidence: verified/partial/weak/unknown). "
            "Session-aware: boosts results matching active session goals."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "verify"],
                    "description": "Search mode (default: hybrid). 'verify' checks if claim is grounded.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["all", "entities", "observations", "errors"],
                    "description": "Where to search (default: all)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10, max 50)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "cuba_alarma",
        "description": (
            "Report errors immediately. Auto-detects patterns (≥3 similar = warning). "
            "Hebbian: similar errors get boosted for easier retrieval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "description": "Error category: TypeError, ConnectionError, etc.",
                },
                "error_message": {
                    "type": "string",
                    "description": "Full error message",
                },
                "context": {
                    "type": "object",
                    "description": "Context: {file, function, stack_trace, line}",
                },
                "project": {
                    "type": "string",
                    "description": "Project name (default: 'default')",
                },
            },
            "required": ["error_type", "error_message"],
        },
    },
    {
        "name": "cuba_remedio",
        "description": (
            "Mark an error as resolved with solution. "
            "Cross-references similar unresolved errors."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "error_id": {
                    "type": "string",
                    "description": "UUID of the error to solve",
                },
                "solution": {
                    "type": "string",
                    "description": "Solution that fixed the error",
                },
            },
            "required": ["error_id", "solution"],
        },
    },
    {
        "name": "cuba_expediente",
        "description": (
            "Search past errors/solutions. Use 'proposed_action' as anti-repetition guard: "
            "warns if similar approach previously failed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text for errors",
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project",
                },
                "resolved_only": {
                    "type": "boolean",
                    "description": "Only return errors with solutions",
                },
                "proposed_action": {
                    "type": "string",
                    "description": "Anti-repetition: describe what you plan to do. "
                    "Returns warning if similar approach failed before.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "cuba_jornada",
        "description": "Track working sessions with goals and outcomes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "end", "list", "current"],
                    "description": "Session action",
                },
                "name": {
                    "type": "string",
                    "description": "Session name (for start)",
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Session goals (for start)",
                },
                "summary": {
                    "type": "string",
                    "description": "What was accomplished (for end)",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "partial", "failed", "abandoned"],
                    "description": "Session outcome (for end)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "cuba_decreto",
        "description": "Record and query architecture/design decisions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["record", "query", "list"],
                    "description": "Decision action",
                },
                "title": {
                    "type": "string",
                    "description": "Decision title (for record)",
                },
                "context": {
                    "type": "string",
                    "description": "Why this decision was needed",
                },
                "alternatives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Options considered",
                },
                "chosen": {
                    "type": "string",
                    "description": "Option chosen",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this option was chosen",
                },
                "query": {
                    "type": "string",
                    "description": "Search text (for query action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "cuba_zafra",
        "description": (
            "Memory maintenance: decay (FSRS adaptive), prune (remove low-importance), "
            "merge (deduplicate), summarize (compress observations), "
            "pagerank (personalized importance), find_duplicates, export, stats."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["decay", "prune", "merge", "summarize", "stats", "pagerank", "find_duplicates", "export", "backfill"],
                    "description": "Consolidation action",
                },
                "threshold": {
                    "type": "number",
                    "description": "Importance threshold for prune (default 0.1)",
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Similarity threshold for merge (default 0.8)",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity to summarize (for summarize action)",
                },
                "compressed_summary": {
                    "type": "string",
                    "description": "Compressed text replacing observations (for summarize)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "cuba_eco",
        "description": (
            "RLHF feedback: positive boosts importance (Oja's rule), "
            "negative decreases, correct updates content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["positive", "negative", "correct"],
                    "description": "Feedback type",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Target entity",
                },
                "observation_id": {
                    "type": "string",
                    "description": "Target observation UUID",
                },
                "correction": {
                    "type": "string",
                    "description": "New content (for correct action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "cuba_vigia",
        "description": (
            "Knowledge graph analytics: summary (counts + token estimate), "
            "health (staleness, entropy, DB size), drift (chi-squared on errors), "
            "communities (Louvain), bridges (betweenness centrality)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["summary", "health", "drift", "communities", "bridges"],
                    "description": "Metric to compute",
                },
            },
            "required": ["metric"],
        },
    },
]
