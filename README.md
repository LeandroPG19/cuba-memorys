<!-- mcp-name: io.github.LeandroPG19/cuba-memorys -->
# Cuba-Memorys

[![CI](https://github.com/LeandroPG19/cuba-memorys/actions/workflows/ci.yml/badge.svg)](https://github.com/LeandroPG19/cuba-memorys/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cuba-memorys?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/cuba-memorys/)
[![npm](https://img.shields.io/npm/v/cuba-memorys?logo=npm&logoColor=white&label=npm)](https://www.npmjs.com/package/cuba-memorys)
[![MCP Registry](https://img.shields.io/badge/MCP_Registry-published-8A2BE2?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMiAxNWwtNS01IDEuNDEtMS40MUwxMCAxNC4xN2w3LjU5LTcuNTlMMTkgOGwtOSA5eiIvPjwvc3ZnPg==)](https://registry.modelcontextprotocol.io)
[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Audit](https://img.shields.io/badge/audit-GO-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)
[![Tech Debt](https://img.shields.io/badge/tech%20debt-0-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server that gives AI coding assistants long-term memory with a knowledge graph, neuroscience-inspired algorithms, and anti-hallucination grounding.

19 tools with Cuban soul. Sub-millisecond handlers. Mathematically rigorous.

> [!IMPORTANT]
> **v0.6.0** — Contextual Retrieval, importance priors, score breakdown, session provenance, compact format, semantic dedup, auto-tagging, Adamic-Adar link prediction, contradiction detection, prospective memory triggers, Bayesian calibration, bulk ingest, episodic memory with power-law decay, temporal search filters, and gap detection. 56 tests, 0 clippy warnings.

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="Cuba-Memorys MCP demo — AI agent session with knowledge graph, hybrid search, and graph analytics" width="700" />
</p>

---

## Why Cuba-Memorys?

AI agents forget everything between conversations. Cuba-Memorys solves this:

- **Stratified exponential decay** — Memories fade by type (facts=30d, errors=14d, context=7d), strengthen with access
- **Hebbian + BCM metaplasticity** — Self-normalizing importance via Oja's rule with EMA sliding threshold
- **Hybrid RRF fusion search** — pg_trgm + full-text + pgvector HNSW, entropy-routed weighting (k=60), temporal filters, tag filters, compact format
- **Knowledge graph** — Entities, observations, typed relations with Leiden community detection and Adamic-Adar link prediction
- **Anti-hallucination grounding** — Verify claims with graduated confidence + Bayesian calibration over time
- **Episodic memory** — Separate temporal events (Tulving 1972) with power-law decay I(t) = I₀/(1+ct)^β (Wixted 2004)
- **Contradiction detection** — Scan for semantic conflicts via embedding cosine + bilingual negation heuristics
- **Prospective memory** — Triggers that fire on entity access, session start, or error match ("remind me when X")
- **Contextual Retrieval** — Entity context prepended before embedding (Anthropic technique, +20% recall)
- **REM Sleep consolidation** — Autonomous stratified decay + PageRank + auto-prune + auto-merge + episode decay
- **Graph intelligence** — PageRank, Leiden communities, Brandes centrality, Shannon entropy, gap detection
- **Session awareness** — Provenance tracking, session diff, importance priors per observation type
- **Error memory** — Never repeat the same mistake (anti-repetition guard + pattern detection)

### Comparison

| Feature | Cuba-Memorys | Basic Memory MCPs |
| ------- | :----------: | :---------------: |
| Knowledge graph with typed relations | Yes | No |
| Exponential importance decay | Yes | No |
| Hebbian learning + BCM metaplasticity | Yes | No |
| Hybrid entropy-routed RRF fusion | Yes | No |
| KG-neighbor query expansion | Yes | No |
| GraphRAG topological enrichment | Yes | No |
| Leiden community detection | Yes | No |
| Brandes betweenness centrality | Yes | No |
| Shannon entropy analytics | Yes | No |
| Adaptive prediction error gating | Yes | No |
| Anti-hallucination verification | Yes | No |
| Error pattern detection | Yes | No |
| Session-aware search boost | Yes | No |
| REM Sleep autonomous consolidation | Yes | No |
| Multilingual ONNX embeddings (e5-small) | Yes | No |
| Episodic memory (power-law decay) | Yes | No |
| Contradiction detection | Yes | No |
| Prospective memory triggers | Yes | No |
| Bayesian confidence calibration | Yes | No |
| Link prediction (Adamic-Adar) | Yes | No |
| Auto-tagging (TF-IDF) | Yes | No |
| Contextual Retrieval (Anthropic) | Yes | No |
| Temporal search filters | Yes | No |
| Zero-config Docker auto-setup | Yes | No |
| Write-time dedup gate | Yes | No |
| Contradiction auto-supersede | Yes | No |
| GDPR Right to Erasure | Yes | No |
| Graceful shutdown (SIGTERM/SIGINT) | Yes | No |

---

## Installation

### PyPI (recommended)

```bash
pip install cuba-memorys
```

### npm

```bash
npm install -g cuba-memorys
```

### From source

```bash
git clone https://github.com/LeandroPG19/cuba-memorys.git
cd cuba-memorys/rust
cargo build --release
```

### Binary download

Pre-built binaries available at [GitHub Releases](https://github.com/LeandroPG19/cuba-memorys/releases).

---

## Quick Start

**Zero configuration required** — just install and add to your editor. Cuba-memorys automatically provisions a PostgreSQL database via Docker on first run.

> **Prerequisite**: [Docker](https://docs.docker.com/get-docker/) must be installed and running.

<details>
<summary><b>Claude Code</b></summary>

```bash
npm install -g cuba-memorys
claude mcp add cuba-memorys -- cuba-memorys
```
That's it. On first run, Cuba-memorys will:
1. Detect that no database is configured
2. Create a Docker container with PostgreSQL + pgvector
3. Initialize the schema automatically
4. Start serving 19 MCP tools

</details>

<details>
<summary><b>Cursor / Windsurf / VS Code</b></summary>

```bash
npm install -g cuba-memorys
```

Add to your MCP config (`.cursor/mcp.json`, `.windsurf/mcp.json`, or `.vscode/mcp.json`):

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "cuba-memorys"
    }
  }
}
```

No `DATABASE_URL` needed — auto-provisioned via Docker on first run.

</details>

<details>
<summary><b>Advanced: Custom PostgreSQL</b></summary>

If you already have PostgreSQL with pgvector, set the environment variable:

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "cuba-memorys",
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/brain"
      }
    }
  }
}
```

</details>

### Optional: Multilingual ONNX Embeddings

For real multilingual-e5-small semantic embeddings (94 languages, 384d) instead of hash-based fallback:

```bash
./scripts/download_model.sh  # Downloads ~113MB model
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"
```

Without ONNX, the server uses deterministic hash-based embeddings — functional but without semantic understanding. With ONNX, Contextual Retrieval prepends `[entity_type:entity_name]` to content before embedding for +20% recall.

---

## The 19 Tools

Every tool is named after Cuban culture — memorable, professional, meaningful.

### Knowledge Graph

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_alma` | **Alma** — soul | CRUD entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Hebbian boost + access tracking. Fires prospective triggers on access. |
| `cuba_cronica` | **Cronica** — chronicle | Observations with **semantic dedup**, **PE gating V5.2**, **importance priors** by type, **auto-tagging** (TF-IDF top-5 keywords), **session provenance**, **contextual embedding**. Also manages **episodic memories** (episode_add/episode_list) and **timeline** view. |
| `cuba_puente` | **Puente** — bridge | Typed relations. **Traverse** walks the graph. **Infer** discovers transitive paths. **Predict** suggests missing relations via Adamic-Adar link prediction. |
| `cuba_ingesta` | **Ingesta** — intake | Bulk knowledge ingestion: accepts arrays of observations or long text with auto-classification by paragraph. |

### Search & Verification

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_faro` | **Faro** — lighthouse | RRF fusion (k=60) with entropy routing, pgvector, temporal filters (`before`/`after`), tag filters, **score breakdown** (text/vector/importance/session), **compact format** (~35% fewer tokens), Bayesian **calibrated accuracy**. |

### Error Memory

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_alarma` | **Alarma** — alarm | Report errors. Auto-detects patterns (>=3 similar = warning). Fires prospective triggers on error match. |
| `cuba_remedio` | **Remedio** — remedy | Resolve errors with cross-reference to similar unresolved issues. |
| `cuba_expediente` | **Expediente** — case file | Search past errors. **Anti-repetition guard**: warns if similar approach failed before. |

### Sessions & Decisions

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_jornada` | **Jornada** — workday | Session tracking with goals, outcomes, **session diff** (what was learned), and **previous session** context on start. Fires prospective triggers. |
| `cuba_decreto` | **Decreto** — decree | Record architecture decisions with context, alternatives, rationale. |

### Cognition & Analysis

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_reflexion` | **Reflexion** — reflection | Gap detection: isolated entities, underconnected hubs, type silos, observation gaps, density anomalies (z-score). |
| `cuba_hipotesis` | **Hipotesis** — hypothesis | Abductive inference: given an effect, find plausible causes via backward causal traversal. Plausibility = path_strength x importance. |
| `cuba_contradiccion` | **Contradiccion** — contradiction | Scan for semantic conflicts between same-entity observations via embedding cosine + bilingual negation heuristics. |
| `cuba_centinela` | **Centinela** — sentinel | Prospective memory triggers: "remind me when X is accessed / session starts / error matches". Auto-deactivate on max_fires, expiration support. |
| `cuba_calibrar` | **Calibrar** — calibrate | Bayesian confidence calibration: track faro/verify predictions, compute P(correct\|grounding_level) via Beta distribution. Closes the verify-correct feedback loop. |

### Memory Maintenance

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_zafra` | **Zafra** — sugar harvest | Stratified decay (30d/14d/7d by type), power-law episode decay, prune, merge, summarize, pagerank, find_duplicates, export, stats, **reembed** (model migration with versioning). Auto-consolidation on >50 observations. |
| `cuba_eco` | **Eco** — echo | RLHF feedback: positive (Oja boost), negative (decrease), correct (update with versioning). |
| `cuba_vigia` | **Vigia** — watchman | Analytics: summary, **enhanced health** (null embeddings, active triggers, table sizes, embedding model), drift (chi-squared), Leiden communities, Brandes bridges. |
| `cuba_forget` | **Forget** — forget | GDPR Right to Erasure: cascading hard-delete of entity and ALL references (observations, episodes, relations, errors, sessions). Irreversible. |

---

## Architecture

```
cuba-memorys/
├── docker-compose.yml           # Dedicated PostgreSQL 18 (port 5488)
├── rust/                        # v0.6.0
│   ├── src/
│   │   ├── main.rs              # mimalloc + graceful shutdown
│   │   ├── protocol.rs          # JSON-RPC 2.0 + REM daemon (4h cycle)
│   │   ├── db.rs                # sqlx PgPool (10 max, 600s idle, 1800s lifetime)
│   │   ├── schema.sql           # 8 tables, 20+ indexes, HNSW
│   │   ├── constants.rs         # Tool definitions, thresholds, importance priors
│   │   ├── handlers/            # 19 MCP tool handlers (1 file each)
│   │   ├── cognitive/           # Hebbian/BCM, access tracking, PE gating V5.2
│   │   ├── search/              # RRF fusion, confidence, LRU cache
│   │   ├── graph/               # Brandes centrality, Leiden, PageRank (NF-IDF)
│   │   └── embeddings/          # ONNX multilingual-e5-small (contextual, spawn_blocking)
│   ├── scripts/
│   │   └── download_model.sh    # Download multilingual-e5-small ONNX
│   └── tests/
└── server.json                  # MCP Registry manifest
```

### Performance: Rust vs Python

| Metric | Python v1.6.0 | Rust v0.6.0 |
| ------ | :-----------: | :---------: |
| Binary size | ~50MB (venv) | **7.6MB** |
| Entity create | ~2ms | **498us** |
| Hybrid search | <5ms | **2.52ms** |
| Analytics | <2.5ms | **958us** |
| Memory usage | ~120MB | **~15MB** |
| Startup time | ~2s | **<100ms** |
| Dependencies | 12 Python packages | **0 runtime deps** |

### Database Schema

| Table | Purpose | Key Features |
|-------|---------|-------------|
| `brain_entities` | KG nodes | tsvector + pg_trgm + GIN indexes, importance, bcm_theta |
| `brain_observations` | Facts with provenance | 9 types, versioning, `vector(384)`, importance priors, auto-tags TEXT[], session_id FK, embedding_model tracking |
| `brain_relations` | Typed edges | 5 types, bidirectional, Hebbian strength, blake3 dedup |
| `brain_errors` | Error memory | JSONB context, synapse weight, pattern detection |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking, session diff |
| `brain_episodes` | Episodic memory | Tulving 1972, actors/artifacts TEXT[], power-law decay (Wixted 2004) |
| `brain_triggers` | Prospective memory | on_access/on_session_start/on_error_match, max_fires, expiration |
| `brain_verify_log` | Bayesian calibration | claim, confidence, grounding_level, outcome (correct/incorrect) |

### Search Pipeline

**Reciprocal Rank Fusion (RRF, k=60)** with entropy-routed weighting:

| # | Signal | Source | Condition |
|---|--------|--------|-----------|
| 1 | Entities (ts_rank + trigrams + importance) | `brain_entities` | Always |
| 2 | Observations (ts_rank + trigrams + importance) | `brain_observations` | Always |
| 3 | Errors (ts_rank + trigrams + synapse_weight) | `brain_errors` | Always |
| 4 | **Vector cosine distance (HNSW)** | `brain_observations.embedding` | pgvector installed |
| 5 | Episodes (ts_rank + trigrams + importance) | `brain_episodes` | Always |

**Post-fusion pipeline:** Dedup -> KG-neighbor expansion -> Session boost -> Score breakdown -> GraphRAG enrichment -> Token-budget truncation -> Compact format (optional) -> Batch access tracking

**Filters:** `before`/`after` (ISO8601 temporal), `tags` (keyword), `format` (verbose/compact)

---

## Mathematical Foundations

Built on peer-reviewed algorithms, not ad-hoc heuristics:

### Stratified Exponential Decay (V4)
```
importance_new = importance * exp(-0.693 * days_since_access / halflife)
```
Stratified by observation type: facts/preferences=30d, errors/solutions=14d, context/tool_usage=7d. Decision/lesson observations are protected (never decay). Episodic memories use power-law: `I(t) = 0.5 / (1 + 0.1*t)^0.5` (Wixted 2004). Importance directly affects search ranking (score*0.7 + importance*0.3).

### Hebbian + BCM — Oja (1982), Bienenstock-Cooper-Munro (1982)
```
Positive: importance += eta * throttle(access_count, theta_M)
BCM EMA: theta_M = max(10, (1-alpha)*theta_prev + alpha*access_count)
```
V3: theta_M persisted in `bcm_theta` column for true temporal smoothing.

### RRF Fusion — Cormack (2009)
```
RRF(d) = sum( w_i / (k + rank_i(d)) )   where k = 60
```
Entropy-routed weighting: keyword-dominant vs mixed vs semantic queries get different signal weights.

### Other Algorithms

| Algorithm | Reference | Used in |
|-----------|-----------|---------|
| **Leiden communities** | Traag et al. (Nature 2019) | `community.rs` -> `vigia.rs` |
| **Personalized PageRank** | Brin & Page (1998) | `pagerank.rs` -> `zafra.rs` |
| **Brandes centrality** | Brandes (2001) | `centrality.rs` -> `vigia.rs` |
| **Adaptive PE gating** | Friston (Nature 2023) | `prediction_error.rs` -> `cronica.rs` |
| **Shannon entropy** | Shannon (1948) | `density.rs` -> information gating |
| **Chi-squared drift** | Pearson (1900) | Error distribution change detection |
| **Power-law forgetting** | Wixted (2004) | `setup.rs` -> episodic memory decay |
| **Contextual Retrieval** | Anthropic (2024) | `onnx.rs` -> entity context prepend |
| **Adamic-Adar** | Adamic & Adar (2003) | `puente.rs` -> link prediction |
| **Episodic/Semantic** | Tulving (1972) | `brain_episodes` vs `brain_observations` |
| **Bayesian calibration** | Beta distribution | `calibrar.rs` -> P(correct\|level) |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (auto-provisioned via Docker if not set) |
| `ONNX_MODEL_PATH` | — | Path to multilingual-e5-small model directory (optional) |
| `ORT_DYLIB_PATH` | — | Path to libonnxruntime.so (optional) |
| `RUST_LOG` | `cuba_memorys=info` | Log level filter |

### Docker Compose

Dedicated PostgreSQL 18 Alpine:

- **Port**: 5488 (avoids conflicts with 5432/5433)
- **Resources**: 256MB RAM, 0.5 CPU
- **Restart**: always
- **Healthcheck**: `pg_isready` every 10s

---

## How It Works

### 1. The agent learns from your project

```
Agent: FastAPI requires async def with response_model.
-> cuba_alma(create, "FastAPI", technology)
-> cuba_cronica(add, "FastAPI", "All endpoints must be async def with response_model")
```

### 2. Error memory prevents repeated mistakes

```
Agent: IntegrityError: duplicate key on numero_parte.
-> cuba_alarma("IntegrityError", "duplicate key on numero_parte")
-> cuba_expediente: Similar error found! Solution: "Add SELECT EXISTS before INSERT"
```

### 3. Anti-hallucination grounding

```
Agent: Let me verify before responding...
-> cuba_faro("FastAPI uses Django ORM", mode="verify")
-> confidence: 0.0, level: "unknown" — "No evidence. High hallucination risk."
```

### 4. Memories decay naturally

```
Initial importance:    0.5  (new observation)
After 30d no access:  0.25 (halved by exponential decay)
After 60d no access:  0.125
Active access resets the clock — frequently used memories stay strong.
```

### 5. Community intelligence

```
-> cuba_vigia(metric="communities")
-> Community 0 (4 members): [FastAPI, Pydantic, SQLAlchemy, PostgreSQL]
  Summary: "Backend stack: async endpoints, V2 validation, 2.0 ORM..."
-> Community 1 (3 members): [React, Next.js, TypeScript]
  Summary: "Frontend stack: React 19, App Router, strict types..."
```

---

## Security & Audit

**Internal Audit Verdict: GO** (2026-03-28)

| Check | Result |
|-------|:------:|
| SQL injection | All queries parameterized (sqlx bind) |
| SEC-002 wildcard injection | Fixed (POSITION-based) |
| CVEs in dependencies | 0 active (sqlx 0.8.6, tokio 1.50.0) |
| UTF-8 safety | `safe_truncate` on all string slicing |
| Secrets | All via environment variables |
| Division by zero | Protected with `.max(1e-9)` |
| Error handling | All `?` propagated with `anyhow::Context` |
| Clippy | 0 warnings |
| Tests | 56 passing (43 unit + 13 smoke) + 49 E2E |
| Licenses | All MIT/Apache-2.0 (0 GPL/AGPL) |

---

## Dependencies

| Crate | Purpose | License |
|-------|---------|---------|
| `tokio` | Async runtime | MIT |
| `sqlx` | PostgreSQL (async) | MIT/Apache-2.0 |
| `serde` / `serde_json` | Serialization | MIT/Apache-2.0 |
| `pgvector` | Vector similarity | MIT |
| `ort` | ONNX Runtime (optional) | MIT/Apache-2.0 |
| `tokenizers` | HuggingFace tokenizers | Apache-2.0 |
| `blake3` | Cryptographic hashing | Apache-2.0/CC0 |
| `mimalloc` | Global allocator | MIT |
| `tracing` | Structured JSON logging | MIT |
| `lru` | O(1) LRU cache | MIT |
| `chrono` | Timezone-aware timestamps | MIT/Apache-2.0 |

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **0.6.0** | Contextual Retrieval (+20% recall), importance priors, score breakdown, compact format (~35% fewer tokens), session provenance/diff, semantic dedup, auto-tagging (TF-IDF), Adamic-Adar link prediction, bulk ingest (cuba_ingesta), enhanced health metrics, partial indexes, embedding model versioning. Auto Docker PostgreSQL setup. 19 tools, 56 tests. |
| **0.5.0** | Temporal reasoning (before/after/timeline), contradiction detection (cosine + negation heuristics), prospective memory triggers (centinela), Bayesian calibration (calibrar), abductive inference (hipotesis), gap detection (reflexion). 18 tools. |
| **0.4.0** | Multilingual embeddings (e5-small, 94 languages), episodic memory (Tulving 1972, power-law Wixted 2004), stratified decay (30d/14d/7d by type), E2E tests in CI with PostgreSQL. 15 tools. |
| **0.3.0** | Deep Research V3: exponential decay replaces FSRS-6, dead code eliminated, SEC-002 fix, embeddings storage on write, GraphRAG CTE fix. 13 tools. |
| **0.2.0** | Complete Rust rewrite. BCM metaplasticity, Leiden communities, Shannon entropy, blake3 dedup. |
| **1.0-1.6** | Python era: 12 tools, Hebbian learning, GraphRAG, REM Sleep, token-budget truncation. |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Perez G.**

- GitHub: [@LeandroPG19](https://github.com/LeandroPG19)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)

## Credits

Mathematical foundations: Oja (1982), Bienenstock, Cooper & Munro (1982, BCM), Cormack (2009, RRF), Brin & Page (1998, PageRank), Traag et al. (2019, Leiden), Brandes (2001), Shannon (1948), Pearson (1900, chi-squared), Friston (2023, PE gating), Tulving (1972, episodic memory), Wixted (2004, power-law forgetting), Adamic & Adar (2003, link prediction), Anthropic (2024, Contextual Retrieval), Wang et al. (2022, E5 embeddings), Malkov & Yashunin (2018, HNSW), O'Connor et al. (2020, blake3).
