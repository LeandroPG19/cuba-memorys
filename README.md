# Cuba-Memorys

[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue)](https://github.com/LeandroPG19/cuba-memorys)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)
[![Audit](https://img.shields.io/badge/audit-GO-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)
[![Tech Debt](https://img.shields.io/badge/tech%20debt-0-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server that gives AI coding assistants long-term memory with a knowledge graph, neuroscience-inspired algorithms, and anti-hallucination grounding.

13 tools with Cuban soul. Sub-millisecond handlers. Mathematically rigorous.

> [!IMPORTANT]
> **v3.0.0** — Deep Research V3: exponential decay replaces FSRS-6, dead code/columns eliminated, zero tech debt. 51 tests, 0 clippy warnings, audited GO.

---

## Why Cuba-Memorys?

AI agents forget everything between conversations. Cuba-Memorys solves this:

- **Exponential decay** — Memories fade realistically (halflife=30d), strengthen with access
- **Hebbian + BCM metaplasticity** — Self-normalizing importance via Oja's rule with EMA sliding threshold
- **Hybrid RRF fusion search** — pg_trgm + full-text + pgvector HNSW, with entropy-routed weighting (k=60)
- **Knowledge graph** — Entities, observations, typed relations with Leiden community detection
- **Anti-hallucination grounding** — Verify claims against stored knowledge with graduated confidence scoring
- **REM Sleep consolidation** — Autonomous background decay + PageRank after idle
- **Graph intelligence** — Personalized PageRank, Leiden communities, Brandes centrality, Shannon entropy
- **Error memory** — Never repeat the same mistake (anti-repetition guard)

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
| Optional ONNX BGE embeddings | Yes | No |
| Write-time dedup gate | Yes | No |
| Contradiction auto-supersede | Yes | No |
| GDPR Right to Erasure | Yes | No |
| Graceful shutdown (SIGTERM/SIGINT) | Yes | No |

---

## Quick Start

```bash
git clone https://github.com/LeandroPG19/cuba-memorys.git
cd cuba-memorys

# Start PostgreSQL
docker compose up -d

# Build Rust binary
cd rust
cargo build --release
```

Configure your AI editor (Claude Code, Cursor, Windsurf, etc.):

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba-memorys/rust/target/release/cuba-memorys",
      "env": {
        "DATABASE_URL": "postgresql://cuba:memorys2026@127.0.0.1:5488/brain"
      }
    }
  }
}
```

The server auto-creates the `brain` database and all tables on first run.

### Optional: ONNX Embeddings

For real BGE-small-en-v1.5 semantic embeddings instead of hash-based fallback:

```bash
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"
```

Without ONNX, the server uses deterministic hash-based embeddings — functional but without semantic understanding.

---

## The 13 Tools

Every tool is named after Cuban culture — memorable, professional, meaningful.

### Knowledge Graph

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_alma` | **Alma** — soul | CRUD entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Triggers Hebbian boost + access tracking. |
| `cuba_cronica` | **Cronica** — chronicle | Attach observations with **dedup gate**, **contradiction detection**, **Shannon density gating**, and **adaptive PE gating V5.1**. Supports `batch_add`. |
| `cuba_puente` | **Puente** — bridge | Typed relations (`uses`, `causes`, `implements`, `depends_on`, `related_to`). **Traverse** walks the graph. **Infer** discovers transitive paths. blake3 dedup. |

### Search & Verification

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_faro` | **Faro** — lighthouse | RRF fusion (k=60) with entropy routing and pgvector. KG-neighbor expansion for low recall. `verify` mode with source triangulation. Session-aware boost. |

### Error Memory

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_alarma` | **Alarma** — alarm | Report errors. Auto-detects patterns (>=3 similar = warning). |
| `cuba_remedio` | **Remedio** — remedy | Resolve errors with cross-reference to similar unresolved issues. |
| `cuba_expediente` | **Expediente** — case file | Search past errors. **Anti-repetition guard**: warns if similar approach failed before. |

### Sessions & Decisions

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_jornada` | **Jornada** — workday | Session tracking with goals and outcomes. Goals used for decay exemption and search boost. |
| `cuba_decreto` | **Decreto** — decree | Record architecture decisions with context, alternatives, rationale. |

### Memory Maintenance

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_zafra` | **Zafra** — sugar harvest | Consolidation: exponential decay (halflife=30d), prune, merge, summarize, pagerank, find_duplicates, export, stats. |
| `cuba_eco` | **Eco** — echo | RLHF feedback: positive (Oja boost), negative (decrease), correct (update with versioning). |
| `cuba_vigia` | **Vigia** — watchman | Analytics: summary, health (Shannon entropy), drift (chi-squared), Leiden communities, Brandes bridges. |
| `cuba_forget` | **Forget** — forget | GDPR Right to Erasure: cascading hard-delete of entity and ALL references. Irreversible. |

---

## Architecture

```
cuba-memorys/
├── docker-compose.yml           # Dedicated PostgreSQL 18 (port 5488)
├── rust/                        # v3.0.0
│   ├── src/
│   │   ├── main.rs              # mimalloc + graceful shutdown
│   │   ├── protocol.rs          # JSON-RPC 2.0 + REM daemon (4h cycle)
│   │   ├── db.rs                # sqlx PgPool (10 max, 600s idle, 1800s lifetime)
│   │   ├── schema.sql           # 5 tables, 15+ indexes, HNSW
│   │   ├── constants.rs         # Tool definitions, thresholds, enums
│   │   ├── handlers/            # 13 MCP tool handlers (1 file each)
│   │   ├── cognitive/           # Hebbian/BCM, access tracking, PE gating
│   │   ├── search/              # RRF fusion, confidence, LRU cache
│   │   ├── graph/               # Brandes centrality, Leiden, PageRank
│   │   └── embeddings/          # ONNX BGE-small (optional, spawn_blocking)
│   ├── scripts/
│   │   └── migrate_v3.sql       # v2.x -> v3.0.0 column cleanup
│   └── tests/
└── src/cuba_memorys/            # Python legacy (v1.6.0)
```

### Performance: Rust vs Python

| Metric | Python v1.6.0 | Rust v3.0.0 |
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
| `brain_observations` | Facts with provenance | 9 types, versioning, `vector(384)` (pgvector), exponential decay |
| `brain_relations` | Typed edges | 5 types, bidirectional, Hebbian strength, blake3 dedup |
| `brain_errors` | Error memory | JSONB context, synapse weight, pattern detection |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking, duration |

### Search Pipeline

**Reciprocal Rank Fusion (RRF, k=60)** with entropy-routed weighting:

| # | Signal | Source | Condition |
|---|--------|--------|-----------|
| 1 | Entities (ts_rank + trigrams + importance) | `brain_entities` | Always |
| 2 | Observations (ts_rank + trigrams + importance) | `brain_observations` | Always |
| 3 | Errors (ts_rank + trigrams + synapse_weight) | `brain_errors` | Always |
| 4 | **Vector cosine distance (HNSW)** | `brain_observations.embedding` | pgvector installed |

**Post-fusion pipeline:** Dedup -> KG-neighbor expansion -> Session boost -> GraphRAG enrichment -> Token-budget truncation -> Batch access tracking

---

## Mathematical Foundations

Built on peer-reviewed algorithms, not ad-hoc heuristics:

### Exponential Decay (V3)
```
importance_new = importance * exp(-0.693 * days_since_access / halflife)
```
halflife=30d by default. Decision/lesson observations are protected from decay. Importance directly affects search ranking (score*0.7 + importance*0.3).

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

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (**required**) |
| `ONNX_MODEL_PATH` | — | Path to BGE model directory (optional) |
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
| Tests | 106/106 passing (51 unit/smoke + 55 E2E) |
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
| **3.0.0** | Deep Research V3: exponential decay replaces FSRS-6, dead code/columns eliminated, SEC-002 fix, importance in ranking, embeddings storage on write, GraphRAG CTE fix, Opus 4.6 token optimization, zero tech debt. 106 tests (51 unit/smoke + 55 E2E), 0 clippy warnings. |
| **2.0.0** | Complete Rust rewrite. BCM metaplasticity, Leiden communities, Shannon entropy, blake3 dedup. Internal audit: GO verdict. |
| **1.6.0** | KG-neighbor expansion, embedding LRU cache, async embed rebuild, community summaries, batch access tracking |
| **1.5.0** | Token-budget truncation, post-fusion dedup, source triangulation, adaptive confidence, session-aware decay |
| **1.3.0** | Modular architecture (CC avg D->A), 87% CC reduction |
| **1.1.0** | GraphRAG, REM Sleep, conditional pgvector, 4-signal RRF |
| **1.0.0** | Initial release: 12 tools, Hebbian learning |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Perez G.**

- GitHub: [@LeandroPG19](https://github.com/LeandroPG19)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)

## Credits

Mathematical foundations: Oja (1982), Bienenstock, Cooper & Munro (1982, BCM), Cormack (2009, RRF), Brin & Page (1998, PageRank), Traag et al. (2019, Leiden), Brandes (2001), Shannon (1948), Pearson (1900, chi-squared), Friston (2023, PE gating), BAAI (2023, BGE), Malkov & Yashunin (2018, HNSW), O'Connor et al. (2020, blake3).
