# Cuba-Memorys — Rust

[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue)](https://github.com/LeandroPG19/cuba-memorys)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-51%20pass-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)
[![Audit](https://img.shields.io/badge/audit-GO-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)
[![Tech Debt](https://img.shields.io/badge/tech%20debt-0-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)

**Persistent memory for AI agents** — Complete Rust rewrite of the Cuba-Memorys MCP server. Knowledge graph with neuroscience-inspired algorithms: exponential decay, Hebbian learning with BCM metaplasticity, Leiden community detection, hybrid search (RRF + pgvector), and anti-hallucination grounding.

13 tools · 7.6MB binary · Sub-millisecond handlers · Zero tech debt · Audited GO

---

## Quick Start

### 1. Prerequisites

- **Rust 1.93+** (edition 2024)
- **PostgreSQL 18** with `pg_trgm` extension (and optionally `vector`)

### 2. Build

```bash
cd cuba-memorys/rust

# Release build
cargo build --release

# Run tests (40 unit + 11 smoke)
cargo test

# Run E2E tests against real DB (55 tests, all 13 tools)
python3 tests/e2e_all_tools.py
```

### 3. Run

```bash
DATABASE_URL="postgresql://user:pass@localhost:5488/brain" \
  ./target/release/cuba-memorys
```

The server auto-creates the database schema on first run.

### 4. Configure your AI editor

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba-memorys",
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5488/brain"
      }
    }
  }
}
```

### 5. Optional: ONNX Embeddings

For real BGE-small-en-v1.5 semantic embeddings:

```bash
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"
```

Without ONNX, the server uses deterministic hash-based embeddings — functional but without semantic understanding.

---

## Performance

| Metric | Python v1.6.0 | Rust v3.0.0 |
| ------ | :-----------: | :---------: |
| Binary size | ~50MB (venv) | **7.6MB** |
| Entity create | ~2ms | **498us** |
| Entity get | ~3ms | **1.86ms** |
| Observation add | ~2ms | **474us** |
| Hybrid search | <5ms | **2.52ms** |
| Analytics | <2.5ms | **958us** |
| Memory usage | ~120MB | **~15MB** |
| Startup time | ~2s | **<100ms** |
| Dependencies | 12 Python packages | **0 runtime deps** |

---

## Architecture

```
rust/
├── Cargo.toml
├── src/
│   ├── main.rs              # Entry point (mimalloc, graceful shutdown)
│   ├── lib.rs               # Public API
│   ├── protocol.rs          # JSON-RPC 2.0 + REM daemon (4h consolidation)
│   ├── db.rs                # sqlx PgPool (10 max, 600s idle, 1800s lifetime)
│   ├── schema.sql           # 5 tables, 15+ indexes, HNSW
│   ├── constants.rs         # Tool definitions, thresholds, enums
│   ├── handlers/            # 13 MCP tool handlers
│   │   ├── mod.rs           # dispatch() router
│   │   ├── alma.rs          # Entity CRUD + Hebbian boost + access tracking
│   │   ├── cronica.rs       # Observations (adaptive PE gating V5.1, Shannon density)
│   │   ├── faro.rs          # Hybrid search (entropy-routed RRF k=60, pgvector)
│   │   ├── expediente.rs    # Error search + anti-repetition guard
│   │   ├── alarma.rs        # Error reporting + pattern detection
│   │   ├── remedio.rs       # Error resolution + cross-reference
│   │   ├── eco.rs           # RLHF feedback (Oja's rule)
│   │   ├── decreto.rs       # Architecture decisions
│   │   ├── jornada.rs       # Session management
│   │   ├── puente.rs        # Relations (CTE traverse, infer, blake3 dedup)
│   │   ├── vigia.rs         # Analytics (Leiden, bridges, drift, health)
│   │   ├── zafra.rs         # Consolidation (exponential decay, prune, merge)
│   │   └── forget.rs        # GDPR Right to Erasure (cascading hard-delete)
│   ├── cognitive/           # Neuroscience-inspired algorithms
│   │   ├── dual_strength.rs # Access tracking (last_accessed + access_count)
│   │   ├── hebbian.rs       # Oja's rule + BCM EMA metaplasticity throttle
│   │   ├── prediction_error.rs # Adaptive PE gating V5.1 (Friston)
│   │   └── density.rs       # Shannon entropy information gating
│   ├── embeddings/
│   │   └── onnx.rs          # ONNX BGE-small (ort v2) + hash fallback + LRU cache
│   ├── search/
│   │   ├── rrf.rs           # Weighted RRF k=60 (Cormack 2009)
│   │   ├── confidence.rs    # Graduated confidence (normalized weights)
│   │   └── cache.rs         # TTL-LRU cache (O(1) lookup)
│   └── graph/
│       ├── centrality.rs    # Brandes betweenness centrality
│       ├── community.rs     # Leiden algorithm (Traag 2019)
│       └── pagerank.rs      # Personalized PageRank (alpha=0.85)
├── scripts/
│   ├── download_model.sh    # ONNX model downloader
│   └── migrate_v3.sql       # v2.x -> v3.0.0 column cleanup
└── tests/
    ├── smoke_test.rs        # 11 smoke tests (no DB required)
    └── e2e_all_tools.py     # 55 E2E tests (all 13 tools vs real DB)
```

---

## Key Algorithms

| # | Algorithm | Reference | Implementation |
|:-:|-----------|-----------|----------------|
| 1 | **Exponential decay** (halflife=30d) | Standard forgetting curve | `zafra.rs` decay action |
| 2 | **Hebbian + BCM EMA** (sliding theta) | Oja (1982), BCM (1982) | `hebbian.rs` -> access boost |
| 3 | **Entropy-routed RRF** (k=60) | Cormack (2009), Azure AI (2025) | `rrf.rs` -> `faro.rs` |
| 4 | **Leiden communities** (3-phase) | Traag et al. (Nature 2019) | `community.rs` -> `vigia.rs` |
| 5 | **Personalized PageRank** | Brin & Page (1998) | `pagerank.rs` -> `zafra.rs` |
| 6 | **Brandes centrality** | Brandes (2001) | `centrality.rs` -> `vigia.rs` bridges |
| 7 | **Adaptive PE gating** (z-score) | Friston (Nature 2023) | `prediction_error.rs` -> `cronica.rs` |
| 8 | **Shannon entropy** | Shannon (1948) | `density.rs` -> information quality |
| 9 | **blake3 triple hash** | O'Connor et al. (2020) | `puente.rs` -> relation dedup |
| 10 | **pgvector HNSW** (384d BGE-small) | Malkov & Yashunin (2018) | `onnx.rs` -> `faro.rs` vector search |

All algorithms are **wired in production handlers** — zero orphaned library code.

---

## REM Sleep Daemon

Background consolidation runs every **4 hours**:

1. **Exponential decay** — importance halves every 30d of no access (decision/lesson protected)
2. **PageRank** — Recalculate importance across the graph

Active session entities are **protected** from decay.

---

## Database Schema

| Table | Purpose | Key Features |
|-------|---------|-------------|
| `brain_entities` | KG nodes | importance, bcm_theta, access_count, GIN search |
| `brain_observations` | Facts with provenance | versioning, vector(384) HNSW, exponential decay |
| `brain_relations` | Typed edges | Hebbian strength, bidirectional, blake3 dedup |
| `brain_errors` | Error memory | synapse weight, pattern detection |
| `brain_sessions` | Working sessions | goals (JSONB), outcome tracking |

All tables: UUIDv4 PKs, `timestamptz` timestamps, cascading deletes. HNSW: `ef_construction=128`, runtime `ef_search=100`.

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime (multi-threaded) |
| `sqlx` | PostgreSQL (async, compile-time checked) |
| `pgvector` | Vector similarity search (cosine, HNSW) |
| `ort` | ONNX Runtime (dynamic loading, optional) |
| `tokenizers` | HuggingFace tokenizers (BGE-small) |
| `blake3` | Cryptographic hashing (relation dedup) |
| `serde` / `serde_json` | JSON-RPC serialization |
| `lru` | O(1) LRU cache with TTL |
| `mimalloc` | Global allocator |
| `tracing` | Structured JSON logging |

All dependencies: MIT/Apache-2.0. Zero GPL/AGPL.

---

## Security & Audit

**Internal Audit Verdict: GO** (2026-03-28)

- 0 active CVEs (sqlx 0.8.6, tokio 1.50.0 — both patched)
- All SQL queries parameterized (sqlx bind)
- SEC-002: ILIKE wildcard injection fixed (POSITION-based)
- `safe_truncate` on all UTF-8 string slicing
- Secrets via environment variables only
- 0 clippy warnings
- 106/106 tests passing (51 unit/smoke + 55 E2E)

---

## Docker

Multi-stage Dockerfile (Builder -> Tester -> Production):

```bash
docker build -t cuba-memorys .
docker run -e DATABASE_URL="..." cuba-memorys
```

- Non-root user (`appuser`)
- `HEALTHCHECK` via process monitor
- `STOPSIGNAL SIGTERM` (exec form, PID 1)
- Stripped binary (7.6MB)

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.
