# 🧠 Cuba-Memorys — Rust

[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Version](https://img.shields.io/badge/version-2.1.0-blue)](https://github.com/LeandroPG19/cuba-memorys)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-77%20pass-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)
[![Tech Debt](https://img.shields.io/badge/tech%20debt-0-brightgreen)](https://github.com/LeandroPG19/cuba-memorys)

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server written in Rust for maximum performance. Knowledge graph with neuroscience-inspired algorithms: FSRS-6 adaptive decay, Hebbian learning with BCM metaplasticity, Dual-Strength memory, Leiden community detection, BM25+ search, adaptive RRF fusion, and anti-hallucination grounding.

12 tools · 4.6MB binary · Sub-millisecond handlers · Zero tech debt

---

## Why Rust?

The Python implementation works. The Rust rewrite makes it **fast**:

| Metric | Python v1.6.0 | Rust v2.1.0 |
| ------ | :-----------: | :---------: |
| Binary size | ~50MB (venv) | **4.6MB** |
| Entity create | ~2ms | **498µs** |
| Entity get | ~3ms | **1.86ms** |
| Observation add | ~2ms | **474µs** |
| Hybrid search | <5ms | **2.52ms** |
| Analytics | <2.5ms | **958µs** |
| Memory usage | ~120MB | **~15MB** |
| Startup time | ~2s | **<100ms** |
| Dependencies | 12 Python packages | **0 runtime deps** |

---

## Quick Start

### 1. Prerequisites

- **Rust 1.93+** (edition 2024)
- **PostgreSQL 18** with `pg_trgm` and `vector` extensions

### 2. Build

```bash
cd cuba-memorys/rust

# Release build (4.6MB stripped binary)
cargo build --release

# Run tests (77 total: 66 unit + 11 smoke)
cargo test
```

### 3. Run

```bash
DATABASE_URL="postgresql://user:pass@localhost:5433/brain" \
  ./target/release/cuba-memorys
```

The server auto-creates the `brain` database schema on first run.

### 4. Configure your AI editor

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba-memorys",
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5433/brain"
      }
    }
  }
}
```

### 5. Optional: ONNX Embeddings

For real BGE-small-en-v1.5 semantic embeddings instead of hash-based fallback:

```bash
# Download model (~130MB)
bash scripts/download_model.sh

# Set environment
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"
```

Without ONNX, the server uses deterministic hash-based embeddings — functional but without semantic understanding.

---

## Architecture

```text
rust/
├── Cargo.toml                  # Dependencies + release profile
├── Dockerfile                  # Multi-stage: Builder → Tester → Production
├── .dockerignore
├── examples/
│   └── bench_handlers.rs       # Benchmark script (6 handlers)
├── scripts/
│   └── download_model.sh       # ONNX model downloader
├── src/
│   ├── main.rs                 # Entry point (mimalloc, graceful shutdown)
│   ├── lib.rs                  # Public API
│   ├── protocol.rs             # JSON-RPC 2.0 + REM daemon (4h consolidation)
│   ├── db.rs                   # sqlx PgPool (10/1/5s) + HNSW ef_search=100
│   ├── schema.sql              # v2.1.0 — 5 tables, 15+ indexes, HNSW ef_construction=128
│   ├── constants.rs            # Tool definitions, thresholds, enums (26 constants)
│   ├── handlers/               # 12 MCP tool handlers
│   │   ├── mod.rs              # dispatch() router (all 12 tools)
│   │   ├── alma.rs             # Entity CRUD + Hebbian + Dual-Strength
│   │   ├── cronica.rs          # Observations (adaptive PE gating V5.1, density, dedup)
│   │   ├── faro.rs             # Hybrid search (3-range entropy RRF, adaptive k)
│   │   ├── expediente.rs       # Error search + anti-repetition guard
│   │   ├── alarma.rs           # Error reporting + pattern detection (≥3)
│   │   ├── remedio.rs          # Error resolution + cross-reference
│   │   ├── eco.rs              # RLHF feedback (Oja's rule)
│   │   ├── decreto.rs          # Architecture decisions
│   │   ├── jornada.rs          # Session management (goals, outcomes)
│   │   ├── puente.rs           # Relations (traverse, infer, blake3 dedup)
│   │   ├── vigia.rs            # Graph analytics (Leiden communities, bridges, drift)
│   │   └── zafra.rs            # Consolidation (decay, prune, merge, export)
│   ├── cognitive/              # Neuroscience-inspired algorithms
│   │   ├── fsrs.rs             # FSRS-6 with adaptive w20 decay (per-entity sigmoid)
│   │   ├── dual_strength.rs    # Dual-Strength model (Bjork 1992)
│   │   ├── hebbian.rs          # Oja's rule + BCM metaplasticity throttle
│   │   ├── spreading.rs        # RWR spreading activation (Collins 1975)
│   │   ├── prediction_error.rs # Adaptive PE gating V5.1 (Friston free energy)
│   │   └── density.rs          # Shannon entropy gating
│   ├── embeddings/
│   │   ├── mod.rs
│   │   └── onnx.rs             # ONNX inference (ort v2) + hash fallback
│   ├── search/
│   │   ├── cache.rs            # TTL-LRU cache (O(1) lookup)
│   │   ├── rrf.rs              # Weighted RRF + adaptive k (Azure AI Search 2025)
│   │   ├── tfidf.rs            # BM25+ scoring (Lv & Zhai 2011, δ=1)
│   │   └── confidence.rs       # Graduated confidence scoring
│   └── graph/
│       ├── centrality.rs       # Betweenness centrality
│       ├── community.rs        # Leiden algorithm (Traag 2019) — wired in vigia
│       └── pagerank.rs         # Personalized PageRank
└── tests/
    ├── integration.rs          # 7 DB integration tests
    └── smoke_test.rs           # 11 smoke tests (no DB required)
```

**~5,134 LOC** across 30 Rust source files + 1 SQL schema.

---

## Key Dependencies

| Crate | Purpose |
| ----- | ------- |
| `tokio` | Async runtime (multi-threaded) |
| `sqlx` | PostgreSQL (async, compile-time checked) |
| `pgvector` | Vector similarity search (cosine, HNSW) |
| `petgraph` | Graph algorithms (PageRank, Leiden, centrality) |
| `ort` | ONNX Runtime (dynamic loading, optional) |
| `tokenizers` | HuggingFace tokenizers (BGE-small) |
| `blake3` | Cryptographic hashing (relation dedup) |
| `serde` / `serde_json` | JSON-RPC serialization |
| `lru` | O(1) LRU cache with TTL |
| `mimalloc` | Global allocator (performance) |
| `tracing` | Structured logging (JSON) |

---

## Database Schema

| Table | Purpose | Key Features |
| ----- | ------- | ------------ |
| `brain_entities` | Knowledge graph nodes | importance, FSRS stability, access_count, GIN search |
| `brain_observations` | Facts with provenance | versioning, embeddings (384d HNSW), dedup, Dual-Strength |
| `brain_relations` | Typed edges | Hebbian strength, bidirectional, blake3 hash dedup |
| `brain_errors` | Error memory | synapse weight, resolution tracking, pattern detection |
| `brain_sessions` | Working sessions | goals (JSONB), outcomes, duration tracking |

Schema v2.1.0: HNSW index with `ef_construction=128`, query-time `ef_search=100`. All tables use `UUIDv4` primary keys, `timestamptz` timestamps, and cascading deletes.

---

## 🇨🇺 The 12 Tools

| Tool | Meaning | Description |
| ---- | ------- | ----------- |
| `cuba_alma` | Soul | Entity CRUD with Hebbian + Dual-Strength boost |
| `cuba_cronica` | Chronicle | Observations with adaptive PE gating V5.1 + Shannon density |
| `cuba_puente` | Bridge | Relations: traverse, infer, blake3 dedup |
| `cuba_faro` | Lighthouse | 3-range entropy RRF + adaptive k + session-aware boosting |
| `cuba_alarma` | Alarm | Report errors (auto-detects patterns ≥3) |
| `cuba_remedio` | Remedy | Resolve errors with cross-reference |
| `cuba_expediente` | Case file | Search errors + anti-repetition guard |
| `cuba_jornada` | Workday | Session tracking with goals/outcomes |
| `cuba_decreto` | Decree | Architecture decisions (record/query/list) |
| `cuba_zafra` | Harvest | Consolidation: FSRS-6 decay, prune, merge, pagerank, export |
| `cuba_eco` | Echo | RLHF feedback (Oja positive/negative/correct) |
| `cuba_vigia` | Watchman | Analytics: summary, health, drift, Leiden communities, bridges |

---

## V2.1.0 SOTA Improvements

Nine research-backed improvements integrated into production handlers:

| # | Improvement | Reference | Wired In |
| :-: | ---------- | --------- | -------- |
| 1 | **FSRS-6 w20 adaptive decay** | Expertium (2025) | `batch_decay` via sigmoid per-entity |
| 2 | **3-range entropy RRF** | Elastic Labs (2025) | `faro.rs` → keyword/mixed/semantic routing |
| 3 | **Adaptive RRF k** | Azure AI Search (2025) | `faro.rs` + `rrf.rs` → scales with result count |
| 4 | **HNSW tuning** | Google Cloud pgvector (2025) | `schema.sql` ef_construction=128, `db.rs` ef_search=100 |
| 5 | **BM25+ (δ=1)** | Lv & Zhai (SIGIR 2011) | `tfidf.rs` monotonic scoring |
| 6 | **Leiden communities** | Traag et al. (Nature 2019) | `vigia.rs` communities → `graph::community::detect()` |
| 7 | **BCM metaplasticity** | Bienenstock, Cooper, Munro (1982) | `hebbian.rs` → access/search boost throttle |
| 8 | **Adaptive PE gating V5.1** | Friston (Nature 2023) | `cronica.rs` → `prediction_error::adaptive_gate()` |
| 9 | **blake3 relation dedup** | O'Connor et al. (2020) | `puente.rs` → triple hash on create |

All improvements are **wired in production handlers** — zero orphaned library code.

---

## REM Sleep Daemon

A background consolidation process runs every **4 hours**, executing:

1. **FSRS-6 adaptive decay** — Power-law forgetting with per-entity w20 (sigmoid of access_count)
2. **PageRank** — Recalculate importance across the knowledge graph
3. **RWR Spreading Activation** — Collins & Loftus (1975) neighbor boosting
4. **TF-IDF rebuild** — BM25+ index refresh (in-memory)

Active session entities are **protected** from decay (exact `entity_id` match, not ILIKE).

---

## Benchmark

```bash
DATABASE_URL="..." cargo run --release --example bench_handlers
```

Results on local PostgreSQL (Intel i7, NVMe):

```text
  ═══ cuba-memorys Rust Benchmark ═══

  embed (hash)           0µs /call  (4000 calls)
  alma::create       498µs /call  (100 calls)
  alma::get         1.86ms /call  (100 calls)
  cronica::add       474µs /call  (100 calls)
  faro::search      2.52ms /call  (50 calls)
  vigia::summary     958µs /call  (50 calls)
```

---

## Docker

Multi-stage Dockerfile (Builder → Tester → Production):

```bash
docker build -t cuba-memorys .
docker run -e DATABASE_URL="..." cuba-memorys
```

Features:

- Non-root user (`appuser`)
- `HEALTHCHECK` via process monitor
- `STOPSIGNAL SIGTERM` (exec form, PID 1)
- Stripped binary (4.6MB)

---

## Testing

```bash
# Unit + smoke tests (no DB required)
cargo test

# Integration tests (requires PostgreSQL)
DATABASE_URL="postgresql://..." cargo test --test integration -- --ignored
```

**77 tests total**: 66 unit + 11 smoke + 7 integration (ignored without DB).

| Category | Count | Coverage |
| -------- | :---: | -------- |
| Cognitive algorithms | 24 | FSRS-6 w20, Hebbian/BCM, Dual-Strength, PE gating, density |
| Search & scoring | 16 | BM25+, RRF adaptive k, confidence, TF-IDF |
| Graph algorithms | 9 | Leiden (community), PageRank, centrality |
| Handler logic | 6 | Cronica dedup/density, protocol validation |
| Smoke (no DB) | 11 | Constants, thresholds, invariants |
| Integration (DB) | 7 | Full handler round-trips (requires PostgreSQL) |

---

## Mathematical Foundations

| Algorithm | Reference | Application |
| --------- | --------- | ----------- |
| **FSRS-6 + w20** | Ye (2024), Expertium (2025) | Adaptive decay: `R = (1 + t/9S)^(-d)`, d=sigmoid(access) |
| **Dual-Strength** | Bjork & Bjork (1992) | Storage + retrieval strength separation |
| **BCM Metaplasticity** | Bienenstock, Cooper, Munro (1982) | Sliding threshold prevents runaway potentiation |
| **Oja's Rule** | Oja (1982) | Normalized Hebbian: `Δw = η·x·(y - w·x)` |
| **BM25+** | Lv & Zhai (SIGIR 2011) | Monotonic term scoring with δ=1 |
| **Weighted RRF** | Cormack (2009), Elastic (2025) | 3-range entropy routing + adaptive k |
| **Leiden** | Traag, Waltman, van Eck (Nature 2019) | Community detection with connectivity guarantee |
| **PageRank** | Brin & Page (1998) | Personalized importance ranking |
| **Shannon Entropy** | Shannon (1948) | Information density gating + search routing |
| **Prediction Error** | Friston (Nature 2023) | Adaptive novelty thresholds (free energy principle) |
| **Spreading Activation** | Collins & Loftus (1975) | Random Walk with Restart (RWR) neighbor boost |
| **BGE-small-en-v1.5** | BAAI (2023) | ONNX quantized 384d semantic embeddings |
| **blake3** | O'Connor et al. (2020) | Cryptographic hash for relation deduplication |
| **HNSW** | Malkov & Yashunin (2018) | Approximate nearest neighbors (pgvector index) |

---

## Code Quality

Last audit: **2026-03-11** — Zero technical debt.

| Metric | Result |
| ------ | :----: |
| `cargo build --release` warnings | **0** |
| `cargo test` pass rate | **77/77 (100%)** |
| TODO/FIXME/HACK in production code | **0** |
| `panic!`/`unreachable!`/`todo!` in prod | **0** |
| Unsafe `unwrap()` in production | **0** |
| Dead code without justification | **0** |
| Hardcoded secrets | **0** |
| SQL injection vectors | **0** |
| Orphaned library functions | **0** |
| Total LOC (src/) | **~5,134** |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

## Author

**Leandro Pérez G.**

- GitHub: [@LeandroPG19](https://github.com/LeandroPG19)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)
