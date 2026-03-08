# 🧠 Cuba-Memorys

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server that gives AI coding assistants long-term memory with a knowledge graph, Hebbian learning, and anti-hallucination grounding.

12 tools with Cuban soul. Zero manual setup. Mathematically rigorous.

---

## Why Cuba-Memorys?

AI agents forget everything between conversations. Cuba-Memorys solves this by giving them:

- **A knowledge graph** — Entities, observations, and relations that persist across sessions
- **Error memory** — Never repeat the same mistake twice (anti-repetition guard)
- **Hebbian learning** — Memories strengthen with use and fade adaptively (FSRS spaced repetition)
- **Anti-hallucination** — Verify claims against stored knowledge with graduated confidence scoring
- **Semantic search** — TF-IDF + pg_trgm + RRF fusion + optional BGE embeddings
- **Graph intelligence** — Personalized PageRank, Louvain community detection, betweenness centrality

| Feature | Cuba-Memorys | Basic Memory MCPs |
|---------|:------------:|:-----------------:|
| Knowledge graph with relations | ✅ | ❌ |
| Hebbian learning (Oja's rule) | ✅ | ❌ |
| FSRS adaptive spaced repetition | ✅ | ❌ |
| TF-IDF + RRF fusion search | ✅ | ❌ |
| Optional BGE embeddings (ONNX) | ✅ | ❌ |
| Contradiction detection | ✅ | ❌ |
| Graduated confidence scoring | ✅ | ❌ |
| Personalized PageRank | ✅ | ❌ |
| Louvain community detection | ✅ | ❌ |
| Betweenness centrality (bridges) | ✅ | ❌ |
| Shannon entropy (knowledge diversity) | ✅ | ❌ |
| Chi-squared concept drift detection | ✅ | ❌ |
| Error pattern detection + MTTR | ✅ | ❌ |
| Entity duplicate detection (SQL similarity) | ✅ | ❌ |
| Observation versioning (audit trail) | ✅ | ❌ |
| Temporal validity (valid_from/valid_until) | ✅ | ❌ |
| Write-time dedup gate | ✅ | ❌ |
| Auto-supersede contradictions | ✅ | ❌ |
| Full JSON export/backup (bounded) | ✅ | ❌ |
| Fuzzy search (typo-tolerant) | ✅ | ❌ |
| Spreading activation | ✅ | ❌ |
| Batch observations (10x fewer calls) | ✅ | ❌ |
| Entity type validation | ✅ | ❌ |
| Graceful shutdown (SIGTERM/SIGINT) | ✅ | ❌ |
| Auto-provisions its own DB | ✅ | ❌ |

---

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Docker** (for PostgreSQL)

### 2. Install

```bash
git clone https://github.com/lENADRO1910/cuba-memorys.git
cd cuba-memorys

docker compose up -d

pip install -e .

# Optional: BGE embeddings for semantic search (~130MB model)
pip install -e ".[embeddings]"
```

### 3. Configure your AI editor

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba_memorys_launcher.sh",
      "disabled": false
    }
  }
}
```

Or run directly:

```bash
DATABASE_URL="postgresql://cuba:memorys2026@127.0.0.1:5488/brain" python -m cuba_memorys
```

The server auto-creates the `brain` database and all tables on first run.

---

## 🇨🇺 Las 12 Herramientas

Every tool is named after Cuban culture — memorable, professional, and meaningful.

### Knowledge Graph

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_alma` | **Alma** — soul, essence | CRUD knowledge entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Triggers spreading activation on neighbors. |
| `cuba_cronica` | **Crónica** — chronicle | Attach observations to entities with **contradiction detection** and **dedup gate**. Supports `batch_add`. Types: `fact`, `decision`, `lesson`, `preference`, `context`, `tool_usage`. |
| `cuba_puente` | **Puente** — bridge | Connect entities with typed relations (`uses`, `causes`, `implements`, `depends_on`, `related_to`). **Traverse** walks the graph. **Infer** discovers transitive connections (A→B→C). |

### Search & Verification

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_faro` | **Faro** — lighthouse | Search memory with **hybrid** mode (TF-IDF + full-text + fuzzy + importance + freshness via RRF) or **verify** mode (anti-hallucination — returns `verified` / `partial` / `weak` / `unknown`). Session-aware: boosts results matching active goals. |

### Error Memory

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_alarma` | **Alarma** — alarm | Report errors immediately. Auto-detects patterns (≥3 similar = warning). Hebbian boosting for retrieval. |
| `cuba_remedio` | **Remedio** — remedy | Mark an error as resolved. Cross-references similar unresolved errors. |
| `cuba_expediente` | **Expediente** — case file | Search past errors/solutions. **Anti-repetition guard**: warns if a similar approach previously failed. |

### Sessions & Decisions

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_jornada` | **Jornada** — workday | Track working sessions with goals and outcomes. |
| `cuba_decreto` | **Decreto** — decree | Record architecture decisions with context, alternatives, and rationale. |

### Memory Maintenance

| Tool | Meaning | Description |
|------|---------|-------------|
| `cuba_zafra` | **Zafra** — sugar harvest 🎯 | Memory consolidation: **decay** (FSRS adaptive), **prune**, **merge**, **summarize**, **pagerank** (personalized), **find_duplicates** (SQL similarity), **export** (bounded JSON backup), **stats**. |
| `cuba_eco` | **Eco** — echo | RLHF feedback: **positive** (Oja's rule boost), **negative** (decrease), **correct** (update with versioning). |
| `cuba_vigia` | **Vigía** — watchman | Graph analytics: **summary** (counts + token estimate), **health** (staleness, Shannon entropy, DB size), **drift** (chi-squared), **communities** (Louvain), **bridges** (betweenness centrality). |

---

## Mathematical Foundations

Cuba-Memorys is built on peer-reviewed algorithms, not ad-hoc heuristics:

### FSRS Adaptive Decay — Wozniak (1987) / Ye (2022)

```
stability = fsrs_stability · (1 + decay_factor)^rating
R(t) = (1 + t/(9·S))^(-1)
```

FSRS (Free Spaced Repetition Scheduler) provides adaptive memory decay. Stability grows with successful reviews — memories that are reinforced survive longer.

### Oja's Rule (1982) — Hebbian Learning

```
Positive: Δw = η · (1 - w²)     → converges to 1.0, cannot explode
Negative: Δw = η · (1 + w²)     → converges to 0.01 (floor)
```

Where `η = 0.05`. The `w²` term provides natural saturation — self-normalizing without explicit clipping.

### TF-IDF + RRF Fusion — Salton (1975) / Cormack (2009)

```
tfidf(t, d) = tf(t, d) · log(N / df(t))
RRF(d) = Σ 1/(k + rank_i(d))     where k = 60
```

Reciprocal Rank Fusion combines multiple scoring signals (TF-IDF, full-text, fuzzy, importance, freshness) into a single ranking.

### Optional BGE Embeddings — BAAI (2023)

```
model: Qdrant/bge-small-en-v1.5-onnx-Q (quantized, ~130MB)
runtime: ONNX (no PyTorch dependency)
similarity: cosine(embed(query), embed(observation))
```

Auto-downloads on first use. Falls back to TF-IDF if not installed.

### Personalized PageRank — Brin & Page (1998)

```
PR(v) = (1-α)/N + α · Σ PR(u)/deg(u)     where α = 0.85
personalization: biased toward recently active entities
final_importance = 0.6·PR + 0.4·current_importance
```

Graph-structural importance via networkx. Personalized bias prevents cold-start domination.

### Shannon Entropy — Knowledge Diversity

```
H = -Σ p·log₂(p)
diversity_score = H / H_max
```

Measures how evenly distributed your knowledge is across entity/observation types. Score 0–1.

### Spreading Activation — Collins & Loftus (1975)

When entity X is accessed, its graph neighbors receive a small importance boost (0.6% per hop, decaying 30% per level).

### Chi-Squared Concept Drift — Pearson (1900)

```
χ² = Σ (observed - expected)² / expected
p-value via scipy.stats.chi2.sf()
```

Detects when the distribution of error types changes significantly — signals that something new is breaking.

### Contradiction Detection

```
conflict = tfidf_similarity(new, existing) > 0.7 AND has_negation(new, existing)
```

Checks for semantic overlap + negation patterns (not, never, instead of, replaced, deprecated) in both English and Spanish.

---

## Architecture

```
cuba-memorys/
├── docker-compose.yml          # Dedicated PostgreSQL (port 5488)
├── pyproject.toml              # Package metadata + optional deps
├── README.md
└── src/cuba_memorys/
    ├── __init__.py
    ├── __main__.py             # Entry point
    ├── server.py               # 12 MCP tool handlers (~1700 LOC)
    ├── db.py                   # asyncpg pool + orjson + auto-DB creation
    ├── schema.sql              # 5 tables, 15 indexes, pg_trgm, versioning
    ├── hebbian.py              # FSRS, Oja's rule, spreading activation
    ├── search.py               # LRU cache, RRF fusion, confidence scoring
    ├── tfidf.py                # TF-IDF semantic search (scikit-learn)
    └── embeddings.py           # Optional BGE embeddings (ONNX Runtime)
```

### Database Schema

| Table | Purpose | Key Features |
|-------|---------|-------------|
| `brain_entities` | Knowledge graph nodes | tsvector + pg_trgm indexes, importance ∈ [0,1], FSRS stability/difficulty |
| `brain_observations` | Facts attached to entities | 9 types, provenance, versioning, temporal validity (valid_from/valid_until) |
| `brain_relations` | Graph edges | 5 types, bidirectional delete, Hebbian strength |
| `brain_errors` | Error memory | JSONB context, synapse weight, MTTR tracking |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking |

### Search Pipeline

Cuba-Memorys uses **Reciprocal Rank Fusion (RRF)** to combine scoring signals:

**With embeddings:**
| Signal | Weight |
|--------|:------:|
| BGE embedding cosine similarity | 35% |
| Full-text search (ts_rank_cd) | 20% |
| TF-IDF cosine similarity | 15% |
| Importance (Hebbian-weighted) | 15% |
| Freshness (recency decay) | 15% |

**Without embeddings (TF-IDF fallback):**
| Signal | Weight |
|--------|:------:|
| Full-text search (ts_rank_cd) | 25% |
| TF-IDF cosine similarity | 25% |
| Fuzzy matching (pg_trgm) | 20% |
| Importance | 20% |
| Freshness | 10% |

### Dependencies

**Core:**
- `asyncpg` — PostgreSQL async driver
- `orjson` — Fast JSON serialization (handles UUID/datetime)
- `scikit-learn` — TF-IDF vectorization
- `networkx` — PageRank + Louvain + betweenness centrality
- `scipy` — Chi-squared statistical tests
- `rapidfuzz` — Entity duplicate detection
- `numpy` — Numerical operations

**Optional** (`pip install -e ".[embeddings]"`):
- `onnxruntime` — ONNX model inference
- `huggingface-hub` — Auto-download BGE model
- `tokenizers` — Fast tokenization

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |

### Docker Compose

Runs a dedicated PostgreSQL 18 Alpine instance:

- **Port**: 5488 (avoids conflicts with 5432/5433)
- **Resources**: 256MB RAM, 0.5 CPU
- **Restart**: always (auto-starts on boot)
- **Healthcheck**: `pg_isready` every 10s

---

## How It Works in Practice

### 1. The agent learns from your project

```
Agent: I learned that FastAPI endpoints must use async def with response_model.
→ cuba_alma(create, "FastAPI", technology)
→ cuba_cronica(add, "FastAPI", "All endpoints must be async def with response_model")
```

### 2. Error memory prevents repeated mistakes

```
Agent: I got IntegrityError: duplicate key on numero_parte.
→ cuba_alarma("IntegrityError", "duplicate key on numero_parte")
→ Similar error found! Solution: "Add SELECT EXISTS before INSERT with FOR UPDATE"
```

### 3. Anti-hallucination grounding

```
Agent: Let me verify this claim before responding...
→ cuba_faro("FastAPI uses Django ORM", mode="verify")
→ confidence: 0.0, level: "unknown"
→ recommendation: "No supporting evidence found. High hallucination risk."
```

### 4. Memories adapt with FSRS

```
Initial stability:    S = 1.0 (decays in ~9 days)
After 5 reviews:      S = 8.2 (decays in ~74 days)
After 20 reviews:     S = 45.0 (survives ~13 months)
```

### 5. Contradiction detection

```
Existing:  "Project uses PostgreSQL 15"
New:       "Project does NOT use PostgreSQL, it uses MySQL instead"
→ ⚠️ CONFLICT detected (similarity=0.82, negation pattern: "does NOT")
```

### 6. Graph intelligence

```
→ cuba_zafra(action="pagerank")
→ Top entities: FastAPI (0.42), SQLAlchemy (0.38), PostgreSQL (0.35)

→ cuba_vigia(metric="communities")
→ Community 1: [FastAPI, Pydantic, SQLAlchemy] — Backend cluster
→ Community 2: [React, Next.js, TypeScript] — Frontend cluster
```

---

## Verification

Tested with NEMESIS protocol (3-level) + 18 live integration tests:

```
18/18 integration tests — ZERO DEFECTS

🟢 Normal    — Entity CRUD, observe, search (hybrid + verify), relate (create + traverse),
                PageRank, communities, decay, find_duplicates, export, health + entropy
🟡 Pessimist — Nonexistent entities, unknown claims (confidence=0.0), invalid entity types
🔴 Extreme   — SQL injection, XSS payloads, 10KB content, concurrent creates,
                Unicode/emoji, bidirectional delete, bounded export
```

### Performance

| Operation | Avg latency |
|-----------|:----------:|
| Hybrid search | < 5ms |
| Analytics | < 2.5ms |
| Entity CRUD | < 1ms |
| PageRank (100 entities) | < 50ms |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Pérez G.**

- GitHub: [@lENADRO1910](https://github.com/lENADRO1910)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)

## Credits

Mathematical foundations: Wozniak (1987), Ye (2022, FSRS), Oja (1982), Salton (1975, TF-IDF), Cormack (2009, RRF), Brin & Page (1998, PageRank), Collins & Loftus (1975), Shannon (1948), Pearson (1900, χ²), Blondel et al. (2008, Louvain), BAAI (2023, BGE embeddings).
