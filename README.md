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

25 tools with Cuban soul. Sub-millisecond handlers. Mathematically rigorous.

> [!IMPORTANT]
> **v0.10.0** (2026-06-04) — Knowledge-graph memory plane on top of the v0.9 hybrid stack. **No breaking MCP API changes** for existing clients.
> **Bitemporal facts** (`brain_facts`, migration `0018`): every `cuba_cronica` add/batch_add and `cuba_ingesta` ingest mirrors into valid-time rows — **on by default** (`CUBA_BITEMPORAL=0` to disable).
> **Graph metrics** (`0022`–`0023`): `brain_node_metrics` (PageRank, energy, betweenness) + `brain_communities`; `cuba_zafra pagerank` / **`communities`** persist; `cuba_vigia` communities metric writes tags.
> **Spreading activation** enriches `cuba_puente predict` alongside Adamic-Adar.
> **Eval harness** (`rust/src/eval/`): nDCG@k, MRR, P@k, R@k over live `cuba_faro` hybrid (production path unchanged).
> **Unified search view** (`v_unified_memory_search`, `0024`) joins facts via `brain_entities` — never `fact_id = node_id`.
> **Shipped**: Cargo/npm `0.10.0`, PyPI `1.12.0`, GitHub Release `v0.10.0`, MCP Registry. QA: `./scripts/merge-gate.sh` (118 unit + smoke, E2E 73, MCP live 25).

> [!NOTE]
> **v0.9.x** — BM25 3-way RRF, MMR, OOD abstention, conformal PE gating, testing-effect decay, tiktoken budget, cross-encoder reranker (`CUBA_RERANKER_PATH`), `cuba_archivo` audit chain, `cuba_pizarra` working memory. **25 sqlx migrations** (`0001`–`0025`), bootstrap transparente para DBs legacy.

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="Cuba-Memorys v0.10.0 terminal demo — 25 MCP tools, bitemporal facts, hybrid BM25+RRF search, PageRank, Leiden communities, session lifecycle" width="700" />
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
- **LLM-judge for ambiguous contradictions** *(v0.8)* — Escalate cosine 0.6-0.8 pairs to Claude Code CLI subprocess (`$0` with subscription) or Anthropic API (feature flag). Verdicts cached permanently
- **Prospective memory** — Triggers that fire on entity access, session start, or error match ("remind me when X")
- **Contextual Retrieval** — Entity context prepended before embedding (Anthropic technique, +20% recall)
- **REM Sleep consolidation** — Autonomous stratified decay + PageRank + auto-prune + auto-merge + episode decay
- **Graph intelligence** — PageRank, Leiden communities, Brandes centrality, Shannon entropy, gap detection
- **Session awareness** — Provenance tracking, session diff, importance priors per observation type
- **Project scoping** *(v0.8)* — Isolate memories per project (`cuba_jornada start --project NAME`); legacy NULL rows stay globally visible (zero-regression upgrade path)
- **Compaction-survival snapshots** *(v0.8)* — `cuba_pre_compact snapshot` persists session state before `/compact`; `restore` re-injects post-compact
- **Git-friendly sync** *(v0.8)* — `cuba_sync export` writes 1 JSON per entity (diff-able in PR review), `import` is idempotent via `ON CONFLICT DO NOTHING`, optional zstd embeddings blob
- **BM25 hybrid 3-way fusion** *(v0.9)* — text + vector + BM25 (`ts_rank_cd`) en una sola RRF (Robertson-Walker 1994 baseline), captura queries con términos raros que dense embeddings pierden
- **MMR diversification** *(v0.9)* — `cuba_faro diversify=true` aplica Carbonell-Goldstein 1998 con Jaccard sim entre candidatos, evita top-K redundantes
- **OOD abstention** *(v0.9)* — `cuba_faro abstain_ood=true` con Mahalanobis ridge-regularized Σ⁻¹ (Lee NeurIPS 2018), retorna abstención formal en lugar de matches espurios
- **Conformal prediction** *(v0.9)* — quantiles empíricos sin asumir normalidad (Vovk 2005, Angelopoulos-Bates 2023); captura anisotropía cosine documentada por Ethayarajh 2019
- **Testing effect decay** *(v0.9)* — halflife escalado por `(1 + ln(1+access_count))` (Karpicke-Roediger Science 2008); high-access obs decae 4-5× más lento
- **Hebbian Δt-aware** *(v0.9)* — burst suppression `boost *= (1 - exp(-Δt/τ))`, τ=600s; anti-saturación inspirada en STDP triplet rules (Pfister-Gerstner 2006)
- **Robbins-Monro stochastic LR** *(v0.9)* — `η = 0.05/√(1 + access_count/100)` en Oja's rule, convergencia O(1/√t)
- **Source credibility tracking** *(v0.9)* — Beta(α,β) Bayesian update per source en `brain_source_trust` (Yin-Han-Yu IEEE TKDE 2008), action `cuba_calibrar trust`
- **sqlx-migrate** — 25 migraciones SQL versionadas en `rust/migrations/` (`0001`–`0025`), bootstrap transparente para DBs legacy
- **Bitemporal facts** *(v0.10)* — `brain_facts` + supersede chain; mirrors observations on write (default on)
- **Graph energy & communities** *(v0.10)* — persisted PageRank/energy in `brain_node_metrics`; Leiden → `brain_communities`
- **Spreading activation** *(v0.10)* — multi-hop graph propagation for link prediction hints
- **Retrieval benchmarks** *(v0.10)* — `eval/` harness measures live `cuba_faro` (nDCG, MRR, P@k, R@k)
- **CFR-21 audit log** *(v0.9)* — `cuba_archivo` hash-chain tamper evidence
- **Working memory buffer** *(v0.9)* — `cuba_pizarra` scratchpad per session
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
| Project scoping (per-project isolation) *(v0.8)* | Yes | No |
| Compaction-survival snapshots *(v0.8)* | Yes | No |
| Git-friendly export/import *(v0.8)* | Yes | No |
| LLM-judge for ambiguous contradictions *(v0.8)* | Yes | No |
| BM25 + vector + text 3-way RRF *(v0.9)* | Yes | No |
| MMR diversification *(v0.9)* | Yes | No |
| OOD abstention via Mahalanobis *(v0.9)* | Yes | No |
| Conformal prediction (distribution-free) *(v0.9)* | Yes | No |
| Testing-effect decay *(v0.9)* | Yes | No |
| Hebbian Δt-aware burst suppression *(v0.9)* | Yes | No |
| Robbins-Monro stochastic LR *(v0.9)* | Yes | No |
| Source credibility tracking Beta(α,β) *(v0.9)* | Yes | No |
| sqlx-migrate versioned migrations *(v0.9+)* | Yes | No |
| Exact tiktoken token budget *(v0.9)* | Yes | No |
| Bitemporal fact store *(v0.10)* | Yes | No |
| Persisted graph metrics & communities *(v0.10)* | Yes | No |
| Spreading activation link hints *(v0.10)* | Yes | No |
| Built-in faro eval harness *(v0.10)* | Yes | No |

---

## Installation

### PyPI (recommended)

```bash
pip install cuba-memorys==1.12.0   # wheels bundle the Rust binary (v0.10.0)
```

### npm

```bash
npm install -g cuba-memorys@0.10.0   # downloads binary from GitHub Release on postinstall
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
4. Start serving 25 MCP tools

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
./rust/scripts/download_model.sh  # Downloads ~113MB model
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"
```

Without ONNX, the server uses deterministic hash-based embeddings — functional but without semantic understanding. With ONNX, Contextual Retrieval prepends `[entity_type:entity_name]` to content before embedding for +20% recall.

---

## The 25 Tools

Every tool is named after Cuban culture — memorable, professional, meaningful.

### Knowledge Graph

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_alma` | **Alma** — soul | CRUD entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Hebbian boost + access tracking. Fires prospective triggers on access. |
| `cuba_cronica` | **Cronica** — chronicle | Observations with **semantic dedup**, **PE gating**, **importance priors**, **auto-tagging**, **session provenance**, **contextual embedding**. **Bitemporal mirror** to `brain_facts` on add/batch_add *(v0.10, default on)*. Episodic memory (`episode_add`/`episode_list`) and **timeline**. |
| `cuba_puente` | **Puente** — bridge | Typed relations. **Traverse**, **infer**, **predict** — Adamic-Adar + **spreading activation** neighbors *(v0.10)*. |
| `cuba_ingesta` | **Ingesta** — intake | Bulk knowledge ingestion: accepts arrays of observations or long text with auto-classification by paragraph. |

### Search & Verification

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_faro` | **Faro** — lighthouse | RRF fusion (k=60) with **sigmoid entropy routing**, pgvector, temporal filters (`before`/`after`), tag filters, **score breakdown** (text/vector/importance/session), **compact format** (~35% fewer tokens), **hybrid verify** (trigram + embedding fusion), Bayesian **calibrated accuracy**, token-budget truncation, `max_tokens` control. |

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
| `cuba_zafra` | **Zafra** — sugar harvest | Stratified decay, episode decay, prune, merge, summarize, **pagerank** (persists `brain_node_metrics` + energy refresh), **`communities`** (Leiden persist), find_duplicates, export, stats, **reembed**. Auto-consolidation on >50 observations. |
| `cuba_eco` | **Eco** — echo | RLHF feedback: positive (Oja boost), negative (decrease), correct (update with versioning). |
| `cuba_vigia` | **Vigia** — watchman | Summary, **health**, drift (chi-squared), **communities** (detect + persist), Brandes bridges. |
| `cuba_forget` | **Forget** — forget | GDPR Right to Erasure: cascading hard-delete of entity and ALL references (observations, episodes, relations, errors, sessions). Irreversible. |

### v0.8 — Engram-inspired additions

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_proyecto` | **Proyecto** — project | Per-project isolation. `switch` upserts a project and binds it to the active session; reads/writes auto-scope via `project_id`. Legacy NULL rows stay globally visible. Actions: `list / current / switch / stats / rename / merge`. |
| `cuba_pre_compact` | **Pre-compact** | Survives `/compact`. `snapshot` persists session state (recent obs, decisions, unresolved errors, pending embeddings, goals) into `brain_compaction_snapshots`. `restore` returns the latest snapshot for the active session. |
| `cuba_sync` | **Sync** | Git-friendly export/import. Writes 1 JSON per entity + monthly-partitioned episodes + decisions + relations.json + manifest.json (sha hash). `import` is idempotent via `ON CONFLICT DO NOTHING`. Optional `embeddings.bin.zst` blob (off by default — re-embed on import). |
| `cuba_juez` | **Juez** — judge | LLM-judge for ambiguous (cosine 0.6-0.8) contradictions. Trait `ContradictionJudge` with three backends: `ClaudeCodeJudge` (subprocess, $0 with subscription), `AnthropicApiJudge` (feature `anthropic-api`), `HeuristicJudge` (fallback). Verdicts cached in `brain_judgments` (UNIQUE per pair). |

### v0.8 environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CUBA_PROJECT_FILTER` | _(unset)_ | Set to `off` to disable project scoping (admin/debug). |
| `CUBA_SYNC_DIR` | `./.cuba-memorys` | Root for `cuba_sync` export/import. |
| `CUBA_JUDGE` | `auto` | Judge backend: `claude_cli` / `anthropic_api` / `heuristic` / `auto`. |
| `CUBA_JUEZ_CLI` | `claude` | Subprocess CLI for `ClaudeCodeJudge`. |
| `CUBA_JUEZ_MODEL` | `claude-haiku-4-5` | Model passed to the judge backend. |
| `CUBA_JUEZ_TIMEOUT_SECS` | `30` | Subprocess/HTTP timeout. |
| `CUBA_JUEZ_MAX_PAIRS` | `5` | Cap on pairs `cuba_juez scan_entity` will escalate per call. |
| `ANTHROPIC_API_KEY` | _(unset)_ | Required for `AnthropicApiJudge` (only when feature `anthropic-api` is built in). |

### v0.9–v0.10 additional tools

| Tool | Meaning | What it does |
|------|---------|-------------|
| `cuba_archivo` | **Archivo** — archive | CFR-21 inspired **hash-chain audit log**: append, verify integrity, tail. Tamper-evident session/ops trail. |
| `cuba_pizarra` | **Pizarra** — chalkboard | **Working memory** scratchpad: write/read/clear short-lived notes bound to the active session. |

### v0.10 environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CUBA_BITEMPORAL` | **on** (unset) | Set to `0`/`false`/`off` to stop mirroring observations into `brain_facts`. |
| `CUBA_RERANKER_PATH` | _(unset)_ | Directory with bge-reranker-v2-m3 ONNX + tokenizer; enables cross-encoder rerank in `cuba_faro`. |

---

## Architecture

```
cuba-memorys/
├── docker-compose.yml           # Dedicated PostgreSQL 18 (port 5488)
├── server.json                  # MCP Registry manifest (npm 0.10.0 / PyPI 1.12.0)
├── pyproject.toml               # Maturin (bindings = "bin") — PyPI wheel
├── package.json                 # npm wrapper → GitHub Release binary
├── scripts/                     # merge-gate, backup/restore, MCP live session test
└── rust/                        # v0.10.0
    ├── migrations/              # sqlx-migrate 0001–0025 (bitemporal, graph, views)
    ├── src/
    │   ├── main.rs              # mimalloc + graceful shutdown
    │   ├── lib.rs               # Library surface (handlers, eval, graph, core)
    │   ├── protocol.rs          # JSON-RPC 2.0 + MCP correlator / sampling / cancel
    │   ├── core/                # bitemporal, entity_linking, temporal_query
    │   ├── eval/                # faro benchmark harness (nDCG, MRR, P@k, R@k)
    │   ├── handlers/            # 25 MCP tool handlers
    │   ├── cognitive/           # Hebbian, conformal PE, calibration, judge
    │   ├── search/              # RRF, BM25, MMR, OOD, rerank, tiktoken budget
    │   ├── graph/               # PageRank, Leiden, energy, activation, k-core
    │   └── embeddings/          # ONNX e5-small (contextual), hash fallback
    ├── scripts/download_model.sh
    └── tests/                   # unit + smoke + integration + e2e_all_tools.py
```

### Performance: Rust vs Python

| Metric | Python v1.6.0 | Rust v0.7.0 |
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
| `brain_relations` | Typed edges | 5 types, bidirectional, Hebbian strength, ON CONFLICT dedup |
| `brain_errors` | Error memory | JSONB context, synapse weight, pattern detection |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking, session diff |
| `brain_episodes` | Episodic memory | Tulving 1972, actors/artifacts TEXT[], power-law decay (Wixted 2004) |
| `brain_triggers` | Prospective memory | on_access/on_session_start/on_error_match, max_fires, expiration |
| `brain_verify_log` | Bayesian calibration | claim, confidence, grounding_level, outcome (correct/incorrect) |
| `brain_facts` *(v0.10)* | Bitemporal semantics | subject/predicate/object, valid_from/to, `is_current`, entity FK |
| `brain_fact_supersedes` *(v0.10)* | Fact lineage | old_fact_id → new_fact_id + reason |
| `brain_node_metrics` *(v0.10)* | Graph analytics | pagerank_score, energy_score, betweenness, community_id |
| `brain_communities` *(v0.10)* | Leiden clusters | community_name, algorithm_version, modularity |
| `v_unified_memory_search` *(v0.10)* | MCP search view | Observations + facts joined via `brain_entities` |

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
Entropy-routed weighting via smooth sigmoid (Jaynes 1957 MaxEnt):
```
t = sigmoid(2.0 * (entropy - 2.75))
text_w  = 0.7 - 0.4*t    (keyword-heavy → balanced → semantic-heavy)
vector_w = 0.3 + 0.4*t   (always sums to 1.0)
```
Replaces V2 step function which had 40% relative jumps at thresholds.

### PageRank Blend — Brin & Page (1998)
```
importance_new = 0.3 * rank_normalized + 0.7 * importance_existing
```
Min-max normalized ranks blended via convex combination (α=0.3). Preserves Hebbian/BCM/RLHF accumulated importance instead of overwriting. Uniform distribution guard: skips blend when all ranks are equal (no structural signal).

### Other Algorithms

| Algorithm | Reference | Used in |
|-----------|-----------|---------|
| **Leiden communities** | Traag et al. (Nature 2019) | `community.rs` -> `vigia.rs` |
| **PageRank + blend** | Brin & Page (1998) | `pagerank.rs` -> convex combination (α=0.3) |
| **Brandes centrality** | Brandes (2001) | `centrality.rs` -> undirected normalization |
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
| `CUBA_BITEMPORAL` | on | Bitemporal mirror on write (`0` to disable) |
| `CUBA_RERANKER_PATH` | — | bge-reranker-v2-m3 ONNX directory for `cuba_faro` rerank |
| `CUBA_JUDGE` | `auto` | Contradiction judge backend (`heuristic` / `claude_cli` / …) |

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

## Release & QA (v0.10)

Pre-merge / pre-release gate (requires Postgres on `:5488`):

```bash
export DATABASE_URL=postgresql://cuba:memorys2026@127.0.0.1:5488/brain
./scripts/merge-gate.sh          # fmt, clippy, unit+integration, E2E 73, MCP live 25, cargo audit
./scripts/backup-db.sh           # optional; keeps last 7 dumps in ./backups/
```

Publishing (maintainers): tag `v0.10.0` triggers [`.github/workflows/publish.yml`](.github/workflows/publish.yml) — GitHub Release binaries, PyPI wheels, npm, MCP Registry.

Regenerate README demo GIF: `./scripts/record-demo-gif.sh` (requires [asciinema](https://asciinema.org/) + [agg](https://github.com/asciinema/agg)).

| Channel | Version | Install |
|---------|---------|---------|
| Cargo / Release | `0.10.0` | [releases/tag/v0.10.0](https://github.com/LeandroPG19/cuba-memorys/releases/tag/v0.10.0) |
| npm | `0.10.0` | `npm i -g cuba-memorys@0.10.0` |
| PyPI | `1.12.0` | `pip install cuba-memorys==1.12.0` |

---

## Security & Audit

**Internal Audit Verdict: GO** (2026-03-28) — re-validated on v0.10 merge gate.

| Check | Result |
|-------|:------:|
| SQL injection | All queries parameterized (sqlx bind) |
| SEC-002 wildcard injection | Fixed (POSITION-based) |
| CVEs in dependencies | `cargo audit` in CI (allowed advisories documented) |
| UTF-8 safety | `safe_truncate` on all string slicing |
| Secrets | All via environment variables |
| Division by zero | Protected with `.max(1e-9)` |
| Error handling | All `?` propagated with `anyhow::Context` |
| Clippy | 0 warnings (`-D warnings` locally) |
| Tests | 118 unit + 13 smoke; E2E 73; MCP live session 25 tools |
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
| `mimalloc` | Global allocator | MIT |
| `tracing` | Structured JSON logging | MIT |
| `lru` | O(1) LRU cache | MIT |
| `chrono` | Timezone-aware timestamps | MIT/Apache-2.0 |

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **0.10.0** | Bitemporal `brain_facts` (default on), graph metrics/communities/activation, eval harness over live `cuba_faro`, views `v_unified_memory_search` + `v_observations_compat`, migrations `0018`–`0025`, merge-gate scripts. **Shipped** npm `0.10.0`, PyPI `1.12.0`. Hybrid search unchanged (RRF+BM25+vector). |
| **0.9.3** | Real bge-reranker-v2-m3 ONNX cross-encoder in `cuba_faro` (`CUBA_RERANKER_PATH`). |
| **0.9.0** | Search & Retrieval upgrades + Cognitive layer refinements + sqlx-migrate foundation. **PR #5**: 14 migraciones versionadas en `rust/migrations/`, bootstrap transparente para DBs v0.7/v0.8. **PR #6**: BM25 ts_rank_cd 3-way RRF (Robertson-Walker 1994), MMR diversification con Jaccard sim (Carbonell-Goldstein 1998), OOD Mahalanobis con ridge regularization (Lee NeurIPS 2018), tiktoken-rs cl100k_base exact counting, `hnsw.ef_search=200` dinámico en verify (recall@10≈0.99). **PR #7**: conformal prediction empírica reemplaza z-score gaussiano (Vovk 2005, Angelopoulos-Bates 2023), testing effect Karpicke-Roediger 2008 en zafra decay, Hebbian Δt-aware burst suppression τ=600s (anti-saturación STDP-light), Robbins-Monro stochastic LR en Oja `η=0.05/√(1+access/100)`, source credibility tracking Beta(α,β) Bayesian Yin-Han-Yu IEEE TKDE 2008 con nueva action `cuba_calibrar trust`. Nuevas deps: `tiktoken-rs 0.7`, `nalgebra 0.33` (no LAPACK), sqlx feature `migrate`, `async-trait 0.1`. 23 tools, 97 tests (+22 nuevos), 0 clippy, 0 tech debt, 0 breaking changes. |
| **0.8.0** | 4 new tools inspired by Engram Cloud + zero-regression refactor of all v0.7 readers/writers. **`cuba_proyecto`** — per-project isolation via `project_id` UUID FK on 6 tables (NULL = global = back-compat). **`cuba_pre_compact`** — snapshot/restore session state across `/compact`. **`cuba_sync`** — git-friendly export (1 JSON per entity + monthly-partitioned episodes + manifest with content-derived hash) and idempotent import. **`cuba_juez`** — LLM-judge for ambiguous (cosine 0.6-0.8) contradictions via subprocess to `claude` CLI ($0 with subscription) or Anthropic API behind feature `anthropic-api`, with heuristic fallback. Verdicts cached permanently. 4 new idempotent migrations (project scoping, compaction snapshots, sync state, judgments). All 19 v0.7 handlers audited (~30 SQL queries patched with `($N::uuid IS NULL OR project_id = $N OR project_id IS NULL)` pattern). New deps: `zstd 0.13`, `async-trait 0.1`, optional `reqwest 0.12`. 75 tests, 0 clippy, 0 tech debt. |
| **0.7.0** | 10 algorithmic improvements: PageRank blend (α=0.3, preserves Hebbian/BCM), hybrid verify (trigram+embedding fusion), ONNX semaphore (Little's Law), sigmoid entropy routing (Jaynes MaxEnt), word-level session boost, weighted Hebbian neighbors (Collins & Loftus), exponential coverage saturation, O(n) entropy. 19 bug fixes: hash embeddings corrupting DB (×5), centrality /2, cache LRU, jornada TOCTOU, alarma self-match, 6 schema mismatches. Removed blake3 dependency. MCP Registry publish fixed. npm postinstall version sync. 68 tests, 0 clippy, 0 tech debt. |
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

Mathematical foundations: Oja (1982), Bienenstock, Cooper & Munro (1982, BCM), Cormack (2009, RRF), Brin & Page (1998, PageRank), Traag et al. (2019, Leiden), Brandes (2001), Shannon (1948), Pearson (1900, chi-squared), Friston (2023, PE gating), Tulving (1972, episodic memory), Wixted (2004, power-law forgetting), Adamic & Adar (2003, link prediction), Anthropic (2024, Contextual Retrieval), Wang et al. (2022, E5 embeddings), Malkov & Yashunin (2018, HNSW), Jaynes (1957, MaxEnt sigmoid routing), Robertson (1977, score fusion), Collins & Loftus (1975, spreading activation).
