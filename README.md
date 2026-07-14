<!-- mcp-name: io.github.LeandroPG19/cuba-memorys -->
# Cuba-Memorys

[![CI](https://github.com/LeandroPG19/cuba-memorys/actions/workflows/ci.yml/badge.svg)](https://github.com/LeandroPG19/cuba-memorys/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cuba-memorys?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/cuba-memorys/)
[![npm](https://img.shields.io/npm/v/cuba-memorys?logo=npm&logoColor=white&label=npm)](https://www.npmjs.com/package/cuba-memorys)
[![MCP Registry](https://img.shields.io/badge/MCP_Registry-published-8A2BE2)](https://registry.modelcontextprotocol.io)
[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL%20v3-green)](https://www.gnu.org/licenses/agpl-3.0)

**Long-term memory for AI coding agents.** An MCP server that gives your agent a knowledge graph it can search, reason over, and be corrected by — so it stops forgetting your codebase between sessions.

Written in Rust. Backed by PostgreSQL + pgvector. **28 MCP tools, 14 CLI commands**, and every number below measured on a benchmark that — as of v0.12 — actually measures what it claims to. (The previous one did not. See [Measured](#measured--and-the-benchmark-that-was-lying).)

<p align="center">
  <img src="assets/demo.gif" alt="cuba-memorys terminal demo — hybrid search, claim verification with an LLM judge, procedural memory, and the CLI" width="760" />
</p>

---

## Install

```bash
pip install cuba-memorys        # or: npm install -g cuba-memorys
claude mcp add cuba-memorys -- cuba-memorys
```

That is the whole setup. On first run it provisions a PostgreSQL 18 + pgvector container via Docker and initializes the schema. **[Docker](https://docs.docker.com/get-docker/) must be running.**

<details>
<summary><b>Cursor / Windsurf / VS Code / Zed</b></summary>

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "cuba-memorys"
    }
  }
}
```

No `DATABASE_URL` needed. Or run `cuba-memorys setup` and it writes the config for every client it finds — then `cuba-memorys setup check` audits them for disagreement, which is the failure that actually bites (two configs, two embedding dimensions, one silently broken search).
</details>

<details>
<summary><b>Bring your own PostgreSQL</b></summary>

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "cuba-memorys",
      "env": { "DATABASE_URL": "postgresql://user:pass@localhost:5432/brain" }
    }
  }
}
```
Needs the `vector` and `pg_trgm` extensions. `cuba-memorys doctor` will tell you if anything is missing.
</details>

<details>
<summary><b>Semantic embeddings (recommended)</b></summary>

Without a model, embeddings are hash-based: deterministic, and semantically meaningless. Search still works through the lexical and BM25 branches, but nothing understands *meaning*.

```bash
./rust/scripts/download_model.sh                      # ~113 MB, multilingual-e5-small (384-d)
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"    # BOTH are required
```

**bge-m3 (1024-d) is better than e5-small**, though the size of the gap is no longer claimed: the +21 nDCG figure that used to sit here came from a benchmark that scored relevance by substring match. It needs a dimension migration (`scripts/migrate-embedding-dim.sh 1024`) and `CUBA_EMBED_MODEL=bge-m3 CUBA_POOLING=cls`.

Set `ONNX_MODEL_PATH` without `ORT_DYLIB_PATH` and the server tells you and degrades to lexical search. (Until v0.11.2 it hung silently instead. That was the worst bug in this project's history.)
</details>

---

## What it actually does

Most memory servers are a key-value store with an embedding bolted on. This one models four kinds of memory, because the psychology literature says they are four different things and they decay differently:

| | What it holds | How it strengthens |
|---|---|---|
| **Semantic** | Facts about entities — "all endpoints are async" | Access (Hebbian/BCM, Oja 1982) |
| **Episodic** | Events with actors and time — "we shipped v2 on Tuesday" | Power-law decay (Tulving 1972, Wixted 2004) |
| **Procedural** | How things are *done* here — recipes with a track record | **Success**, not access (ACT-R) |
| **Working** | Scratch notes bound to the current session | Cleared with the session |

Procedural memory is a separate table rather than a ninth observation type for a specific reason: ACT-R separates declarative memory (reinforced by *access*) from procedural (reinforced by *success*). As an observation, a recipe consulted constantly *because it keeps failing* would climb in importance. It is ranked by **Wilson lower bound**, so 1/1 successes scores 0.21 and 47/50 scores 0.84 — a lucky first try does not outrank a track record.

### Retrieval

Hybrid RRF fusion (k=60, Cormack 2009) over three signals — full-text, BM25 (`ts_rank_cd`), and pgvector HNSW — with entropy-routed weighting that shifts from keyword-heavy to semantic as the query's Shannon entropy rises.

Answers arrive in **`compact` by default**: abbreviated keys, content truncated at 1200 chars. **28% fewer tokens at identical nDCG** — identical to four decimal places, because the response format cannot change which documents rank, only how they are printed. Pass `"format": "verbose"` for the full per-branch score breakdown.

### Verification that actually verifies

`cuba_faro mode=verify` checks a claim against what is stored. It used to score claims by **cosine similarity to the retrieved evidence**, and that does not work — similarity measures what a text is *about*, not what it *asserts*. "cuba-memorys is written in Rust" and "…in Java" are nearly the same vector. Measured on the live corpus, the **false claim scored 0.61 and the true one 0.59.**

It now escalates the evidence to an LLM judge (`supports` / `contradicts` / `unrelated`) and derives confidence from the verdicts, weighting each by the evidence's similarity. Same corpus, after:

| Claim | Before | After |
|---|---|---|
| "written in Rust" (true) | 0.59 | **0.83 · verified** |
| "written in Java" (false) | **0.61** | **0.00 · contradicted** |
| "the best paella uses saffron" (unrelated) | 0.45, with 10 "evidence" items | **0.00 · unknown**, no evidence |

Being on-topic is not support, and the judge is told so explicitly. The backend is resolved automatically: your MCP client's own model via sampling (costs this server nothing), a local `claude` CLI, the Anthropic API, or — with none of those — an honest `unknown` rather than an invented verdict.

### Calibrated abstention

The out-of-distribution gate rejects queries the corpus cannot answer. The threshold is **not** a magic constant: Ledoit-Wolf covariance shrinkage plus a conformal quantile, calibrated against your own corpus with `cuba-memorys calibrate --apply` and persisted. (The theoretical χ² threshold rejected **100% of answerable queries.** Distribution-free calibration is not a nicety here.)

### And it tells you when it is broken

```
$ cuba-memorys doctor
[  ok  ] migrations           33 aplicadas, ninguna dirty
[  ok  ] embedding_dim        runtime 1024-d == columna vector(1024)
[  ok  ] runtime_role         'cuba_app' sin superuser — RLS y audit efectivos
[ warn ] binary_freshness     4 proceso(s) MCP corren un binario más viejo que el de disco
```

This exists because the failure mode of a hybrid search engine is not a crash — it is a vector branch dying and the search quietly becoming lexical, with no symptom. The server now **refuses to start** on an embedding-dimension mismatch, and search sets `degraded: true` in the response when a branch fails.

---

## The CLI: your memory without an LLM in the middle

Fourteen commands. `cuba-memorys --help` lists them all.

| | |
|---|---|
| `search <query>` · `save` · `delete` · `export` | Read and write the brain from a shell |
| `dashboard` | A self-contained HTML view of what is in there |
| **`doctor`** | Health check: schema, dimensions, config coherence, stale processes |
| `recall` | Session-start context injection — wire it with `setup hook` |
| `reembed` | Re-encode what needs it (default: only stale rows, not all of them) |
| `calibrate` | Recompute the abstention threshold from your corpus |
| `link` | Auto-link entities by NPMI co-occurrence |
| **`dedupe`** | Entities that are the same thing under different names — see below |
| `skills <dir>` | Export procedures as Claude Code Skills |
| `eval` | Retrieval benchmark — nDCG@10 with confidence intervals, MRR, recall, token cost |
| `setup` | Wire this into your MCP clients; `setup check` audits them |

### `dedupe` — because a different string is a different entity

`cuba_alma create` inserts with `ON CONFLICT (name)`. So one project fragments into `Mapupita-Web`, `Mapupitta-Web` (typo), `Mapupita Web`, `mapupita`… and searching one finds none of the others. On a real 266-entity graph, **158 of them (59%) had not a single relation** — for PageRank and multi-hop retrieval, they did not exist.

What decides a merge is **not** the embedding centroid. That was the obvious idea and it is wrong: `M-Codes Reference Guide` and `G-Codes Reference Guide` sit at **0.811 cosine** between centroids. On a corpus about one domain, centroid similarity measures the *domain*, not the *entity* — a 0.80 threshold would have merged two different CNC guides, irreversibly.

So `--apply` merges only what is **provable** (identical after normalizing case and separators). Typos and near-matches are shown, and judged one at a time with `--judge`. The old name is written to `brain_entity_aliases`, so nothing is lost: looking it up still resolves.

---

## The 28 tools

Named after Cuban culture. `cuba-memorys` advertises all of them, or set `CUBA_TOOL_PROFILE=lean` to advertise only `cuba_tools` + `cuba_call` — **67% smaller tool catalogue, zero functions lost**, schemas loaded on demand.

**Knowledge graph** — `cuba_alma` (entities) · `cuba_cronica` (observations, episodes, timeline) · `cuba_puente` (typed relations, traversal, link prediction) · `cuba_ingesta` (bulk import)

**Search** — `cuba_faro` (hybrid RRF, verification, MMR diversification, OOD abstention)

**Error memory** — `cuba_alarma` (report) · `cuba_remedio` (resolve) · `cuba_expediente` (search past errors; warns if an approach failed before)

**Sessions & decisions** — `cuba_jornada` (session lifecycle, diff) · `cuba_decreto` (architecture decisions) · `cuba_proyecto` (per-project isolation) · `cuba_pre_compact` (survive `/compact`)

**Procedural** — `cuba_receta` (recipes ranked by Wilson lower bound)

**Cognition** — `cuba_reflexion` (gap detection) · `cuba_hipotesis` (abductive inference) · `cuba_contradiccion` (semantic conflicts) · `cuba_juez` (LLM judge) · `cuba_centinela` (prospective triggers) · `cuba_calibrar` (Bayesian calibration, source credibility)

**Maintenance** — `cuba_zafra` (decay, prune, merge, PageRank, Leiden communities) · `cuba_eco` (RLHF feedback) · `cuba_vigia` (health, drift, centrality) · `cuba_forget` (GDPR erasure) · `cuba_archivo` (CFR-21 hash-chain audit log) · `cuba_pizarra` (working memory) · `cuba_sync` (git-friendly export/import)

**Meta** — `cuba_tools` (discover) · `cuba_call` (invoke)

---

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `DATABASE_URL` | auto (Docker) | PostgreSQL connection |
| `ONNX_MODEL_PATH` + `ORT_DYLIB_PATH` | — | Semantic embeddings. **Both or neither.** |
| `CUBA_EMBED_MODEL` · `CUBA_EMBEDDING_DIM` · `CUBA_POOLING` | `multilingual-e5-small` · `384` · `mean` | Set to `bge-m3` · `1024` · `cls` for the +21 nDCG model |
| `CUBA_TOOL_PROFILE` | `full` | `lean` → 2 tools, 67% smaller catalogue, nothing lost |
| `CUBA_JUDGE` | `auto` | `mcp_sampling` / `claude_cli` / `anthropic_api` / `heuristic` |
| `CUBA_COMPACT_CHARS` | `1200` | Compact truncation (measured knee) |
| `CUBA_OOD_THRESHOLD` | calibrated | Override the abstention threshold |
| `CUBA_BITEMPORAL` | on | Mirror observations into `brain_facts` |

---

## Measured — and the benchmark that was lying

Until v0.12 this section carried a line reading *"every number here is measured rather than assumed"*, and every number in it was wrong. The benchmark was broken in three ways, and finding out cost two published conclusions.

**It had ten queries.** A 95% interval of roughly ±0.12; the smallest effect it could detect was ~0.25 nDCG. Any claim about a smaller difference was noise wearing a decimal point.

**Relevance was judged by substring match.** A result counted as correct if its text merely *contained* a marker word — so every observation mentioning "postgres" scored as a right answer to any question about postgres, whether it answered anything or not. That measures keyword presence, not retrieval, and it tilts the whole benchmark toward the lexical branch and against the vector one.

**nDCG normalized against what was retrieved, not what exists.** With 5 relevant documents in the corpus and 2 found, the "ideal" ranking was taken to be those 2 — so a system that missed 60% of the answer scored a perfect **1.0**. (And `R@10 = 3.125` shipped in this file. Recall is a proportion.)

The real number is not 0.894. On 221 id-scored queries it is **nDCG@10 = 0.50** [95% CI 0.44–0.56]. The system did not get worse. It was never 0.894.

### What that cost

- ~~"The cross-encoder reranker earns nothing"~~ — **it had never run.** Three bugs in series: `faro` wrapped the call in `if let Ok(..)` and dropped the error; it fed `token_type_ids` to a model that is XLM-RoBERTa and has none; it read `f16` logits as `f32`. The output was "bit for bit identical" to no reranking not because reranking changed nothing, but because it never happened. Fixed; being measured properly now.

- **Associative retrieval does degrade** — but the old evidence (−0.03 at n=10) could not have shown it. On the new dataset with a **paired bootstrap** (the correct test: same queries in both arms), the interval is **[−0.051, −0.018]** and never touches zero. It improves 0 queries and hurts 23. The decision was right; the reasoning was not. *The power was never in more data — it was in using the right test.*

### What survives, re-measured honestly

| | |
|---|---|
| **`compact` by default** | **−28% tokens at identical nDCG** (paired difference: exactly 0.0000 — format cannot change which documents rank, only how they are shown). The old "−40%" came from the broken benchmark. |
| **Conformal abstention** | 100% of out-of-distribution queries caught, 0% false abstentions. |
| **`lean` tool profile** | −67% catalogue, zero functions lost. |
| **bge-m3 over e5-small** | Direction almost certainly right; **the +21.2 nDCG figure is withdrawn** — it came from the broken benchmark and re-establishing it would mean re-embedding the corpus twice. |
| **The benchmark itself** | 221 queries (was 10), relevance by document **id**, bootstrap confidence intervals, and the **minimum detectable effect** printed beside every result — so nobody reads a 3-point difference as a finding again. |

---

## Foundations

| Algorithm | Reference |
|---|---|
| RRF fusion (k=60) | Cormack et al. (2009) |
| Hebbian + BCM metaplasticity | Oja (1982); Bienenstock, Cooper & Munro (1982) |
| Conformal prediction | Vovk (2005); Angelopoulos & Bates (2023) |
| Ledoit-Wolf covariance shrinkage | Ledoit & Wolf (2004) |
| Mahalanobis OOD detection | Lee et al. (NeurIPS 2018) |
| Wilson score interval | Wilson (1927) |
| Declarative vs procedural memory | Anderson & Lebiere (ACT-R) |
| Testing effect | Karpicke & Roediger (Science 2008) |
| Power-law forgetting | Wixted (2004) |
| Episodic vs semantic memory | Tulving (1972) |
| PageRank · Leiden · Brandes | Brin & Page (1998); Traag et al. (2019); Brandes (2001) |
| NPMI co-occurrence | Bouma (2009) |
| MMR diversification | Carbonell & Goldstein (1998) |
| Contextual Retrieval | Anthropic (2024) |
| Prompt-injection spotlighting | Hines et al. (2024) |

---

## Development

```bash
git clone https://github.com/LeandroPG19/cuba-memorys.git
cd cuba-memorys/rust && cargo build --release

./scripts/demo.sh          # runs on a throwaway Postgres it removes on exit
./scripts/merge-gate.sh    # fmt · clippy -D warnings · 223 tests · audit · integration
```

Publishing is tag-driven: `v*` triggers GitHub Release binaries (5 platforms), PyPI wheels, npm, and the MCP Registry. A test pins all four files that hold a version number to the same value, because they used to drift and nothing caught it.

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0) — free to use, modify and run, including inside a company. If you offer a modified version to others over a network, you have to publish your changes under the same license.

## Author

**Leandro Perez G.** — [@LeandroPG19](https://github.com/LeandroPG19)
