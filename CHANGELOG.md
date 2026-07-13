# Changelog

All notable changes to cuba-memorys are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows
[SemVer](https://semver.org/) for the Rust crate (`Cargo.toml`). PyPI
versioning is independent (~ +1.0 offset since v0.6.0 era to allow wheel
revisions without binary changes).

## [0.11.0] — 2026-07-13 (Cargo `0.11.0` · npm `0.11.0` · PyPI `1.13.0`)

The fourth memory, and every optimization measured on a real corpus instead of
assumed. Several long-standing features turned out not to work at all; they are
fixed or cut, and the negative results are recorded rather than buried.

### ⚠ Breaking — `cuba_faro` now answers in `compact` by default

The default response shape changed from `verbose` to `compact`: abbreviated keys
(`e` entity, `c` content, `t` type, `i` importance, `s` score) and no per-branch
score breakdown. It costs **40% fewer tokens at identical nDCG** — the truncation
point was swept and set at its measured knee — and an agent reasoning over
memories does not need `bm25_score` to do it.

**If you parse the response**, this breaks you. Pass `"format": "verbose"` to get
the old shape back, unchanged:

```json
{ "query": "...", "format": "verbose" }
```

Agents reading the JSON are fine — the tool description documents the short keys.
Scripts and tests that index `entity_name` / `content` / `*_score` are not, and
must ask for `verbose`. Both shapes are now pinned by an integration test, so
neither can drift again.

### Added

- **Procedural memory** — `cuba_receta` (migration `0033`): how things are *done*
  here, not just what is true. Ranked by the **Wilson lower bound** of the success
  rate, so a recipe with a track record beats a lucky first try (1-of-1 scores
  0.21; 47-of-50 scores 0.84). Reinforced by outcome, not by access — the
  ACT-R distinction between declarative and procedural memory. `cuba-memorys
  skills <dir>` exports them as Claude Code Skills, which load lazily.
- **Progressive tool loading** — `cuba_tools` + `cuba_call`, and
  `CUBA_TOOL_PROFILE=lean`. The catalogue shrinks 67% (25,060 → 8,413 chars)
  while **every tool stays callable**: schemas are deferred, not deleted.
- **Calibrated abstention** — `cuba-memorys calibrate`. The OOD gate now detects
  out-of-distribution queries (100%) without rejecting answerable ones (0% false
  abstentions). Persisted in `brain_calibration` (`0032`).
- **RBAC** — `brain_principals` × `brain_grants` (`0031`), enforced by a
  RESTRICTIVE RLS policy. Zero regression: with no principals defined, nothing
  is denied.
- **New subcommands** — `doctor` (health check), `calibrate`, `recall` (session
  context for a `SessionStart` hook), `skills`, `reembed`, `link`, `setup`,
  `search` / `save` / `delete` / `export` / `dashboard`.
- **Graph auto-linking** — `cuba-memorys link`, scored by normalized pointwise
  mutual information so a ubiquitous entity earns no edges from being ubiquitous.
- **Model-agnostic embeddings** — e5-small (384-d) or bge-m3 (1024-d) by config.
  Measured on a real 1,443-observation corpus: **nDCG@10 0.682 → 0.894**.

### Fixed

- **Hybrid search could silently become lexical search.** A failing vector branch
  was discarded by an `if let Ok(..)` — no log, no flag, no symptom. Now it logs
  at ERROR, sets `degraded: true` in the response, and the server **refuses to
  start** when the model's dimension disagrees with the column.
- **`setup check` reported "all consistent" while a stale project-level
  `.mcp.json` spawned 384-d servers against a 1024-d column.** It now audits
  project configs too, and treats an absent `CUBA_EMBEDDING_DIM` as the 384-d
  value it actually is — so it can disagree with one that sets 1024.
- **Retrieval was not deterministic.** Fusion happened in a `HashMap` and sorted
  by score with no tie-break; Rust randomizes iteration order per process, so
  three identical eval runs scored 0.7389 / 0.7344 / 0.7389. Every optimization
  number previously recorded rested on that. Now tie-broken by id: 5/5 identical.
- **The token budget counted text it then threw away**, spending a 5,000-token
  budget to return 798. Shape first, then budget. Compact truncation swept and
  set at its measured knee (1200 chars): **40% fewer tokens, identical nDCG**.
- **The OOD threshold rejected 100% of answerable queries.** The covariance was
  fitted from 500 samples in 384 dimensions with a fixed ridge mislabelled
  "Ledoit-Wolf". Now real Ledoit-Wolf shrinkage plus a conformal threshold.
- **The eval panicked on an empty result list** (`relevances[..1]` on a
  zero-length slice) — it never fired only because nothing ever abstained.
- **The LLM judge shipped credentials to a third party.** Observation text is now
  redacted (Postgres URLs, provider tokens, JWTs) and length-capped.
- **`doctor` could not see a stale process** — Linux appends `" (deleted)"` to
  the exe name, and the filter dropped exactly the processes it existed to find.
- **`cuba-memorys --version` connected to your database and ran migrations.**
  Argument parsing had a catch-all that fell through to the MCP server, so the one
  command a person runs *because they do not yet trust what they installed* was the
  one that quietly reshaped their schema. `--version` is now inert — it prints and
  exits, with a test that pins it by pointing `DATABASE_URL` at a closed port.
- **`--help` did not exist**, for the same reason, which is why nothing ever
  documented the 13 subcommands. And a typo (`doctro`) launched the server on a
  stdio socket nobody was speaking to — indistinguishable from a hang. An
  unrecognised argument is now a usage error (exit 2). The server is what you get
  with *no* arguments, which is how MCP clients launch it.
- **npm could silently run a different version than the one you installed.**
  `bin.js` fell back to any `cuba-memorys` on the `PATH` when the postinstall
  binary was missing — and postinstall does not run under `--ignore-scripts`,
  standard practice in hardened CI. Installing 0.11.0 and getting an 0.6.0 left
  over from an old pip install is not a fallback; here it is a *migration* run by
  the wrong binary. The `PATH` binary must now prove its version matches, or the
  launcher refuses and says why.
- A test now pins `Cargo.toml` and `package.json` to the same version. npm's
  postinstall downloads from `releases/download/v{package.json.version}/`, an asset
  the release workflow only builds for the *Cargo* version — nothing connected
  those two numbers, and a drift would have 404'd every install.
- Zero `unwrap()` in production code; zero clippy warnings.

### Changed

- `cuba_faro` defaults to `compact` (40% cheaper, same quality).
- The eval reports **token cost beside every quality metric**, and tracks false
  abstentions — abstention accuracy alone is trivially maximized by answering
  nothing.

### Removed / not adopted

- **The cross-encoder reranker does not earn its place.** Its integration added
  `score × 0.0001` to fusion scores separated by 0.00016 — arithmetically
  incapable of reordering anything. Fixed the wiring, measured it properly, and
  it still bought nothing for 0.33 s/query and 1.1 GB. Off by default, with the
  negative result documented in the module.
- **Associative multi-hop retrieval degrades every metric** (nDCG 0.734 → 0.705,
  MRR 0.833 → 0.660, recall 2.31 → 1.88). The previous "+10 points recall" claim
  predates the determinism fix. Stays opt-in and off.

## [0.10.0] — 2026-06-04 (Cargo `0.10.0` · npm `0.10.0` · PyPI `1.12.0`)

Knowledge-graph memory plane: bitemporal facts, graph metrics, retrieval benchmarks,
and MCP unified search view — built on the v0.9 hybrid `cuba_faro` stack (not replaced).

### Added
- **Bitemporal core** (`core::bitemporal`, migration `0018`): `brain_facts` +
  `brain_fact_supersedes`; writes mirror observations on `cuba_cronica` add/batch_add
  and `cuba_ingesta` (via batch). **Default on**; disable with `CUBA_BITEMPORAL=0`.
- **Entity linking & temporal query** (`core::entity_linking`, `core::temporal_query`,
  migrations `0019`–`0020`).
- **Graph metrics** (migration `0022`): `brain_node_metrics` with PageRank, energy,
  betweenness; `cuba_zafra` `pagerank` persists ranks then refreshes energy scores.
- **Communities** (migration `0023`): Leiden detection + `detect_and_persist`;
  `cuba_zafra` action `communities`; `cuba_vigia` health metric persists tags.
- **Spreading activation** (`graph::activation`): multi-hop propagation; enriches
  `cuba_puente` `predict` alongside Adamic-Adar.
- **Eval harness** (`eval/`): nDCG@k, MRR, P@k, R@k over live `cuba_faro` hybrid;
  JSONL dataset loader + builtin smoke set; JSON reporters.
- **MCP memory v2 view** (migration `0024`): `v_unified_memory_search` joins facts via
  `brain_entities` (never `fact_id = node_id`).
- **Compatibility views** (migration `0025`): `v_observations_compat`.
- **Calibration alignment** (migration `0021`), scripts: `backup-db.sh`, `restore-db.sh`,
  `merge-gate.sh`, `mcp_live_session_test.py`.

### Changed
- `cuba_faro` remains production hybrid search (RRF + BM25 + vector + optional rerank).
- PageRank REM cycle also upserts `brain_node_metrics.pagerank_score`.

### Notes
- Optional Cargo features `bitemporal`, `graph-energy`, `eval-benchmarks` are markers;
  modules ship in the default library build.
- Run `./scripts/merge-gate.sh` before merging to `main`.

---

## [0.9.3] — 2026-05-04 (Cargo `0.9.3` · npm `0.9.3` · PyPI `1.11.3`)

Final piece of the v0.9.x roadmap. The cross-encoder reranker is now a
real bge-reranker-v2-m3 ONNX forward pass, not the heuristic baseline
that v0.9.2 shipped as scaffolding.

### Added
- **Real bge-reranker-v2-m3 ONNX forward pass** (`search::rerank`).
  Mirrors the `embeddings::onnx` loader pattern: lazy-init `Session`
  behind a `Mutex`, `tokio::task::spawn_blocking` for inference, and a
  semaphore capping concurrent calls at 2 to match
  `with_intra_threads(2)` (Little's Law — prevents threadpool
  starvation under load). Tokenizer encodes the (query, candidate)
  sentence pair with `[CLS]/[SEP]` segments and 512-token truncation.
  Output handled for both `[batch, 1]` regression heads and
  `[batch, 2]` binary classification heads (logit difference). Sigmoid
  to [0, 1] before sorting.
- Activation: drop a directory containing `model.onnx` (or
  `model_quantized.onnx`) plus `tokenizer.json` and point
  `CUBA_RERANKER_PATH` at it. Identity fallback otherwise — production
  behavior unchanged when the asset is absent.
- `cuba_faro` keeps the same `rerank: bool` arg surface from v0.9.2;
  no client-side change required.
- Expected gain: +12-25% nDCG@10 (Xiao 2023, BGE-Reranker paper).

### Changed
- Replaced the v0.9.2 heuristic body (token overlap + length penalty)
  with the real cross-encoder forward pass. Heuristic-only callers
  see unchanged behavior because the env var gates activation.

### Notes
- Adds zero new Rust deps — `ort 2.0.0-rc.12` and `tokenizers 0.21`
  were already present for `embeddings::onnx`.
- bge-reranker-v2-m3 quantized ONNX is ~280 MB; download from
  https://huggingface.co/BAAI/bge-reranker-v2-m3 (or use
  `huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir
  models/bge-reranker-v2-m3`). The asset is NOT bundled in the
  release artifact — operators provide it explicitly.

---

## [0.9.2] — 2026-05-04 (Cargo `0.9.2` · npm `0.9.2` · PyPI `1.11.2`)

Closes the v0.9.x roadmap with the deferred MCP correlator + reranker
scaffolding. No breaking changes.

### Added
- **MCP request/response correlator** (`protocol.rs` major refactor).
  The reader/writer split into three concurrent tasks: a single-owner
  stdout writer task draining an `mpsc::UnboundedSender<Value>`, a
  `PENDING` map of `oneshot::Sender<Value>` keyed by server-initiated
  request id, and per-request handler tasks. This enables:
  - **Real `MCPSamplingJudge`** — `protocol::request_sampling()` issues
    a `sampling/createMessage` to the connected client and awaits the
    reply on a oneshot. 30s timeout matches `HANDLER_TIMEOUT`. When the
    client did not advertise `sampling`, fails fast with an actionable
    message and the resolver auto-falls back to CLI / API / heuristic.
  - **`notifications/progress`** — `protocol::notify_progress()` emits
    standard MCP progress events. Wired into `cuba_zafra reembed`
    (~5% increments) so re-embedding 500+ observations is no longer
    silent.
  - **`notifications/cancelled`** — per-request `CancelToken`
    registered by `tools/call`, signaled by the cancellation
    notification. Handlers race against the token via `tokio::select!`
    and return a clean error instead of running to completion.
- **Cross-encoder reranker scaffold** (`search::rerank`). Activated by
  `CUBA_RERANKER_PATH` env var pointing to a bge-reranker-v2-m3 ONNX.
  When unset, identity fallback preserves upstream RRF order. New
  `cuba_faro` argument `rerank` for explicit override. Pipeline:
  top-50 RRF → rerank → MMR → top-K. Heuristic body included as a
  baseline that exercises the integration path; full model forward
  documented as a one-file follow-up to drop in.
- New `cuba_faro` arg surface: `rerank` (boolean).
- New helper `protocol::register_cancel_token` /
  `protocol::unregister_cancel_token` exported for any future handler
  that wants explicit cancellation hooks.

### Changed
- `JsonRpcResponse` / `JsonRpcError` structs removed — every outbound
  envelope is built ad-hoc with `serde_json::json!()` and pushed to the
  `OUTBOUND` channel. Narrower surface, no temptation to construct
  envelopes from places that should not.
- `cuba_zafra reembed` accepts `_meta.progressToken` to correlate
  progress with the MCP `tools/call` request id.

### Notes
- Real bge-reranker forward pass (ONNX session + tokenizer) is the
  only piece marked as TODO in the rerank module. The integration
  point, env var, schema arg, and identity fallback are all live.
  When the asset is bundled, swap the ~30-line heuristic body for the
  real inference and `enabled()`-true path becomes production.

---

## [0.9.1] — 2026-05-04 (Cargo `0.9.1` · npm `0.9.1` · PyPI `1.11.1`)

Production hardening + MCP spec usage. Closes the v0.9.x roadmap with
PRs #8–#11 plus the deferred infrastructure pieces from #10/#11.

### Added
- **PR #8** — `graph::closeness` (Bavelas 1950 + Boldi-Vigna 2014 harmonic)
  and `graph::kcore` (Seidman 1983, Batagelj-Zaversnik 2003 with
  running-max for correct k-core numbers). Exposed via new
  `cuba_vigia metric=structural` action.
- **PR #9** — working memory (`cuba_pizarra` + `brain_wm` table with
  GENERATED `expires_at` from `ttl_seconds`), Allen interval algebra
  (`cognitive::allen`, all 13 relations in O(1)), ADWIN drift detector
  (`cognitive::adwin`, Bifet-Gavaldà SDM 2007 with Hoeffding bound +
  Bonferroni correction), MI tagging (`cognitive::mi_tagging`,
  Brown JMLR 2012).
- **PR #10** — Tamper-evident audit log (`cuba_archivo` + `brain_audit_log`
  with SHA-256 hash chain, append-only PostgreSQL trigger, `cuba_admin`
  bypass role). Spotlighting prompt-injection defense in
  `cognitive::judge::build_prompt` (Hines 2024 — per-call nonce markers).
  Brier score + Expected Calibration Error in `cuba_calibrar metrics`
  (Brier 1950 / Naeini AAAI 2015) with reliability diagram.
- **Optional Prometheus `/metrics` endpoint** behind feature flag
  `observability` (`metrics 0.24` + `metrics-exporter-prometheus 0.16`).
  Default bind `127.0.0.1:9090` (env `CUBA_METRICS_PORT`/`CUBA_METRICS_BIND`).
  Pre-registered metrics: `cuba_handler_duration_seconds`,
  `cuba_handler_calls_total`, `cuba_judge_calls_total`,
  `cuba_judge_timeout_total`.
- **PostgreSQL Row-Level Security** per project (migration 0017).
  `tenant_isolation` policy across the six scoped tables. Defense in
  depth — the handler-side WHERE clause stays as the primary gate, RLS
  catches direct DB connections that bypass handlers. Sentinel `*` =
  bypass, NULL = back-compat.
- **PR #11** — MCP `resources/list` + `resources/read` with the
  `cuba://` URI scheme: `cuba://entity/<name>`,
  `cuba://project/<name>`, `cuba://snapshot/<id>`. Server now advertises
  the `resources` capability during initialize. Client capability
  detection captures `capabilities.sampling` so future Sampling calls
  prefer it (today it errors out with a clear migration message — full
  loop correlator scheduled for v0.10).
- New backend `MCPSamplingJudge` in `cognitive::judge` (auto-preferred
  when client supports sampling).
- New deps: `sha2 0.10`, `hex 0.4`. Optional: `metrics 0.24`,
  `metrics-exporter-prometheus 0.16`.

### Changed
- `cuba_juez` resolver order: `mcp_sampling` → `claude_cli` → `anthropic_api` → `heuristic`.
- `cuba_calibrar` JSON Schema gains `metrics` action.
- 25 MCP tools (was 24): `cuba_archivo` joins as the audit handler.
- Smoke test count bumped to 25.
- 4 new migrations (0014 source_trust, 0015 working_memory,
  0016 audit_log, 0017 rls_policies). Total: 17.

### Notes
- 3 RUSTSEC advisories from upstream transitive deps remain open
  (rustls-webpki via sqlx, rand via tokenizers/reqwest). All upstream;
  no remediation in this scope.
- MCP Sampling backend currently fails fast — wiring requires the
  request/response correlator refactor planned for v0.10. Auto-fallback
  keeps production unaffected.

---

## [0.9.0] — 2026-05-04 (Cargo `0.9.0` · npm `0.9.0` · PyPI `1.11.0`)

Search & Retrieval upgrades + Cognitive layer refinements + sqlx-migrate
foundation. Zero breaking changes — every new feature is opt-in via
`cuba_faro` arguments or activates automatically with safe defaults.

### Added
- **PR #5 sqlx-migrate** — replaced ad-hoc `*_MIGRATION` constants with
  versioned files in `rust/migrations/` (14 migrations, 0001 → 0014). The
  bootstrap is transparent for legacy v0.7/v0.8 DBs because every
  migration is idempotent (`DO $$ ... IF NOT EXISTS ... END $$`).
- **PR #6 Phase 1** — three new search modules:
  - `search::bm25` — BM25-flavored sparse retrieval via PostgreSQL
    `ts_rank_cd` (Robertson-Walker SIGIR 1994 baseline).
  - `search::mmr` — Maximal Marginal Relevance diversification with
    Jaccard token-set similarity (Carbonell-Goldstein SIGIR 1998).
  - `search::ood` — Out-of-distribution detection via Mahalanobis
    distance with ridge-regularized Σ⁻¹ (Lee NeurIPS 2018).
  - `search::budget` — exact `tiktoken-rs` cl100k_base counting (replaces
    the "len/4 chars per token" heuristic that mis-counted Spanish 30%).
- **PR #6 Phase 2-3** — `cuba_faro` exposes 5 new arguments:
  `enable_bm25` (default `true`), `diversify`, `mmr_lambda`,
  `abstain_ood`, `ood_threshold`. Output adds `bm25_score` to the score
  breakdown alongside `text_score`/`vector_score`. `verify` mode now
  bumps `hnsw.ef_search` to 200 transactionally for recall@10≈0.99.
- **PR #7 Phase 1** — `cognitive::prediction_error::adaptive_thresholds_conformal`
  uses empirical quantiles instead of z-score — distribution-free
  (Vovk-Gammerman-Shafer 2005, Angelopoulos-Bates 2023). Cosine
  similarities are anisotropic skewed-right (Ethayarajh EMNLP 2019), so
  z-score over-fires REINFORCE; conformal does not.
- **PR #7 Phase 2** — testing effect (Karpicke-Roediger Science 2008):
  `cuba_zafra decay` now scales halflife by `(1 + ln(1 + access_count))`,
  so a memory accessed 50× decays ≈4× slower than one accessed 0×.
- **PR #7 Phase 3** — Hebbian Δt-aware burst suppression in
  `cognitive::hebbian::boost_on_access`. The boost is multiplied by
  `(1 - exp(-Δt/τ))` with τ=600s. Re-access in the same second yields
  factor 0 (anti-saturation), Δt > 1h yields ≈1 (full boost).
- **PR #7 Phase 4** — Robbins-Monro stochastic learning rate in
  `cuba_eco`'s Oja positive/negative: `η = 0.05 / sqrt(1 + access_count/100)`.
  Convergence O(1/√t) bounds importance volatility on heavily-fed
  observations.
- **PR #7 Phase 5** — source credibility tracking. Migration
  `0014_source_trust.up.sql` adds `brain_source_trust(source, alpha,
  beta, updated_at)` pre-seeded with five standard sources. Each
  `cuba_calibrar resolve` updates the Beta(α, β) posterior of every
  source supporting the verified claim (Yin-Han-Yu IEEE TKDE 2008). New
  `cuba_calibrar trust` action returns posteriors with credible-interval
  width.
- New deps (lib): `sqlx` feature `migrate`, `tiktoken-rs 0.7`,
  `nalgebra 0.33` (no LAPACK), `async-trait 0.1`.
- 22 new tests (97 total: 84 unit + 13 smoke + 2 integration ignored).

### Changed
- `cuba_faro` JSON Schema in `constants.rs::tool_definitions()` extended
  with v0.9 args. Description updated to advertise MMR / OOD / tiktoken.
- `cuba_calibrar` JSON Schema gains the `trust` action.
- `cuba_zafra` `decay` response includes `testing_effect` annotation
  with the Karpicke-Roediger citation.
- `db.rs` shrunk from 310 → 100 lines (sqlx-migrate replaces nine
  hand-rolled migration constants).
- Smoke test `test_handler_dispatch_coverage` keeps the same 23-tool
  list (no new MCP tools added in v0.9 — all upgrades are arg
  extensions or new actions on existing handlers).

### Fixed
- 6 pre-existing clippy 1.94 warnings cleaned: `vec![...]` → array
  literals in `graph/pagerank.rs::tests`; `assert!(CONST > 0)` →
  `const _: () = assert!(...)` in `tests/smoke_test.rs` and
  `pagerank.rs::tests::test_pagerank_convergence_constants`.

### Notes for upgraders
- Existing v0.7 / v0.8 DBs auto-migrate on first boot. The
  `_sqlx_migrations` table is created automatically and populated with
  the 14 historical migrations on the first run; subsequent boots
  apply only deltas.
- Legacy `embeddings/onnx.rs` heuristic `count_tokens` is now
  superseded by `search::budget::count_tokens`.

---

## [0.8.0] — 2026-05-04 (Cargo `0.8.0` · npm `0.8.0` · PyPI `1.10.0`)

Engram-Cloud-inspired additions: 4 new tools + audit of all 19 v0.7
handlers for project scoping. Zero breaking changes — every filter is
opt-in via `cuba_jornada start --project NAME`.

### Added
- **`cuba_proyecto`** (PR #1) — per-project isolation via `project_id
  UUID NULL` on six core tables, `brain_projects` registry, six actions
  (`list / current / switch / stats / rename / merge`). NULL = global =
  back-compat. Kill-switch `CUBA_PROJECT_FILTER=off`.
- **`cuba_pre_compact`** (PR #2) — survives `/compact`. `snapshot`
  persists session state (recent obs, decisions, unresolved errors,
  pending embeddings, goals) into `brain_compaction_snapshots`;
  `restore` returns the latest snapshot for the active session.
  `cuba_jornada current` now returns `compaction_hint: bool`.
- **`cuba_sync`** (PR #3) — git-friendly export/import with
  `export | import | diff | status`. Layout: `manifest.json`,
  `entities/<slug>.json` (each with embedded observations),
  `episodes/<yyyy-mm>/<id>.json`, `decisions/<id>.json`,
  `errors/<id>.json`, `relations.json`, optional `embeddings.bin.zst`.
  Idempotent via `ON CONFLICT DO NOTHING` + `brain_sync_state`. Schema
  versioning with hash-derived dedup. Path traversal protection.
- **`cuba_juez`** (PR #4) — LLM-judge for ambiguous (cosine 0.6-0.8)
  contradictions. Trait `ContradictionJudge` with three backends:
  `ClaudeCodeJudge` (subprocess to `claude` CLI, $0 with subscription),
  `AnthropicApiJudge` (feature flag `anthropic-api`), `HeuristicJudge`
  (fallback wrapping the bilingual negation marker check).
  Permanent cache via `brain_judgments(observation_a, observation_b)`
  UNIQUE index.
- WRITE audit: `cronica`, `alma`, `alarma`, `puente`, `decreto` populate
  `project_id` from current session.
- READ audit: `faro`, `vigia`, `expediente`, `contradiccion`,
  `reflexion`, `hipotesis`, `decreto`, `puente`, `alma`, `cronica` apply
  `($N::uuid IS NULL OR project_id = $N OR project_id IS NULL)` filter.

### Changed
- 23 MCP tools (was 19). 75 tests (was 68). 0 clippy warnings.

---

## [0.7.0] — Earlier 2026

10 algorithmic improvements + 19 bug fixes + comprehensive audit
(condensed): PageRank α=0.3 blend (preserves Hebbian/BCM learned
importance), hybrid verify (trigram + embedding fusion), ONNX
concurrency semaphore (Little's Law), sigmoid entropy routing
(Jaynes 1957 MaxEnt), word-level session boost, weighted Hebbian
neighbor diffusion (Collins-Loftus 1975), exponential coverage
saturation, O(n) entropy. Fixed: hash embeddings corrupting DB,
centrality normalization, cache LRU, jornada race condition, six MCP
schemas. Removed `blake3` dependency. 68 tests, 0 clippy warnings,
0 tech debt.

[0.9.0]: https://github.com/LeandroPG19/cuba-memorys/releases/tag/v0.9.0
[0.8.0]: https://github.com/LeandroPG19/cuba-memorys/releases/tag/v0.8.0
[0.7.0]: https://github.com/LeandroPG19/cuba-memorys/releases/tag/v0.7.0
