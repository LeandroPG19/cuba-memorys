# Changelog

All notable changes to cuba-memorys are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows
[SemVer](https://semver.org/) for the Rust crate (`Cargo.toml`). PyPI
versioning is independent (~ +1.0 offset since v0.6.0 era to allow wheel
revisions without binary changes).

## [0.12.0] — 2026-07-13 (Cargo `0.12.0` · npm `0.12.0` · PyPI `1.14.0`)

The benchmark could not measure what this project claimed to have measured, and the
graph was quietly broken.

### ⚠ Two published findings are withdrawn

The evaluation had **ten queries**. At n=10 the 95% interval on nDCG is roughly
±0.12, and the smallest detectable effect is ~0.25. Two conclusions this project
published rested inside that noise:

- ~~**"The cross-encoder reranker earns nothing"**~~ — **it had never run.** Three
  bugs in series, each hiding the next:

  1. `faro` wrapped the call in `if let Ok(..)`, so the error was **dropped** and the
     RRF ranking returned untouched. The same silent-degradation pattern that let the
     vector branch die unnoticed. *This is why the output was "bit for bit identical"
     to not reranking:* not because reranking changed nothing, but because it never
     happened.
  2. It fed the model **`token_type_ids`**. bge-reranker-v2-m3 is XLM-RoBERTa, which
     has no segment embeddings — that is a BERT input. Every inference threw
     `Invalid input name: token_type_ids`.
  3. It read the logits as **`f32`**. The checkpoint emits `f16` (needs ort's `half`
     feature). This one only surfaced once (2) was fixed.

  Two of these were architectural mismatches, obvious from a single error message.
  **Nobody saw the message, because bug 1 ate it.** A feature cannot earn anything
  when its results are thrown away. Fixed and being measured properly.

- **"Associative retrieval degrades all four metrics"** — the conclusion holds, the
  evidence did not. −0.03 at n=10 is a quarter of the error bar. But the correct test
  for two configurations over the *same* queries is a **paired** one, and under a
  paired bootstrap on the new dataset the interval is **[−0.051, −0.018]** and never
  touches zero: it improves **0** queries and hurts **23**. The decision to disable it
  was right; the reasoning was not. *The power was never in more data — it was in
  using the right test.*

### The real numbers

The system's nDCG is not 0.894. On 221 id-scored queries it is **0.50** [95% CI
0.44–0.56]. It did not get worse; it was never 0.894.

`compact` saves **28% of tokens** (not 40%) at **exactly identical nDCG** — identical
to four decimal places, because a response format cannot change *which* documents rank,
only how they are printed. That the old benchmark measured a quality cost for
truncation was itself an artefact: truncating the text removed the marker substrings it
was grading on.

The **+21.2 nDCG for bge-m3 is withdrawn.** The direction is almost certainly right;
the magnitude came from the broken benchmark and re-establishing it would mean
re-embedding the corpus twice.

### Fixed — the benchmark itself

- **Relevance was judged by substring match.** A result counted as correct if its
  text merely *contained* a marker word, so every observation mentioning "postgres"
  scored as a right answer to any question about postgres — whether it answered
  anything or not. That measures keyword presence, not retrieval, and it biases the
  whole benchmark toward the lexical branch and against the vector one. Ground truth
  is now a set of observation **ids** per query (TREC-style qrels).
- **nDCG normalized against what was RETRIEVED, not what EXISTS.** With 5 relevant
  documents in the corpus and 2 found, the "ideal" ranking was taken to be those 2 —
  so a system that missed 60% of the answer scored a **perfect 1.0**. The ideal is
  now built from `min(total_relevant, k)`, so documents you failed to retrieve count
  against you. This makes the numbers go **down**, which is the expected direction
  when you stop grading on a curve you drew yourself.
- **R@10 = 3.125 shipped in the README.** Recall is a proportion. The denominator
  was the count of *marker strings*, not of relevant *documents*.
- **Every metric now carries a bootstrap 95% interval** (Efron 1979, deterministic
  resampling) and the run reports its **minimum detectable effect**. A benchmark that
  cannot see a 5-point change should not be used to claim a 3-point regression.

### Added

- **`cuba-memorys dedupe`** — entities that are the same thing under different names.
  `cuba_alma create` inserts with `ON CONFLICT (name)`: a different string is a
  different entity, so one project fragments into `Mapupita-Web`, `Mapupitta-Web`
  (typo), `Mapupita Web`, `mapupita`… On the live brain: **266 entities, 158 (59%)
  with not a single relation** — for PageRank and multi-hop retrieval they do not
  exist.

  The infrastructure to fix this was already present and dead: `brain_entity_aliases`
  has a schema, indexes, and a `resolve_entity()` that matches exactly and fuzzily.
  Zero rows; nothing called the function. Merging now writes the old name there, so
  nothing is lost.

  **What decides a merge is not the embedding centroid.** That was the obvious idea
  and it is wrong: `M-Codes Reference Guide` and `G-Codes Reference Guide` sit at
  **0.811 cosine** between centroids. On a corpus about one domain, centroid
  similarity measures the domain, not the entity — trusting a 0.80 threshold would
  have merged two different CNC guides irreversibly. So `--apply` merges only what is
  *provable* (identical after normalizing case and separators), and everything else
  is shown, or judged one by one with `--judge`.

  (The LLM judge, asked whether `Mapupitta-Web` and `Mapupita-Web` were the same
  entity, first answered *"different — there are separate memory records for each"*.
  That is the bug offered as proof there is no bug. The prompt now disarms that
  argument explicitly, and a test pins it.)

- **`reranker_degraded` in the search response.** You asked for reranking, the
  cross-encoder threw on every pair, and you got the RRF order back looking exactly
  like a reranked one. Same reason `degraded` exists for the vector branch: an agent
  handed a silently un-reranked top-10 will simply trust it.

### Fixed — the CLI was eating your flags

- **`search "x" --format verbose` searched, literally, for «x --format verbose».**
  Unknown flags fell into the catch-all and were **concatenated onto the query**. It
  returned nothing, with no hint why.
- **`save "x" --importancia 0.9` stored «x --importancia 0.9» AS THE MEMORY CONTENT.**
  Same catch-all. This one corrupts data.
- Both, plus `delete`, now reject unknown `--flags` with a usage error. Same family as
  the `--batch 64` that `reembed` silently ignored: an argument a tool pretends not to
  see is an argument that lies about what it did.

### Added — build limits

`.cargo/config.toml` (3 jobs) and a `quick` profile (`lto = "thin"`, 16 codegen units).
The release profile's fat LTO with `codegen-units = 1` peaked past 8 GB in a single
unit and froze a 14.9 GB laptop running zram. Use `--profile quick` to iterate;
`--release` only to measure and ship.

## [0.11.2] — 2026-07-13 (Cargo `0.11.2` · npm `0.11.2` · PyPI `1.13.2`)

The anti-hallucination feature was hallucinating. Found by pointing the demo at it.

### ⚠ Breaking — `cuba_faro mode=verify` now calls an LLM judge

Verification escalates its evidence to a judge and derives confidence from the
verdicts. It costs a model call (free via MCP sampling — your client's model — or a
local `claude` CLI) and takes a few seconds. With no judge available it answers
`unknown` instead of inventing a verdict. Response gains `interpretation`,
`judged_by`, and a per-evidence `verdict`/`reason`.

### Fixed

- **`verify` scored false claims HIGHER than true ones.** Confidence came from
  cosine similarity to the retrieved evidence — and similarity measures what a text
  is *about*, not what it *asserts*. "cuba-memorys is written in Rust" and "…in
  Java" are nearly the same vector: same subject, same shape, one word apart.
  Measured on the live 1,461-observation corpus:

  | claim | before | after |
  |---|---|---|
  | "usa RRF con k=60" (true) | 0.59 | **0.83 · verified** |
  | "está escrito en Java" (false) | **0.61** | **0.00 · contradicted** |
  | "la mejor paella lleva azafrán" (unrelated) | 0.45, 10 "evidence" items | **0.00 · unknown**, none |

  No threshold could have fixed it — true claims landed at 0.43–0.57 similarity and
  false ones at 0.55–0.59, completely overlapping. Entailment is a different
  question from similarity and needs something that reads. Evidence below a
  similarity floor is now discarded (retrieval always returns its top-K; that is
  right for search and wrong for verification), and what survives goes to a judge.
  Verdicts are weighted by similarity, so similarity decides how much a verdict
  counts — never what the verdict is. "Unrelated" contributes to neither side: being
  on-topic is not support.

- **`cuba_juez` with the `claude_cli` backend never worked.** `claude --print
  --output-format json` returns a report *about* the call, with the model's answer
  as a string field inside it. The parser took the first `{` and last `}` — that
  envelope — found no `verdict`, and fell back to "unknown". Since v0.8. The
  heuristic quietly did all the work while the logs showed a model being called.

- **Setting `ONNX_MODEL_PATH` without `ORT_DYLIB_PATH` hung the server.** `ort` loads
  the runtime dynamically; when it cannot find the library it does not error, the
  process just stops answering — after starting, connecting, migrating and
  announcing itself ready. It logs an ERROR and degrades to lexical search now.

- **`compact` reported `"i": null`** on most results. Only the vector branch failed
  to select `importance`, and a semantic hit usually wins the fusion — so the field
  looked broken exactly where it mattered.

- Judge verdicts are fetched **concurrently**. Serially, a three-evidence verify cost
  over a minute of wall clock and would have been unusable however correct it was.

### Changed

- **README rewritten** for someone arriving new, not for someone who followed the
  version history. Every number in it is checked against the code by a test or was
  measured — the old one claimed 25 tools (there are 28), pinned installs to
  versions two releases stale, and documented none of the 13 CLI commands.
- **The demo no longer writes to your database.** It defaulted `DATABASE_URL` to the
  real brain on `:5488`, so recording the README GIF created entities in a live
  memory store and ran PageRank over it. It now starts a throwaway Postgres and
  destroys it on exit, and ignores your embedding config rather than inheriting it.

## [0.11.1] — 2026-07-13 (Cargo `0.11.1` · npm `0.11.1` · PyPI `1.13.1`)

Two bugs found by *using* v0.11.0 rather than testing it — both in the same family
as the ones v0.11.0 set out to kill.

### Fixed

- **Every new memory was stamped with the wrong model name.** `embeddings::onnx`
  exposed a `pub const CURRENT_MODEL = "multilingual-e5-small"` beside a
  `current_model()` that reads `CUBA_EMBED_MODEL`. The split was perverse: every
  site that **wrote** an embedding used the constant, every site that **compared**
  one used the function. So on a bge-m3 corpus, each new observation got a correct
  1024-d bge-m3 vector labelled with a 384-d model that had not run in months —
  permanently stale to `doctor`, whose warning count could only grow, and to
  `zafra reembed`, which could never converge: it re-encoded the row, and the next
  write re-mislabelled it.

  The vectors were always fine (measured, not assumed: cross-label cosine on
  same-entity pairs sits inside the range of within-label cosine — one vector
  space, not two). Only the name lied. But that name is what tells you, after the
  next model change, which rows still need re-encoding. `CURRENT_MODEL` is private
  now, so the compiler forbids the mistake — and it immediately found a fifth site:
  a smoke test asserting the constant's value, which had pinned the bug in place.

  Only affects setups that override `CUBA_EMBED_MODEL`; on the default model the
  label was accidentally correct.

- **`reembed`'s smallest unit of work was "everything".** One observation missing a
  vector, and the only cure on offer was to recompute all 1,461 — overwriting 1,460
  good vectors to fill one empty. It now re-encodes the stale set by default (no
  vector, or tagged with another model), which is right in both real cases without
  a flag: changing models makes every row qualify; a single failed embedding makes
  exactly one. `--all` still forces the full pass.

- **`reembed --batch 64` was silently ignored** — only `--batch=64` parsed, and the
  space-separated form fell into a catch-all that dropped it. Both forms work now,
  and an unrecognised argument is an error instead of a shrug.

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
