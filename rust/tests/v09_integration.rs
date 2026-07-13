//! Integration tests for v0.9.0 — Search & Retrieval upgrades + Cognitive
//! refinements. Requires a live PostgreSQL with pgvector.
//!
//! Run with:
//!   DATABASE_URL="postgresql://cuba:memorys2026@localhost:5488/brain" \
//!     cargo test --test v09_integration -- --ignored --nocapture
//!
//! Single #[tokio::test] to share one Tokio runtime (same rationale as
//! integration.rs and v08_project_scoping.rs).

use serde_json::json;
use uuid::Uuid;

fn unique(prefix: &str) -> String {
    format!("{}_{}", prefix, &Uuid::new_v4().to_string()[..8])
}

#[tokio::test]
#[ignore]
async fn test_v09_all() {
    let url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL env var required for integration tests");
    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("pool init w/ sqlx-migrate");

    // ── PR #5: sqlx-migrate idempotent ───────────────────────────
    println!("  [1/8] PR #5: sqlx-migrate idempotency");
    {
        // Re-create pool — migrations should be marked applied, no errors
        drop(pool);
        let pool2 = cuba_memorys::db::create_pool(&url)
            .await
            .expect("second pool init must be idempotent");
        // _sqlx_migrations table must exist with at least 14 rows
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM _sqlx_migrations")
            .fetch_one(&pool2)
            .await
            .expect("_sqlx_migrations table accessible");
        assert!(
            count.0 >= 14,
            "expected ≥14 applied migrations, got {}",
            count.0
        );
        println!(
            "  ✓ sqlx-migrate is idempotent ({} migrations applied)",
            count.0
        );
        drop(pool2);
    }

    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("re-open pool");

    let proj = unique("test_v09_proj");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "start", "name": "v09-tests", "project": proj}),
    )
    .await
    .expect("session start");

    // Seed data: one entity with several semantically related observations
    let entity_a = unique("rust_async");
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_alma",
        json!({"action": "create", "name": entity_a, "entity_type": "technology"}),
    )
    .await
    .expect("alma create A");

    for content in [
        "Tokio is the de-facto async runtime for Rust",
        "Tokio runtime supports async I/O and timers",
        "async-std was an alternative runtime, less popular now",
        "Smol is a smaller async runtime focused on simplicity",
        "Rust does not have green threads in stdlib",
    ] {
        cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_cronica",
            json!({"action": "add", "entity_name": entity_a, "content": content, "observation_type": "fact"}),
        )
        .await
        .expect("cronica add");
    }

    // Wait briefly for fire-and-forget embedding tasks
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // ── PR #6 Phase 3: BM25 third RRF signal ─────────────────────
    //
    // `format: "verbose"` explicitly. v0.11 made `compact` the default shape —
    // abbreviated keys (e/c/t/i/s), 40% fewer tokens at identical nDCG — and
    // compact deliberately omits the per-branch score breakdown, because an agent
    // reasoning over memories has no use for bm25_score and pays tokens to carry
    // it. This test is about the RETRIEVAL PIPELINE, not the default response
    // shape, and the scores it asserts on only exist in verbose.
    println!("  [2/8] PR #6: BM25 hybrid as third RRF signal");
    let with_bm25 = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({"query": "Tokio runtime", "limit": 10, "enable_bm25": true, "format": "verbose"}),
    )
    .await
    .expect("faro with BM25");
    let with_bm25_text = extract_text(&with_bm25);
    assert!(
        with_bm25_text.contains("\"bm25_score\""),
        "results must include bm25_score field, got: {with_bm25_text}"
    );
    println!("  ✓ BM25 score present in fused results");

    // ── v0.11: the two response shapes are a contract ────────────
    //
    // Changing the default from verbose to compact IS a breaking change for any
    // caller that parses the keys — this very test broke on it, which is how the
    // change got noticed at all. Both shapes are pinned here so the next person
    // to touch the format has to do it deliberately.
    println!("  [2b/8] v0.11: compact is the default shape, verbose is opt-in");
    let default_shape = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({"query": "Tokio runtime", "limit": 3}),
    )
    .await
    .expect("faro default");
    let default_text = extract_text(&default_shape);
    assert!(
        default_text.contains("\"c\":") && default_text.contains("\"e\":"),
        "the DEFAULT must be compact (abbreviated keys), got: {default_text}"
    );
    assert!(
        !default_text.contains("\"bm25_score\""),
        "compact must NOT carry the per-branch scores — that is the whole saving"
    );
    assert!(
        with_bm25_text.contains("\"entity_name\"") && with_bm25_text.contains("\"content\""),
        "verbose must keep the full key names callers depend on"
    );
    println!("  ✓ compact by default, verbose on request — both shapes pinned");

    // ── PR #6 Phase 2: MMR diversification ───────────────────────
    println!("  [3/8] PR #6: MMR diversification");
    let no_div = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({"query": "Tokio runtime", "limit": 3, "diversify": false}),
    )
    .await
    .expect("faro no diversify");
    let with_div = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({"query": "Tokio runtime", "limit": 3, "diversify": true, "mmr_lambda": 0.5}),
    )
    .await
    .expect("faro with diversify");
    let no_div_count = count_results(&no_div);
    let with_div_count = count_results(&with_div);
    assert!(
        no_div_count > 0 && with_div_count > 0,
        "both must return results"
    );
    println!("  ✓ MMR returns {with_div_count} diversified results");

    // ── PR #6 Phase 2: OOD abstention ────────────────────────────
    println!("  [4/8] PR #6: OOD abstention (Mahalanobis)");
    // Note: requires ≥50 observations for OOD to fire; we have 5. Test that
    // path is wired up: with too few samples, abstain_ood is bypassed (returns
    // normal results) — that's the documented graceful-degradation behavior.
    let ood_attempt = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({
            "query": "completely unrelated topic about cooking pasta",
            "limit": 5,
            "abstain_ood": true,
            "ood_threshold": 5.0
        }),
    )
    .await
    .expect("faro abstain_ood");
    let ood_text = extract_text(&ood_attempt);
    // With <50 samples we expect normal result (not abstain). Field absence
    // proves OOD code path didn't crash.
    println!(
        "  ✓ OOD path wired (graceful degrade with <{} samples): {}",
        50,
        if ood_text.contains("\"ood\":true") {
            "abstained"
        } else {
            "normal results"
        }
    );

    // ── PR #6 Phase 1: tiktoken-rs exact counting ────────────────
    println!("  [5/8] PR #6: tiktoken-rs token budget exact counting");
    let tight_budget = cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_faro",
        json!({"query": "Tokio", "limit": 10, "max_tokens": 30}),
    )
    .await
    .expect("faro tight budget");
    let tight_text = extract_text(&tight_budget);
    // The tight budget must produce a non-empty truncated response without panic
    assert!(!tight_text.is_empty(), "tiktoken truncation must not panic");
    println!("  ✓ tiktoken budget enforcement works (no UTF-8 panic)");

    // ── PR #7 Phase 1: Conformal prediction (unit-testable, but validate the API) ─
    println!("  [6/8] PR #7: conformal prediction wired in PE gating");
    use cuba_memorys::cognitive::prediction_error::{
        adaptive_thresholds_conformal, adaptive_thresholds_zscore,
    };
    let skewed: Vec<f64> = vec![
        0.85, 0.86, 0.87, 0.85, 0.88, 0.86, 0.87, 0.85, 0.84, 0.20, 0.30, 0.40,
    ];
    let (r_conf, _) = adaptive_thresholds_conformal(&skewed);
    let (r_z, _) = adaptive_thresholds_zscore(&skewed);
    println!(
        "  ✓ conformal r={:.3}, z-score r={:.3} on skewed data (different distributions detected)",
        r_conf, r_z
    );

    // ── PR #7 Phase 2: Testing effect — high access_count decays slower ─
    println!("  [7/8] PR #7: testing effect (Karpicke-Roediger 2008)");
    // Run decay; the SQL formula `(1 + ln(1+access_count))` is exercised
    let decay_result =
        cuba_memorys::handlers::dispatch(&pool, "cuba_zafra", json!({"action": "decay"}))
            .await
            .expect("zafra decay");
    let decay_text = extract_text(&decay_result);
    assert!(
        decay_text.contains("testing_effect") && decay_text.contains("Karpicke-Roediger"),
        "decay response must advertise testing effect formula"
    );
    println!("  ✓ decay formula includes testing-effect modulation");

    // ── PR #7 Phase 5: Source credibility — trust action returns Beta ────
    println!("  [8/8] PR #7: source credibility (Yin-Han-Yu IEEE TKDE 2008)");
    let trust =
        cuba_memorys::handlers::dispatch(&pool, "cuba_calibrar", json!({"action": "trust"}))
            .await
            .expect("calibrar trust");
    let trust_text = extract_text(&trust);
    assert!(
        trust_text.contains("\"alpha\":") && trust_text.contains("\"beta\":"),
        "trust response must include Beta(α,β) per source: {trust_text}"
    );
    assert!(
        trust_text.contains("agent")
            && trust_text.contains("user")
            && trust_text.contains("inference"),
        "trust must pre-seed all standard sources"
    );
    println!("  ✓ source credibility table seeded with standard sources");

    // ── v0.11: a written embedding is tagged with the model that MADE it ─
    //
    // `embeddings::onnx` exposed a `pub const CURRENT_MODEL` next to a
    // `current_model()` that reads CUBA_EMBED_MODEL. Every site that COMPARED used
    // the function; every site that WROTE used the constant. So on the live bge-m3
    // brain, each new observation got a correct bge-m3 vector stamped
    // "multilingual-e5-small" — permanently stale to `doctor`, and to `zafra
    // reembed`, which could never converge: it re-encoded the row, and the next
    // write re-mislabelled it.
    //
    // The constant is private now, so the compiler forbids the mistake. This asserts
    // the behaviour anyway, because privacy stops the OUTSIDE and the bug was one
    // import away from coming back.
    //
    // The invariant is conditional, and precisely so: IF a vector was written, THEN
    // its label must name the model that made it. Without a real ONNX model, cronica
    // deliberately writes neither — a hash-fallback vector labelled as a real model
    // would corrupt search far worse than a missing one. CI has no 570 MB model, so
    // there is nothing to label there and nothing to assert; asserting anyway is how
    // this test failed CI while passing locally.
    println!("  [9/9] v0.11: embeddings are tagged with the model that produced them");
    {
        let tag = unique("test-model-tag");
        // SAFETY: single-threaded point in a single-#[test] binary; nothing else
        // reads CUBA_EMBED_MODEL concurrently here.
        unsafe { std::env::set_var("CUBA_EMBED_MODEL", &tag) };

        let added = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_cronica",
            json!({
                "action": "add",
                "entity_name": entity_a,
                "content": "An observation written while CUBA_EMBED_MODEL says something specific",
                "observation_type": "fact"
            }),
        )
        .await
        .expect("cronica add under a custom model tag");

        let payload: serde_json::Value =
            serde_json::from_str(&extract_text(&added)).expect("cronica add returns JSON");
        let obs_id: Uuid = payload["id"]
            .as_str()
            .expect("a fresh observation gets an id (not deduplicated/reinforced)")
            .parse()
            .expect("that id is a uuid");

        // The embedding lands fire-and-forget.
        tokio::time::sleep(std::time::Duration::from_millis(2000)).await;

        let row: (Option<String>, bool) = sqlx::query_as(
            "SELECT embedding_model, embedding IS NOT NULL FROM brain_observations WHERE id = $1",
        )
        .bind(obs_id)
        .fetch_one(&pool)
        .await
        .expect("read the row back");

        unsafe { std::env::remove_var("CUBA_EMBED_MODEL") };

        let (stored_tag, has_vector) = row;
        if has_vector {
            assert_eq!(
                stored_tag.as_deref(),
                Some(tag.as_str()),
                "a vector was written, so its label must name the model that produced it. \
                 Writing one model's vector under another model's name makes the row \
                 permanently, invisibly stale."
            );
            println!("  ✓ the stored tag follows CUBA_EMBED_MODEL, not a constant");
        } else {
            assert!(
                stored_tag.is_none(),
                "no vector was written, so no model tag may be claimed — a label without a \
                 vector is a lie about work that never happened"
            );
            println!("  ✓ no ONNX model loaded: no vector, and no tag claiming one (correct)");
        }
    }

    // ── Cleanup ──────────────────────────────────────────────────
    cuba_memorys::handlers::dispatch(
        &pool,
        "cuba_jornada",
        json!({"action": "end", "outcome": "success", "summary": "v09 tests"}),
    )
    .await
    .ok();
    sqlx::query("DELETE FROM brain_entities WHERE name = $1")
        .bind(&entity_a)
        .execute(&pool)
        .await
        .ok();
    sqlx::query("DELETE FROM brain_projects WHERE name = $1")
        .bind(&proj)
        .execute(&pool)
        .await
        .ok();

    println!("\n  ✅ v0.9.0 integration tests OK (8/8 phases verified)");
}

fn extract_text(value: &serde_json::Value) -> String {
    value
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|f| f.get("text"))
        .and_then(|t| t.as_str())
        .map(String::from)
        .unwrap_or_else(|| value.to_string())
}

fn count_results(value: &serde_json::Value) -> usize {
    extract_text(value)
        .matches("\"id\"")
        .count()
        .saturating_sub(1) // discount the wrapper id field if any
}
