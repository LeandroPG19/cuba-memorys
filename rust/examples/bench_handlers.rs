//! Benchmark: Rust handler latency + embedding performance.
//!
//! Usage:
//!   DATABASE_URL="postgresql://user:pass@localhost:5433/cuba_memorys_test" \
//!     cargo run --release --example bench_handlers

use std::time::Instant;

fn fmt_duration(d: std::time::Duration) -> String {
    if d.as_millis() > 0 {
        format!("{:.2}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.0}µs", d.as_secs_f64() * 1_000_000.0)
    }
}

#[tokio::main]
async fn main() {
    let url = std::env::var("DATABASE_URL").expect("Set DATABASE_URL to run benchmarks");

    let pool = cuba_memorys::db::create_pool(&url)
        .await
        .expect("Failed to connect");

    println!("\n  ═══ cuba-memorys Rust Benchmark ═══\n");

    // ── 1. Embedding (hash fallback) ──────────────────────────────
    {
        let texts = [
            "Rust is a systems programming language focused on safety",
            "The quick brown fox jumps over the lazy dog",
            "CNC machining requires precise tool path calculations",
            "Knowledge graphs store entities and their relationships",
        ];
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            for text in &texts {
                let _ = cuba_memorys::embeddings::onnx::embed(text).await;
            }
        }
        let elapsed = start.elapsed();
        let per_embed = elapsed / (iterations * texts.len() as u32);
        let is_onnx = cuba_memorys::embeddings::onnx::is_model_loaded();
        println!(
            "  embed ({})      {:>8} /call  ({} calls)",
            if is_onnx { "ONNX" } else { "hash" },
            fmt_duration(per_embed),
            iterations * texts.len() as u32
        );
    }

    // ── 2. alma::create ───────────────────────────────────────────
    {
        let iterations = 100;
        let start = Instant::now();
        for i in 0..iterations {
            let name = format!("bench_alma_{i}_{}", uuid::Uuid::new_v4());
            let _ = cuba_memorys::handlers::dispatch(
                &pool,
                "cuba_alma",
                serde_json::json!({"action": "create", "name": &name, "entity_type": "concept"}),
            )
            .await;
        }
        let elapsed = start.elapsed();
        println!(
            "  alma::create    {:>8} /call  ({iterations} calls)",
            fmt_duration(elapsed / iterations)
        );
    }

    // ── 3. alma::get ──────────────────────────────────────────────
    {
        let name = format!("bench_get_{}", uuid::Uuid::new_v4());
        let _ = cuba_memorys::handlers::dispatch(
            &pool,
            "cuba_alma",
            serde_json::json!({"action": "create", "name": &name, "entity_type": "concept"}),
        )
        .await;

        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cuba_memorys::handlers::dispatch(
                &pool,
                "cuba_alma",
                serde_json::json!({"action": "get", "name": &name}),
            )
            .await;
        }
        let elapsed = start.elapsed();
        println!(
            "  alma::get       {:>8} /call  ({iterations} calls)",
            fmt_duration(elapsed / iterations)
        );
    }

    // ── 4. cronica::add ───────────────────────────────────────────
    {
        let name = format!("bench_cronica_{}", uuid::Uuid::new_v4());
        let iterations = 100;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = cuba_memorys::handlers::dispatch(
                &pool,
                "cuba_cronica",
                serde_json::json!({
                    "action": "add",
                    "entity_name": &name,
                    "content": format!("Bench observation number {i} with unique content"),
                    "observation_type": "fact",
                    "source": "agent"
                }),
            )
            .await;
        }
        let elapsed = start.elapsed();
        println!(
            "  cronica::add    {:>8} /call  ({iterations} calls)",
            fmt_duration(elapsed / iterations)
        );
    }

    // ── 5. faro::search (hybrid) ──────────────────────────────────
    {
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cuba_memorys::handlers::dispatch(
                &pool, "cuba_faro",
                serde_json::json!({"query": "rust programming safety", "mode": "hybrid", "limit": 10}),
            ).await;
        }
        let elapsed = start.elapsed();
        println!(
            "  faro::search    {:>8} /call  ({iterations} calls)",
            fmt_duration(elapsed / iterations)
        );
    }

    // ── 6. vigia::summary ─────────────────────────────────────────
    {
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cuba_memorys::handlers::dispatch(
                &pool,
                "cuba_vigia",
                serde_json::json!({"metric": "summary"}),
            )
            .await;
        }
        let elapsed = start.elapsed();
        println!(
            "  vigia::summary  {:>8} /call  ({iterations} calls)",
            fmt_duration(elapsed / iterations)
        );
    }

    println!("\n  ═══ Benchmark Complete ═══\n");

    pool.close().await;
}
