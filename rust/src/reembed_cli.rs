//! `cuba-memorys reembed` — re-encode the corpus with the current model.
//!
//! This existed only as an MCP tool, and that is the wrong place for it. Re-embedding
//! 1400 observations with a 570 MB model takes minutes; an MCP call is a request
//! with a timeout on both ends, and a maintenance job that outlives its own
//! protocol round-trip has no business being one. Switching embedding models —
//! the exact moment you need this — is also the moment the MCP server is in a
//! state where its clients cannot safely talk to it: the column has one dimension
//! and the loaded model has another.
//!
//! So: a plain command, run by a human, with progress on stderr.

use anyhow::{Context, Result};

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut batch: i64 = 500;
    for a in args {
        match a.as_str() {
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys reembed [--batch N]\n\n\
                     Re-encodes every observation with the currently configured model.\n\
                     Run it after changing CUBA_EMBEDDING_DIM / ONNX_MODEL_PATH, and after\n\
                     scripts/migrate-embedding-dim.sh has retyped the column.\n\n\
                     Refuses to run without a real ONNX model: re-embedding with the hash\n\
                     fallback would overwrite meaningful vectors with noise."
                );
                return Ok(());
            }
            other => {
                if let Some(n) = other.strip_prefix("--batch=") {
                    batch = n.parse().context("--batch needs an integer")?;
                }
            }
        }
    }

    // The guard that matters. The hash fallback produces vectors that are the right
    // SHAPE and semantically meaningless; writing them over a real corpus is a
    // silent, total loss of retrieval quality — and it would look like it worked.
    if !crate::embeddings::onnx::is_model_loaded() {
        anyhow::bail!(
            "no hay modelo ONNX cargado. Reembeber con el fallback de hash reemplazaría \
             los vectores por ruido sin que nada lo avise.\n\
             Comprobá ONNX_MODEL_PATH y ORT_DYLIB_PATH (cuba-memorys doctor lo diagnostica)."
        );
    }

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for reembed")?;

    let model = crate::embeddings::onnx::current_model();
    let dim = crate::embeddings::onnx::embedding_dim();
    eprintln!("reembed: modelo={model} dim={dim}");

    // Fail before writing a single row rather than halfway through. A dimension
    // mismatch mid-run leaves the corpus split across two vector spaces, which is
    // worse than either one alone.
    let col: String = sqlx::query_scalar(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute
         WHERE attrelid = 'brain_observations'::regclass AND attname = 'embedding'",
    )
    .fetch_one(&pool)
    .await
    .context("reading the embedding column type")?;
    let expected = format!("vector({dim})");
    if col != expected {
        anyhow::bail!(
            "la columna es {col} y el modelo produce {expected}.\n\
             Corré primero:  DATABASE_URL=… scripts/migrate-embedding-dim.sh {dim}"
        );
    }

    let rows: Vec<(uuid::Uuid, String)> =
        sqlx::query_as("SELECT id, content FROM brain_observations ORDER BY id")
            .fetch_all(&pool)
            .await
            .context("reading observations")?;

    let total = rows.len();
    eprintln!("reembed: {total} observaciones");

    let mut done = 0usize;
    let mut failed = 0usize;
    for chunk in rows.chunks(batch.clamp(1, 5000) as usize) {
        for (id, content) in chunk {
            match crate::embeddings::onnx::embed_passage(content).await {
                Ok(v) => {
                    sqlx::query(
                        "UPDATE brain_observations
                         SET embedding = $1, embedding_model = $2 WHERE id = $3",
                    )
                    .bind(pgvector::Vector::from(v))
                    .bind(&model)
                    .bind(id)
                    .execute(&pool)
                    .await
                    .context("writing the embedding")?;
                    done += 1;
                }
                Err(e) => {
                    // Keep going and report the count: one unembeddable row must not
                    // abort a job that has already re-encoded a thousand others.
                    tracing::warn!(error = %e, id = %id, "no se pudo embeber");
                    failed += 1;
                }
            }
        }
        eprintln!("  {done}/{total}");
    }

    println!("Reembebidas {done} de {total} observaciones con {model} ({dim}-d).");
    if failed > 0 {
        println!("{failed} fallaron (ver los warnings).");
    }
    println!("\nEl umbral de abstención estaba calibrado para el modelo anterior y ya no vale:");
    println!("  cuba-memorys calibrate --dataset <dataset.jsonl> --apply");
    Ok(())
}
