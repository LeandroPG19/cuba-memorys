//! `cuba-memorys reembed` — re-encode what needs re-encoding.
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
//!
//! It used to re-encode the ENTIRE corpus, unconditionally, and that made it a
//! tool you could not reach for. One observation missing a vector? The only cure on
//! offer was to recompute all 1,461 — overwriting 1,460 good vectors to fix one
//! empty. A maintenance command whose smallest unit of work is "everything" is a
//! command people avoid, and the thing it was supposed to fix stays broken.
//!
//! The default is now the stale set: rows with no vector, or tagged with a model
//! other than the one loaded. That covers both real cases without a flag —
//! changing models leaves every row "tagged with another model", so all of them
//! qualify; a single failed embedding leaves exactly one. `--all` remains for
//! forcing the issue.

use anyhow::{Context, Result};

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut batch: i64 = 500;
    let mut all = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys reembed [--all] [--batch N]\n\n\
                     Re-encodes the observations that need it: missing a vector, or tagged\n\
                     with a model other than the one currently loaded. Run it after changing\n\
                     CUBA_EMBED_MODEL / CUBA_EMBEDDING_DIM / ONNX_MODEL_PATH — and after\n\
                     scripts/migrate-embedding-dim.sh has retyped the column.\n\n\
                     --all       re-encode every observation, stale or not\n\
                     --batch N   rows per progress report (default 500)\n\n\
                     Refuses to run without a real ONNX model: re-embedding with the hash\n\
                     fallback would overwrite meaningful vectors with noise."
                );
                return Ok(());
            }
            "--all" => all = true,
            // Accept both `--batch=N` and `--batch N`. The second form used to be
            // swallowed in silence, so `--batch 64` ran with the default and said
            // nothing — an argument the tool ignores is an argument that lies.
            "--batch" => {
                let n = it.next().context("--batch needs a number: --batch 500")?;
                batch = n.parse().context("--batch needs an integer")?;
            }
            other => {
                if let Some(n) = other.strip_prefix("--batch=") {
                    batch = n.parse().context("--batch needs an integer")?;
                } else {
                    anyhow::bail!("reembed: argumento desconocido `{other}` (probá --help)");
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

    let rows: Vec<(uuid::Uuid, String)> = if all {
        sqlx::query_as("SELECT id, content FROM brain_observations ORDER BY id")
            .fetch_all(&pool)
            .await
    } else {
        sqlx::query_as(
            "SELECT id, content FROM brain_observations
             WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM $1
             ORDER BY id",
        )
        .bind(&model)
        .fetch_all(&pool)
        .await
    }
    .context("reading observations")?;

    let total = rows.len();
    if total == 0 {
        println!("Nada que reembeber: todo el corpus ya está en {model} ({dim}-d).");
        return Ok(());
    }

    let scope = if all { "todas" } else { "pendientes" };
    eprintln!("reembed: {total} observaciones ({scope})");

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
    if all || total > 100 {
        println!(
            "\nEl umbral de abstención estaba calibrado para el modelo anterior y ya no vale:"
        );
        println!("  cuba-memorys calibrate --dataset <dataset.jsonl> --apply");
    }
    Ok(())
}
