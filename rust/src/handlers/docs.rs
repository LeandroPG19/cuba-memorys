//! `cuba_docs` — read a library's real documentation instead of remembering it wrong.
//!
//! This is the one capability worth keeping from cuba-search, and it exists for a
//! failure this project has hit repeatedly: a model confidently writing an API that was
//! renamed two versions ago. A memory server can tell you what *you* decided; it cannot
//! tell you what `tokio::spawn` signature shipped last month. This can.
//!
//! It is behind the `docs` Cargo feature, and every request goes through
//! [`crate::net::guard`]. cuba-memorys makes no outbound requests without both.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::sync::{Mutex, OnceLock};

use crate::net::fetch;
use crate::search::cache::TtlLruCache;

/// Documentation is not a memory: it is fetched, read once, and worth keeping for the
/// rest of the session rather than re-fetched for every follow-up question.
static DOC_CACHE: OnceLock<Mutex<TtlLruCache<String>>> = OnceLock::new();

fn cache() -> &'static Mutex<TtlLruCache<String>> {
    DOC_CACHE.get_or_init(|| Mutex::new(TtlLruCache::new()))
}

/// Where a library's documentation actually lives.
///
/// Guessing a URL from a name is how you end up scraping a squatted domain, so the
/// mapping is explicit for the ecosystems whose layout is predictable, and refuses
/// otherwise. A wrong answer here is worse than no answer: it is a *confident* wrong
/// answer, fetched over the network, that the caller has no reason to doubt.
fn resolve(library: &str) -> Result<String> {
    let lib = library.trim().to_lowercase();
    if lib.is_empty() {
        anyhow::bail!("¿documentación de qué? `library` está vacío");
    }
    // Names come from a model, and a model will eventually pass a URL, a path, or a
    // sentence. Only a package name is a package name.
    if !lib
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        anyhow::bail!(
            "`{library}` no parece un nombre de paquete. Pasá sólo el nombre \
             (p.ej. `tokio`, `sqlx`, `fastapi`), no una URL ni una frase."
        );
    }

    // docs.rs renders every published Rust crate at a predictable path, which is why
    // Rust gets a rule and the rest get a list.
    Ok(match lib.as_str() {
        // Python
        "fastapi" => "https://fastapi.tiangolo.com/".into(),
        "pydantic" => "https://docs.pydantic.dev/latest/".into(),
        "sqlalchemy" => "https://docs.sqlalchemy.org/en/20/".into(),
        "httpx" => "https://www.python-httpx.org/".into(),
        "pytest" => "https://docs.pytest.org/en/stable/".into(),
        "structlog" => "https://www.structlog.org/en/stable/".into(),
        "granian" => "https://github.com/emmett-framework/granian".into(),
        // JS/TS
        "react" => "https://react.dev/reference/react".into(),
        "next" | "nextjs" | "next.js" => "https://nextjs.org/docs".into(),
        "zod" => "https://zod.dev/".into(),
        "zustand" => "https://zustand.docs.pmnd.rs/".into(),
        "tailwind" | "tailwindcss" => "https://tailwindcss.com/docs".into(),
        "vitest" => "https://vitest.dev/guide/".into(),
        // Infra
        "postgresql" | "postgres" => "https://www.postgresql.org/docs/current/".into(),
        "pgvector" => "https://github.com/pgvector/pgvector".into(),
        "docker" => "https://docs.docker.com/".into(),
        // Anything else: assume a Rust crate. Wrong for a Python package that shares a
        // name with one, and docs.rs answers 404 for a crate that does not exist —
        // both of which are visible failures, not silent ones.
        other => format!(
            "https://docs.rs/{other}/latest/{}/",
            other.replace('-', "_")
        ),
    })
}

/// Fetch and flatten a library's documentation.
///
/// `query` does not search the site — it filters the fetched text to the paragraphs
/// that mention it, which is the difference between handing a model 400 KB of
/// navigation chrome and handing it the three paragraphs about the function it asked
/// about.
pub async fn handle(args: &Value) -> Result<Value> {
    let library = args
        .get("library")
        .and_then(Value::as_str)
        .context("falta `library`")?;
    let query = args.get("query").and_then(Value::as_str).unwrap_or("");

    let url = resolve(library)?;

    let cached = cache()
        .lock()
        .map_err(|e| anyhow::anyhow!("cache envenenada: {e}"))?
        .get(&url);

    let text = match cached {
        Some(t) => t,
        None => {
            let html = fetch::get(&url)
                .await
                .with_context(|| format!("trayendo la documentación de `{library}` desde {url}"))?;
            let t = fetch::html_to_text(&html, 100);
            cache()
                .lock()
                .map_err(|e| anyhow::anyhow!("cache envenenada: {e}"))?
                .put(url.clone(), t.clone());
            t
        }
    };

    let (body, filtered) = match query.trim() {
        "" => (truncate(&text, 8_000), false),
        q => {
            let hits = grep(&text, q);
            if hits.is_empty() {
                // Say that the filter found nothing, rather than returning the whole
                // page as if it had. A model handed 8 KB of unrelated docs will find
                // something in them.
                return Ok(json!({
                    "library": library,
                    "url": url,
                    "query": q,
                    "found": false,
                    "documentation": null,
                    "note": format!(
                        "La documentación de `{library}` se descargó bien, pero no menciona «{q}». \
                         No se devuelve el resto de la página: no responde a lo que preguntaste."
                    ),
                }));
            }
            (truncate(&hits, 8_000), true)
        }
    };

    Ok(json!({
        "library": library,
        "url": url,
        "query": if query.is_empty() { Value::Null } else { json!(query) },
        "found": true,
        "filtered": filtered,
        "documentation": body,
    }))
}

/// Paragraphs mentioning the query, with the one before and after for context — a
/// signature is useless without the sentence that says what it does.
fn grep(text: &str, query: &str) -> String {
    let needle = query.to_lowercase();
    let paras: Vec<&str> = text.split("\n\n").collect();
    let mut keep: Vec<usize> = Vec::new();

    for (i, p) in paras.iter().enumerate() {
        if p.to_lowercase().contains(&needle) {
            keep.extend([i.saturating_sub(1), i, (i + 1).min(paras.len() - 1)]);
        }
    }
    keep.sort_unstable();
    keep.dedup();
    keep.iter()
        .map(|&i| paras[i])
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}\n\n[…truncado en {max} caracteres]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_crates_resolve_to_docs_rs() {
        assert_eq!(
            resolve("tokio").unwrap(),
            "https://docs.rs/tokio/latest/tokio/"
        );
        // El guion del paquete es un guion bajo en el nombre del módulo.
        assert_eq!(
            resolve("tiktoken-rs").unwrap(),
            "https://docs.rs/tiktoken-rs/latest/tiktoken_rs/"
        );
    }

    #[test]
    fn known_ecosystems_have_explicit_urls() {
        assert!(resolve("fastapi").unwrap().contains("tiangolo"));
        assert!(resolve("React").unwrap().contains("react.dev"));
    }

    /// A model WILL eventually pass a URL here, and that URL must not become a fetch
    /// target that skipped the name check.
    #[test]
    fn a_url_is_not_a_package_name() {
        for bad in [
            "http://169.254.169.254/",
            "../../etc/passwd",
            "tokio; rm -rf /",
            "",
        ] {
            assert!(resolve(bad).is_err(), "`{bad}` no es un nombre de paquete");
        }
    }
}
