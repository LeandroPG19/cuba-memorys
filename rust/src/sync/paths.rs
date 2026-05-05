//! Path utilities for cuba_sync — resolves CUBA_SYNC_DIR and prevents
//! directory traversal escapes.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const ENV_VAR: &str = "CUBA_SYNC_DIR";
const DEFAULT_DIR: &str = ".cuba-memorys";

/// Resolve the export/import root, applying overrides from arg → env → default.
///
/// Always canonicalized to an absolute path. The directory is created if
/// missing (export will populate it; import expects it).
pub fn resolve_dir(override_arg: Option<&str>) -> Result<PathBuf> {
    let raw = override_arg
        .map(|s| s.to_string())
        .or_else(|| std::env::var(ENV_VAR).ok())
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let path = PathBuf::from(raw);
    if !path.exists() {
        std::fs::create_dir_all(&path).with_context(|| format!("creating sync dir {path:?}"))?;
    }
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize {path:?}"))?;
    Ok(canonical)
}

/// Slugify an entity name to a filesystem-safe basename.
///
/// Keeps ASCII alphanumerics + `-_`, replaces every other rune with `-`,
/// collapses runs, trims trailing dashes. Falls back to "entity" if empty.
pub fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "entity".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Reject paths that would escape `root` (defense in depth — fed paths come
/// from manifest/import, so a malicious manifest cannot cause writes outside
/// the sync dir).
pub fn ensure_within(root: &Path, candidate: &Path) -> Result<()> {
    let canonical = candidate.canonicalize().or_else(|_| {
        // path may not exist yet (we're about to create it) — check parent
        let parent = candidate
            .parent()
            .context("no parent")?
            .canonicalize()
            .with_context(|| format!("canonicalize parent of {candidate:?}"))?;
        Ok::<PathBuf, anyhow::Error>(parent)
    })?;
    if !canonical.starts_with(root) {
        anyhow::bail!("path traversal blocked: {candidate:?} escapes root {root:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slug_basic() {
        assert_eq!(slug("Postgres Database"), "postgres-database");
        assert_eq!(slug("foo/../bar"), "foo-bar");
        assert_eq!(slug("---"), "entity");
        assert_eq!(slug(""), "entity");
        assert_eq!(slug("Auth_Flow-v2"), "auth_flow-v2");
    }
}
