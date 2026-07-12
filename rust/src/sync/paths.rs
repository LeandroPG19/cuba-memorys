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
    // Lexical check first, and it must come first: a content-addressed write
    // targets `chunks/ab/<hash>.json`, whose parent directory does not exist yet.
    // The previous version canonicalized the parent, which fails with NotFound on
    // any path more than one level deep — so every nested write was rejected as a
    // traversal attempt. Creating the directory before validating is not an option
    // either: that is exactly how `../../etc/x` gets a directory made for it.
    //
    // Resolving `..` on the string cannot be fooled by a non-existent path.
    let normalized = lexical_join(root, candidate);
    if !normalized.starts_with(root) {
        anyhow::bail!("path traversal blocked: {candidate:?} escapes root {root:?}");
    }

    // Then, if the path (or an existing ancestor) is really there, canonicalize
    // it too. Lexical normalization cannot see a symlink pointing out of the
    // root; this catches that.
    let existing = candidate
        .ancestors()
        .find(|p| p.exists())
        .unwrap_or(candidate);
    if let Ok(real) = existing.canonicalize()
        && let Ok(real_root) = root.canonicalize()
        && !real.starts_with(&real_root)
    {
        anyhow::bail!("path traversal blocked: {candidate:?} resolves outside {root:?}");
    }
    Ok(())
}

/// Resolve `.` and `..` against `root` without touching the filesystem.
///
/// A `..` that would climb above the root is not clamped — it is allowed to walk
/// off, so the caller's `starts_with` check sees it and rejects the path.
fn lexical_join(root: &Path, candidate: &Path) -> PathBuf {
    use std::path::Component;

    let mut out = if candidate.is_absolute() {
        PathBuf::new()
    } else {
        root.to_path_buf()
    };

    for component in candidate.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
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
