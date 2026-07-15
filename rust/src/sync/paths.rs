use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const ENV_VAR: &str = "CUBA_SYNC_DIR";
const DEFAULT_DIR: &str = ".cuba-memorys";

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

pub fn ensure_within(root: &Path, candidate: &Path) -> Result<()> {
    let normalized = lexical_join(root, candidate);
    if !normalized.starts_with(root) {
        anyhow::bail!("path traversal blocked: {candidate:?} escapes root {root:?}");
    }

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
