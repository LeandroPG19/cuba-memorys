use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const ENV_VAR: &str = "CUBA_SYNC_DIR";
const DEFAULT_DIR: &str = ".cuba-memorys";

fn configured_root() -> Option<PathBuf> {
    std::env::var(ENV_VAR)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn target_under_root(
    root: &Path,
    override_arg: Option<&str>,
    root_is_explicit: bool,
) -> Result<PathBuf> {
    let candidate = match override_arg.map(str::trim).filter(|s| !s.is_empty()) {
        Some(raw) => {
            let requested = PathBuf::from(raw);
            if requested.is_absolute() {
                requested
            } else {
                root.join(requested)
            }
        }
        None if root_is_explicit => root.to_path_buf(),
        None => root.join(DEFAULT_DIR),
    };
    ensure_within(root, &candidate).with_context(|| {
        format!("sync directories are confined to {root:?}. Set {ENV_VAR} to work somewhere else")
    })?;
    Ok(candidate)
}

pub fn resolve_dir(override_arg: Option<&str>) -> Result<PathBuf> {
    let explicit = configured_root();
    let root_is_explicit = explicit.is_some();
    let root = match explicit {
        Some(r) => r,
        None => std::env::current_dir().context("resolving the working directory as sync root")?,
    };
    if !root.exists() {
        std::fs::create_dir_all(&root).with_context(|| format!("creating sync root {root:?}"))?;
    }

    let path = target_under_root(&root, override_arg, root_is_explicit)?;
    if !path.exists() {
        std::fs::create_dir_all(&path).with_context(|| format!("creating sync dir {path:?}"))?;
    }
    path.canonicalize()
        .with_context(|| format!("canonicalize {path:?}"))
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

    fn root() -> PathBuf {
        PathBuf::from("/home/someone/repo")
    }

    #[test]
    fn a_caller_supplied_absolute_path_cannot_leave_the_root() {
        for escape in ["/etc", "/tmp/anywhere", "/home/someone/other-repo"] {
            let denied = target_under_root(&root(), Some(escape), false);
            assert!(
                denied.is_err(),
                "cuba_sync takes `dir` straight from the MCP tool arguments and then \
                 creates it, writes the graph into it and deletes *.json from it — an \
                 absolute path would make that remote file write: {escape}"
            );
        }
    }

    #[test]
    fn dot_dot_cannot_climb_out_of_the_root() {
        for escape in ["../elsewhere", "sub/../../elsewhere", "./../.."] {
            assert!(
                target_under_root(&root(), Some(escape), false).is_err(),
                "{escape} normalizes above the root"
            );
        }
    }

    #[test]
    fn a_path_inside_the_root_is_still_allowed() {
        let ok = target_under_root(&root(), Some("exports/backup"), false)
            .expect("a subdirectory of the root is the normal case");
        assert_eq!(ok, root().join("exports/backup"));

        let dotted = target_under_root(&root(), Some("./exports/../exports"), false)
            .expect("a path that normalizes back inside is fine");
        assert_eq!(dotted, root().join("./exports/../exports"));
    }

    #[test]
    fn without_an_override_the_default_stays_where_it_always_was() {
        let implicit = target_under_root(&root(), None, false).expect("default must resolve");
        assert_eq!(
            implicit,
            root().join(DEFAULT_DIR),
            "the historical default is {DEFAULT_DIR} under the working directory, and \
             confining the root must not move it"
        );

        let configured = target_under_root(&root(), None, true).expect("explicit root resolves");
        assert_eq!(
            configured,
            *root(),
            "with {ENV_VAR} set, that directory IS the sync dir — it is not nested again"
        );
    }

    #[test]
    fn an_explicit_root_is_the_way_to_work_outside_the_working_directory() {
        let elsewhere = PathBuf::from("/var/backups/cuba");
        let inside = target_under_root(&elsewhere, Some("nightly"), true)
            .expect("a subdirectory of the configured root is allowed");
        assert_eq!(inside, elsewhere.join("nightly"));

        assert!(
            target_under_root(&elsewhere, Some("/etc"), true).is_err(),
            "the environment moves the root; it does not remove the confinement"
        );
    }

    #[test]
    fn an_empty_override_is_not_an_override() {
        for blank in ["", "   "] {
            let resolved =
                target_under_root(&root(), Some(blank), false).expect("blank falls through");
            assert_eq!(
                resolved,
                root().join(DEFAULT_DIR),
                "a blank string must not resolve to the root itself"
            );
        }
    }
}
