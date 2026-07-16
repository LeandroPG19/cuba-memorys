use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::sync::chunk::{EntityFile, ProjectRow, RelationRow};

const MARKER: &str = "# cuba-memorys hook — installed by `cuba-memorys hook install`";
const MERGE_DRIVER_NAME: &str = "cuba-memorys";

pub async fn run_cli(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("install") => install(),
        Some("merge-driver") => merge_driver(&args[1..]),
        Some("-h") | Some("--help") | None => {
            eprintln!(
                "usage: cuba-memorys hook install\n\n\
                 Wires this repo's git so the knowledge graph under .cuba-memorys/\n\
                 (or $CUBA_SYNC_DIR) stays in sync automatically:\n\
                 \x20 - post-commit  runs `sync export` after every commit\n\
                 \x20 - post-checkout runs `sync import` after checkout/branch switch\n\
                 \x20 - a git merge driver that unions observations/relations/entities\n\
                 \x20   by id instead of leaving conflict markers in graph JSON\n\n\
                 Existing hooks are appended to, never overwritten. Safe to run twice."
            );
            Ok(())
        }
        Some(other) => anyhow::bail!("unknown hook subcommand: {other} (try --help)"),
    }
}

fn git_root() -> Result<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("running `git rev-parse --show-toplevel` — is git installed?")?;
    if !out.status.success() {
        anyhow::bail!("not inside a git repository (git rev-parse failed)");
    }
    let s = String::from_utf8(out.stdout).context("git output was not utf8")?;
    Ok(PathBuf::from(s.trim()))
}

fn hooks_dir(root: &Path) -> PathBuf {
    let out = Command::new("git")
        .args(["config", "--get", "core.hooksPath"])
        .current_dir(root)
        .output();
    if let Ok(out) = out
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            let p = PathBuf::from(&s);
            return if p.is_absolute() { p } else { root.join(p) };
        }
    }
    root.join(".git").join("hooks")
}

fn append_hook_block(path: &Path, block: &str) -> Result<bool> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    if existing.contains(MARKER) {
        return Ok(false);
    }
    let mut body = existing;
    if body.is_empty() {
        body.push_str("#!/bin/sh\n");
    } else if !body.ends_with('\n') {
        body.push('\n');
    }
    body.push('\n');
    body.push_str(block);
    std::fs::write(path, body).with_context(|| format!("writing hook {path:?}"))?;
    set_executable(path)?;
    Ok(true)
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(perms.mode() | 0o111);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

fn append_gitattributes_line(root: &Path, line: &str) -> Result<bool> {
    let path = root.join(".gitattributes");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    if existing.lines().any(|l| l.trim() == line.trim()) {
        return Ok(false);
    }
    let mut body = existing;
    if !body.is_empty() && !body.ends_with('\n') {
        body.push('\n');
    }
    body.push_str(line);
    body.push('\n');
    std::fs::write(&path, body).with_context(|| format!("writing {path:?}"))?;
    Ok(true)
}

fn install() -> Result<()> {
    let root = git_root()?;
    let hooks = hooks_dir(&root);
    std::fs::create_dir_all(&hooks).context("creating hooks dir")?;

    let exe = std::env::current_exe().context("resolving path to this binary")?;
    let exe = exe.display();

    // A hook runs unattended — it must never guess which database to talk to.
    // resolve_database_url() falls back to auto-detecting ANY cuba-memorys Postgres
    // container running on the machine, which is exactly right for a human typing
    // a command but wrong for a background hook: on a box with more than one
    // cuba-memorys project (or a stray test container), it can silently export
    // from — or import into — the wrong database. So the hook resolves the URL from
    // `git config --local cuba-memorys.database-url` (persists per clone, doesn't
    // depend on the shell that fires the hook having anything exported) or, failing
    // that, its own DATABASE_URL environment variable — and is a strict no-op if
    // neither is set.
    let resolve_url_sh = "db_url=$(git config --local --get cuba-memorys.database-url 2>/dev/null || true); [ -z \"$db_url\" ] && db_url=\"$DATABASE_URL\"";
    let post_commit_block = format!(
        "{MARKER}\n\
         {resolve_url_sh}\n\
         if [ -n \"$db_url\" ]; then\n\
         \x20 DATABASE_URL=\"$db_url\" \"{exe}\" sync export --scope all >/dev/null 2>&1 || true\n\
         fi\n"
    );
    let commit_changed = append_hook_block(&hooks.join("post-commit"), &post_commit_block)?;

    let post_checkout_block = format!(
        "{MARKER}\n\
         {resolve_url_sh}\n\
         if [ -n \"$db_url\" ]; then\n\
         \x20 DATABASE_URL=\"$db_url\" \"{exe}\" sync import --conflict merge >/dev/null 2>&1 || true\n\
         fi\n"
    );
    let checkout_changed = append_hook_block(&hooks.join("post-checkout"), &post_checkout_block)?;

    Command::new("git")
        .args(["config", &format!("merge.{MERGE_DRIVER_NAME}.name"), "cuba-memorys structural merge (union by id)"])
        .current_dir(&root)
        .status()
        .context("git config merge.name")?;
    Command::new("git")
        .args([
            "config",
            &format!("merge.{MERGE_DRIVER_NAME}.driver"),
            &format!("\"{exe}\" hook merge-driver %O %A %B %P"),
        ])
        .current_dir(&root)
        .status()
        .context("git config merge.driver")?;

    let sync_dir = std::env::var("CUBA_SYNC_DIR").unwrap_or_else(|_| ".cuba-memorys".to_string());
    let attr_line = format!("{sync_dir}/** merge={MERGE_DRIVER_NAME}");
    let attrs_changed = append_gitattributes_line(&root, &attr_line)?;

    println!("post-commit hook:   {}", if commit_changed { "installed" } else { "already present" });
    println!("post-checkout hook: {}", if checkout_changed { "installed" } else { "already present" });
    println!("merge driver:       configured (merge.{MERGE_DRIVER_NAME}.driver in .git/config)");
    println!(".gitattributes:     {}", if attrs_changed { format!("added `{attr_line}`") } else { "already present".to_string() });
    println!(
        "\nNOTE: both hooks are a no-op until this repo's database is set explicitly.\n\
         They deliberately do NOT fall back to auto-detecting a running container —\n\
         on a machine with more than one cuba-memorys database, that guess can export\n\
         from, or import into, the wrong one. Set it once, it persists in .git/config:\n\
         \x20 git config --local cuba-memorys.database-url \"postgresql://...\"\n\
         (DATABASE_URL in the environment also works as a fallback.)"
    );
    Ok(())
}

fn merge_driver(args: &[String]) -> Result<()> {
    let [ancestor, ours, theirs, path] = args else {
        anyhow::bail!("usage: hook merge-driver %O %A %B %P (git passes these itself)");
    };

    let path_lower = path.to_lowercase();
    let merged: Option<Vec<u8>> = if path_lower.contains("/entities/") || path_lower.ends_with(".json") && path_lower.contains("entities") {
        merge_entity_file(ours, theirs)?
    } else if path_lower.ends_with("relations.json") {
        merge_relations(ours, theirs)?
    } else if path_lower.ends_with("projects.json") {
        merge_projects(ours, theirs)?
    } else {
        None
    };

    let _ = ancestor; // no 3-way diff needed: union-by-id doesn't require the common ancestor

    match merged {
        Some(bytes) => {
            std::fs::write(ours, bytes).with_context(|| format!("writing merged {ours}"))?;
            Ok(())
        }
        None => {
            // Unknown shape (manifest.json, embeddings.bin.zst, ...): keep ours, it will be
            // regenerated by the next `sync export` anyway. Exit 0 so git doesn't leave
            // conflict markers.
            Ok(())
        }
    }
}

fn merge_entity_file(ours_path: &str, theirs_path: &str) -> Result<Option<Vec<u8>>> {
    let a: Option<EntityFile> = read_json(ours_path)?;
    let b: Option<EntityFile> = read_json(theirs_path)?;
    let (Some(mut a), Some(b)) = (a, b) else {
        return Ok(None);
    };

    let mut by_id: HashMap<_, _> = a.observations.drain(..).map(|o| (o.id, o)).collect();
    for obs in b.observations {
        by_id.entry(obs.id).or_insert(obs);
    }
    let mut merged: Vec<_> = by_id.into_values().collect();
    merged.sort_by_key(|o| o.created_at);
    a.observations = merged;
    a.access_count = a.access_count.max(b.access_count);
    a.importance = a.importance.max(b.importance);

    Ok(Some(serde_json::to_vec_pretty(&a)?))
}

fn merge_relations(ours_path: &str, theirs_path: &str) -> Result<Option<Vec<u8>>> {
    let a: Option<Vec<RelationRow>> = read_json(ours_path)?;
    let b: Option<Vec<RelationRow>> = read_json(theirs_path)?;
    let (Some(a), Some(b)) = (a, b) else {
        return Ok(None);
    };

    let mut by_key: HashMap<(uuid::Uuid, uuid::Uuid, String), RelationRow> = HashMap::new();
    for rel in a.into_iter().chain(b) {
        let key = (rel.from_entity, rel.to_entity, rel.relation_type.clone());
        by_key
            .entry(key)
            .and_modify(|existing| {
                if rel.strength > existing.strength {
                    *existing = rel.clone();
                }
            })
            .or_insert(rel);
    }
    let mut merged: Vec<_> = by_key.into_values().collect();
    merged.sort_by_key(|r| r.created_at);

    Ok(Some(serde_json::to_vec_pretty(&merged)?))
}

fn merge_projects(ours_path: &str, theirs_path: &str) -> Result<Option<Vec<u8>>> {
    let a: Option<Vec<ProjectRow>> = read_json(ours_path)?;
    let b: Option<Vec<ProjectRow>> = read_json(theirs_path)?;
    let (Some(a), Some(b)) = (a, b) else {
        return Ok(None);
    };

    let mut by_id: HashMap<uuid::Uuid, ProjectRow> = HashMap::new();
    for p in a.into_iter().chain(b) {
        by_id.entry(p.id).or_insert(p);
    }
    let mut merged: Vec<_> = by_id.into_values().collect();
    merged.sort_by_key(|p| p.created_at);

    Ok(Some(serde_json::to_vec_pretty(&merged)?))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<Option<T>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    Ok(serde_json::from_slice(&bytes).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::chunk::ObservationRow;
    use chrono::Utc;
    use uuid::Uuid;

    fn obs(id: Uuid, content: &str) -> ObservationRow {
        ObservationRow {
            id,
            content: content.to_string(),
            observation_type: "fact".to_string(),
            source: "agent".to_string(),
            importance: 0.5,
            tags: vec![],
            project_id: None,
            session_id: None,
            created_at: Utc::now(),
            embedding_model: None,
        }
    }

    fn entity_file(observations: Vec<ObservationRow>) -> EntityFile {
        EntityFile {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            entity_type: "concept".to_string(),
            importance: 0.5,
            access_count: 0,
            project_id: None,
            created_at: Utc::now(),
            observations,
        }
    }

    #[test]
    fn union_merge_keeps_observations_unique_to_each_side() {
        let shared_id = Uuid::new_v4();
        let only_a_id = Uuid::new_v4();
        let only_b_id = Uuid::new_v4();

        let mut a = entity_file(vec![obs(shared_id, "shared"), obs(only_a_id, "only in ours")]);
        let b = entity_file(vec![obs(shared_id, "shared"), obs(only_b_id, "only in theirs")]);
        a.id = b.id;
        a.name = b.name.clone();

        let dir = std::env::temp_dir().join(format!("cuba-merge-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let a_path = dir.join("a.json");
        let b_path = dir.join("b.json");
        std::fs::write(&a_path, serde_json::to_vec(&a).unwrap()).unwrap();
        std::fs::write(&b_path, serde_json::to_vec(&b).unwrap()).unwrap();

        let merged_bytes = merge_entity_file(a_path.to_str().unwrap(), b_path.to_str().unwrap())
            .unwrap()
            .unwrap();
        let merged: EntityFile = serde_json::from_slice(&merged_bytes).unwrap();

        let ids: std::collections::HashSet<_> = merged.observations.iter().map(|o| o.id).collect();
        assert_eq!(ids.len(), 3, "shared + only_a + only_b, deduplicated by id");
        assert!(ids.contains(&shared_id));
        assert!(ids.contains(&only_a_id));
        assert!(ids.contains(&only_b_id));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_file_shape_returns_none_and_keeps_ours() {
        let dir = std::env::temp_dir().join(format!("cuba-merge-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let a_path = dir.join("manifest.json");
        std::fs::write(&a_path, b"{\"not\":\"an entity file\"}").unwrap();

        let result = merge_entity_file(a_path.to_str().unwrap(), a_path.to_str().unwrap()).unwrap();
        assert!(result.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }
}
