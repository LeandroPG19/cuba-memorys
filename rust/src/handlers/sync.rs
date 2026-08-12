use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::Value;
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::sync::chunk::{
    Counts, EntityFile, EpisodeFile, ErrorFile, MAX_EMBEDDING_DIM, Manifest, ObservationRow,
    ProjectRow, RelationRow, SCHEMA_VERSION, payload_hash, payload_hash_bytes,
};
use crate::sync::paths::{ensure_within, resolve_dir, slug};

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let dir_arg = args.get("dir").and_then(|v| v.as_str());
    let scope = args
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or("project");
    let with_embeddings = args
        .get("with_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let conflict = args
        .get("conflict")
        .and_then(|v| v.as_str())
        .unwrap_or("merge");

    match action {
        "export" => export(pool, dir_arg, scope, with_embeddings).await,
        "import" => import(pool, dir_arg, conflict).await,
        "diff" => diff(pool, dir_arg).await,
        "status" => status(pool, dir_arg).await,
        _ => anyhow::bail!("Invalid action: {action}. Use export/import/diff/status"),
    }
}

pub const SYNC_LOCK: i64 = 0x0CBA_A0D1_7106_0002;

#[derive(Clone, Copy)]
enum PruneScope {
    Everything,
    Project(Uuid),
}

impl PruneScope {
    fn may_delete(self, path: &Path) -> bool {
        match self {
            PruneScope::Everything => true,
            PruneScope::Project(exported) => declared_project_id(path) == Some(exported),
        }
    }
}

fn declared_project_id(path: &Path) -> Option<Uuid> {
    let bytes = std::fs::read(path).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("project_id")?
        .as_str()
        .and_then(|s| Uuid::parse_str(s).ok())
}

fn prune_stale_files(dir: &Path, keep: &HashSet<PathBuf>, scope: PruneScope) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        if !keep.contains(&path) && scope.may_delete(&path) {
            std::fs::remove_file(&path).with_context(|| format!("prune stale file {path:?}"))?;
        }
    }
    Ok(())
}

fn prune_stale_episode_files(dir: &Path, keep: &HashSet<PathBuf>, scope: PruneScope) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for month_entry in std::fs::read_dir(dir)? {
        let month_path = month_entry?.path();
        if !month_path.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&month_path)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            if !keep.contains(&path) && scope.may_delete(&path) {
                std::fs::remove_file(&path)
                    .with_context(|| format!("prune stale episode file {path:?}"))?;
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct BundleDigest {
    entries: Vec<(String, String)>,
}

impl BundleDigest {
    fn record(&mut self, root: &Path, path: &Path, bytes: &[u8]) {
        let relative = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        self.entries.push((relative, payload_hash_bytes(bytes)));
    }

    fn finish(mut self, project_id: Option<Uuid>) -> String {
        self.entries.sort();
        let mut acc = format!(
            "{SCHEMA_VERSION}\n{}\n",
            project_id.map(|p| p.to_string()).unwrap_or_default()
        );
        for (relative, hash) in &self.entries {
            acc.push_str(relative);
            acc.push(' ');
            acc.push_str(hash);
            acc.push('\n');
        }
        payload_hash(&acc)
    }
}

fn write_bundle_file(
    root: &Path,
    path: &Path,
    bytes: &[u8],
    digest: &mut BundleDigest,
) -> Result<()> {
    ensure_within(root, path)?;
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes).with_context(|| format!("write bundle file {tmp:?}"))?;
    std::fs::rename(&tmp, path).with_context(|| format!("publish bundle file {path:?}"))?;
    digest.record(root, path, bytes);
    Ok(())
}

fn embedding_record_size(dim: usize) -> Option<usize> {
    if dim == 0 || dim > MAX_EMBEDDING_DIM {
        return None;
    }
    dim.checked_mul(4)?.checked_add(16)
}

fn trust_for_imported(content: &str) -> (&'static str, Option<&'static str>) {
    match crate::redact::looks_like_secret(content) {
        Some(pattern) => (crate::core::trust::QUARANTINED, Some(pattern)),
        None => (crate::core::trust::TRUSTED, None),
    }
}

async fn export(
    pool: &PgPool,
    dir_arg: Option<&str>,
    scope: &str,
    with_embeddings: bool,
) -> Result<Value> {
    let root = resolve_dir(dir_arg)?;
    let mut lock = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(SYNC_LOCK)
        .execute(&mut *lock)
        .await
        .context("taking the sync lock")?;

    let project_id = if scope == "all" {
        None
    } else {
        crate::project::current_project_id(pool).await?
    };
    let project_name: Option<String> = match project_id {
        Some(pid) => {
            sqlx::query_scalar("SELECT name FROM brain_projects WHERE id = $1")
                .bind(pid)
                .fetch_optional(pool)
                .await?
        }
        None => None,
    };

    let projects: Vec<ProjectRow> = sqlx::query_as::<_, (Uuid, String, chrono::DateTime<Utc>)>(
        "SELECT id, name, created_at FROM brain_projects",
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|(id, name, created_at)| ProjectRow {
        id,
        name,
        created_at,
    })
    .collect();

    type EntityCols = (
        Uuid,
        String,
        String,
        f64,
        i32,
        Option<Uuid>,
        chrono::DateTime<Utc>,
    );
    let entity_rows: Vec<EntityCols> = sqlx::query_as(
        "SELECT id, name, entity_type, importance::float8, access_count, project_id, created_at
         FROM brain_entities
         WHERE ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY name",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    let prune_scope = match project_id {
        Some(pid) => PruneScope::Project(pid),
        None => PruneScope::Everything,
    };
    let mut digest = BundleDigest::default();

    let entities_dir = root.join("entities");
    std::fs::create_dir_all(&entities_dir).context("mkdir entities/")?;

    let mut entity_files = 0u32;
    let mut obs_count = 0u32;
    let mut emb_blob: Vec<u8> = Vec::new();
    let mut emb_dim: Option<usize> = None;
    let mut entity_paths: HashSet<PathBuf> = HashSet::new();

    for (id, name, entity_type, importance, access_count, p_id, created_at) in entity_rows {
        let observations: Vec<ObservationRow> = sqlx::query_as::<
            _,
            (
                Uuid,
                String,
                String,
                String,
                f64,
                Vec<String>,
                Option<Uuid>,
                Option<Uuid>,
                chrono::DateTime<Utc>,
                Option<String>,
            ),
        >(
            "SELECT id, content, observation_type, source, importance::float8, tags,
                    project_id, session_id, created_at, embedding_model
             FROM brain_observations
             WHERE entity_id = $1 AND observation_type != 'superseded'
             ORDER BY created_at",
        )
        .bind(id)
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(|t| ObservationRow {
            id: t.0,
            content: t.1,
            observation_type: t.2,
            source: t.3,
            importance: t.4,
            tags: t.5,
            project_id: t.6,
            session_id: t.7,
            created_at: t.8,
            embedding_model: t.9,
        })
        .collect();
        obs_count += observations.len() as u32;

        if with_embeddings {
            for obs in &observations {
                let emb: Option<pgvector::Vector> =
                    sqlx::query_scalar("SELECT embedding FROM brain_observations WHERE id = $1")
                        .bind(obs.id)
                        .fetch_optional(pool)
                        .await
                        .ok()
                        .flatten();
                if let Some(v) = emb {
                    let floats: Vec<f32> = v.to_vec();
                    if emb_dim.is_none() {
                        emb_dim = Some(floats.len());
                    }
                    emb_blob.extend_from_slice(obs.id.as_bytes());
                    for f in floats {
                        emb_blob.extend_from_slice(&f.to_le_bytes());
                    }
                }
            }
        }

        let file = EntityFile {
            id,
            name: name.clone(),
            entity_type,
            importance,
            access_count,
            project_id: p_id,
            created_at,
            observations,
        };
        let basename = format!("{}-{}.json", slug(&name), &id.to_string()[..8]);
        let path = entities_dir.join(&basename);
        write_bundle_file(
            &root,
            &path,
            &serde_json::to_vec_pretty(&file)?,
            &mut digest,
        )?;
        entity_paths.insert(path);
        entity_files += 1;
    }
    prune_stale_files(&entities_dir, &entity_paths, prune_scope)?;

    type EpCols = (
        Uuid,
        Uuid,
        String,
        Vec<String>,
        Vec<String>,
        f64,
        Option<Uuid>,
        chrono::DateTime<Utc>,
        Option<chrono::DateTime<Utc>>,
    );
    let episode_rows: Vec<EpCols> = sqlx::query_as(
        "SELECT id, entity_id, content, actors, artifacts, importance::float8,
                project_id, started_at, ended_at
         FROM brain_episodes
         WHERE ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY started_at",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    let mut episode_count = 0u32;
    let mut episode_paths: HashSet<PathBuf> = HashSet::new();
    for ep in episode_rows {
        let yyyymm = ep.7.format("%Y-%m").to_string();
        let dir = root.join("episodes").join(&yyyymm);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.json", ep.0));
        let f = EpisodeFile {
            id: ep.0,
            entity_id: ep.1,
            content: ep.2,
            actors: ep.3,
            artifacts: ep.4,
            importance: ep.5,
            project_id: ep.6,
            started_at: ep.7,
            ended_at: ep.8,
        };
        write_bundle_file(&root, &path, &serde_json::to_vec_pretty(&f)?, &mut digest)?;
        episode_paths.insert(path);
        episode_count += 1;
    }
    prune_stale_episode_files(&root.join("episodes"), &episode_paths, prune_scope)?;

    type ErrCols = (
        Uuid,
        String,
        String,
        Option<String>,
        bool,
        String,
        Option<Uuid>,
        chrono::DateTime<Utc>,
    );
    let error_rows: Vec<ErrCols> = sqlx::query_as(
        "SELECT id, error_type, error_message, solution, resolved, project, project_id, created_at
         FROM brain_errors
         WHERE ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY created_at",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    let errors_dir = root.join("errors");
    std::fs::create_dir_all(&errors_dir)?;
    let mut err_count = 0u32;
    let mut error_paths: HashSet<PathBuf> = HashSet::new();
    for e in error_rows {
        let path = errors_dir.join(format!("{}.json", e.0));
        let f = ErrorFile {
            id: e.0,
            error_type: e.1,
            error_message: e.2,
            solution: e.3,
            resolved: e.4,
            project: e.5,
            project_id: e.6,
            created_at: e.7,
        };
        write_bundle_file(&root, &path, &serde_json::to_vec_pretty(&f)?, &mut digest)?;
        error_paths.insert(path);
        err_count += 1;
    }
    prune_stale_files(&errors_dir, &error_paths, prune_scope)?;

    let decisions: Vec<(Uuid, String, Option<Uuid>)> = sqlx::query_as(
        "SELECT id, content, project_id FROM brain_observations
         WHERE observation_type = 'decision'
           AND ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY created_at",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    let dec_dir = root.join("decisions");
    std::fs::create_dir_all(&dec_dir)?;
    let mut dec_count = 0u32;
    let mut decision_paths: HashSet<PathBuf> = HashSet::new();
    for (id, content, owner) in &decisions {
        let path = dec_dir.join(format!("{id}.json"));
        let body = serde_json::json!({
            "id": id.to_string(),
            "content": content,
            "project_id": owner.map(|p| p.to_string()),
        });
        write_bundle_file(
            &root,
            &path,
            &serde_json::to_vec_pretty(&body)?,
            &mut digest,
        )?;
        decision_paths.insert(path);
        dec_count += 1;
    }
    prune_stale_files(&dec_dir, &decision_paths, prune_scope)?;

    let relation_rows: Vec<RelationRow> = sqlx::query_as::<
        _,
        (
            Uuid,
            Uuid,
            Uuid,
            String,
            f64,
            bool,
            Option<Uuid>,
            chrono::DateTime<Utc>,
            String,
        ),
    >(
        "SELECT id, from_entity, to_entity, relation_type, strength::float8,
                bidirectional, project_id, created_at, provenance
         FROM brain_relations
         WHERE ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)
         ORDER BY created_at",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|t| RelationRow {
        id: t.0,
        from_entity: t.1,
        to_entity: t.2,
        relation_type: t.3,
        strength: t.4,
        bidirectional: t.5,
        project_id: t.6,
        created_at: t.7,
        provenance: t.8,
    })
    .collect();
    let rel_count = relation_rows.len() as u32;
    write_bundle_file(
        &root,
        &root.join("relations.json"),
        &serde_json::to_vec_pretty(&relation_rows)?,
        &mut digest,
    )?;
    write_bundle_file(
        &root,
        &root.join("projects.json"),
        &serde_json::to_vec_pretty(&projects)?,
        &mut digest,
    )?;

    if with_embeddings && !emb_blob.is_empty() {
        let compressed = crate::sync::compressor::compress(&emb_blob)?;
        let blob_path = root.join("embeddings.bin.zst");
        ensure_within(&root, &blob_path)?;
        std::fs::write(&blob_path, compressed)?;
        digest.record(&root, &blob_path, &emb_blob);
    }

    let counts = Counts {
        entities: entity_files,
        observations: obs_count,
        episodes: episode_count,
        decisions: dec_count,
        errors: err_count,
        relations: rel_count,
    };
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        manifest_hash: digest.finish(project_id),
        project_id,
        project_name,
        exported_at: Utc::now(),
        counts: counts.clone(),
        with_embeddings,
        embedding_dim: emb_dim,
    };
    std::fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    let warning = if entity_files > 5000 {
        Some(format!(
            "exported {entity_files} entity files; consider partitioning further"
        ))
    } else {
        None
    };

    lock.commit().await?;

    Ok(serde_json::json!({
        "action": "export",
        "dir": root.display().to_string(),
        "manifest_hash": manifest.manifest_hash,
        "counts": counts,
        "with_embeddings": with_embeddings,
        "warning": warning,
    }))
}

async fn import(pool: &PgPool, dir_arg: Option<&str>, conflict: &str) -> Result<Value> {
    let root = resolve_dir(dir_arg)?;
    let manifest_path = root.join("manifest.json");
    if !manifest_path.exists() {
        anyhow::bail!("no manifest.json at {}", root.display());
    }
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest: Manifest =
        serde_json::from_slice(&manifest_bytes).context("parse manifest.json")?;

    if manifest.schema_version > SCHEMA_VERSION + 1 {
        anyhow::bail!(
            "manifest schema_version {} is too new (this build supports {})",
            manifest.schema_version,
            SCHEMA_VERSION
        );
    }

    let overwrite = match conflict {
        "skip" | "merge" => false,
        "overwrite" => true,
        _ => anyhow::bail!("invalid conflict policy: {conflict}"),
    };

    let mut tx = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(SYNC_LOCK)
        .execute(&mut *tx)
        .await
        .context("taking the sync lock")?;

    let already: Option<(i32,)> =
        sqlx::query_as("SELECT rows_inserted FROM brain_sync_state WHERE manifest_hash = $1")
            .bind(&manifest.manifest_hash)
            .fetch_optional(&mut *tx)
            .await?;
    if let Some((prev,)) = already {
        return Ok(serde_json::json!({
            "action": "import",
            "skipped": true,
            "reason": "manifest already imported",
            "previous_rows_inserted": prev,
        }));
    }

    let mut inserted = 0u32;
    let mut diverged: Vec<Uuid> = Vec::new();
    let mut quarantined = 0u32;
    let mut quarantine_reasons: HashMap<&'static str, u32> = HashMap::new();

    let projects_path = root.join("projects.json");
    if projects_path.exists() {
        let projects: Vec<ProjectRow> = serde_json::from_slice(&std::fs::read(projects_path)?)?;
        for p in projects {
            let r = sqlx::query(&format!(
                "INSERT INTO brain_projects (id, name, created_at)
                 VALUES ($1, $2, $3)
                 ON CONFLICT (id) DO {}",
                if overwrite {
                    "UPDATE SET name = EXCLUDED.name, created_at = EXCLUDED.created_at"
                } else {
                    "NOTHING"
                }
            ))
            .bind(p.id)
            .bind(&p.name)
            .bind(p.created_at)
            .execute(&mut *tx)
            .await?;
            inserted += r.rows_affected() as u32;
        }
    }

    let entities_dir = root.join("entities");
    if entities_dir.exists() {
        for entry in std::fs::read_dir(entities_dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let file: EntityFile = serde_json::from_slice(&std::fs::read(&path)?)?;
            let r = sqlx::query(&format!(
                "INSERT INTO brain_entities (id, name, entity_type, importance, access_count, project_id, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 ON CONFLICT (id) DO {}",
                if overwrite {
                    "UPDATE SET name = EXCLUDED.name, entity_type = EXCLUDED.entity_type, \
                     importance = EXCLUDED.importance, access_count = EXCLUDED.access_count, \
                     project_id = EXCLUDED.project_id, created_at = EXCLUDED.created_at"
                } else {
                    "NOTHING"
                }
            ))
            .bind(file.id)
            .bind(&file.name)
            .bind(&file.entity_type)
            .bind(file.importance)
            .bind(file.access_count)
            .bind(file.project_id)
            .bind(file.created_at)
            .execute(&mut *tx)
            .await?;
            inserted += r.rows_affected() as u32;

            for obs in &file.observations {
                let (trust, reason) = trust_for_imported(&obs.content);
                let r = sqlx::query(&format!(
                    "INSERT INTO brain_observations
                        (id, entity_id, content, observation_type, source, importance,
                         tags, session_id, project_id, embedding_model, created_at, trust)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                     ON CONFLICT (id) DO {}",
                    if overwrite {
                        "UPDATE SET entity_id = EXCLUDED.entity_id, content = EXCLUDED.content, \
                         observation_type = EXCLUDED.observation_type, source = EXCLUDED.source, \
                         importance = EXCLUDED.importance, tags = EXCLUDED.tags, \
                         session_id = EXCLUDED.session_id, project_id = EXCLUDED.project_id, \
                         embedding_model = EXCLUDED.embedding_model, \
                         created_at = EXCLUDED.created_at, trust = EXCLUDED.trust"
                    } else {
                        "NOTHING"
                    }
                ))
                .bind(obs.id)
                .bind(file.id)
                .bind(&obs.content)
                .bind(&obs.observation_type)
                .bind(&obs.source)
                .bind(obs.importance)
                .bind(&obs.tags)
                .bind(obs.session_id)
                .bind(obs.project_id)
                .bind(&obs.embedding_model)
                .bind(obs.created_at)
                .bind(trust)
                .execute(&mut *tx)
                .await?;
                inserted += r.rows_affected() as u32;
                if let Some(pattern) = reason.filter(|_| r.rows_affected() > 0) {
                    quarantined += 1;
                    *quarantine_reasons.entry(pattern).or_insert(0) += 1;
                }
                if !overwrite && r.rows_affected() == 0 {
                    let same: Option<bool> = sqlx::query_scalar(
                        "SELECT content = $2 FROM brain_observations WHERE id = $1",
                    )
                    .bind(obs.id)
                    .bind(&obs.content)
                    .fetch_optional(&mut *tx)
                    .await?
                    .flatten();
                    if same == Some(false) {
                        diverged.push(obs.id);
                    }
                }
            }
        }
    }

    let episodes_root = root.join("episodes");
    if episodes_root.exists() {
        for month_entry in std::fs::read_dir(episodes_root)? {
            let month = month_entry?.path();
            if !month.is_dir() {
                continue;
            }
            for ep_entry in std::fs::read_dir(month)? {
                let path = ep_entry?.path();
                if path.extension().is_none_or(|e| e != "json") {
                    continue;
                }
                let f: EpisodeFile = serde_json::from_slice(&std::fs::read(&path)?)?;
                let (trust, reason) = trust_for_imported(&f.content);
                if let Some(pattern) = reason {
                    quarantined += 1;
                    *quarantine_reasons.entry(pattern).or_insert(0) += 1;
                }
                let r = sqlx::query(&format!(
                    "INSERT INTO brain_episodes
                        (id, entity_id, content, actors, artifacts, importance,
                         project_id, started_at, ended_at, trust)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                     ON CONFLICT (id) DO {}",
                    if overwrite {
                        "UPDATE SET entity_id = EXCLUDED.entity_id, content = EXCLUDED.content, \
                         actors = EXCLUDED.actors, artifacts = EXCLUDED.artifacts, \
                         importance = EXCLUDED.importance, project_id = EXCLUDED.project_id, \
                         started_at = EXCLUDED.started_at, ended_at = EXCLUDED.ended_at, \
                         trust = EXCLUDED.trust"
                    } else {
                        "NOTHING"
                    }
                ))
                .bind(f.id)
                .bind(f.entity_id)
                .bind(&f.content)
                .bind(&f.actors)
                .bind(&f.artifacts)
                .bind(f.importance)
                .bind(f.project_id)
                .bind(f.started_at)
                .bind(f.ended_at)
                .bind(trust)
                .execute(&mut *tx)
                .await?;
                inserted += r.rows_affected() as u32;
            }
        }
    }

    let errors_dir = root.join("errors");
    if errors_dir.exists() {
        for entry in std::fs::read_dir(errors_dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let e: ErrorFile = serde_json::from_slice(&std::fs::read(&path)?)?;
            let searchable = format!(
                "{}\n{}",
                e.error_message,
                e.solution.as_deref().unwrap_or("")
            );
            let (trust, reason) = trust_for_imported(&searchable);
            if let Some(pattern) = reason {
                quarantined += 1;
                *quarantine_reasons.entry(pattern).or_insert(0) += 1;
            }
            let r = sqlx::query(&format!(
                "INSERT INTO brain_errors
                    (id, error_type, error_message, solution, resolved,
                     project, project_id, created_at, trust)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                 ON CONFLICT (id) DO {}",
                if overwrite {
                    "UPDATE SET error_type = EXCLUDED.error_type, \
                     error_message = EXCLUDED.error_message, solution = EXCLUDED.solution, \
                     resolved = EXCLUDED.resolved, project = EXCLUDED.project, \
                     project_id = EXCLUDED.project_id, created_at = EXCLUDED.created_at, \
                     trust = EXCLUDED.trust"
                } else {
                    "NOTHING"
                }
            ))
            .bind(e.id)
            .bind(&e.error_type)
            .bind(&e.error_message)
            .bind(&e.solution)
            .bind(e.resolved)
            .bind(&e.project)
            .bind(e.project_id)
            .bind(e.created_at)
            .bind(trust)
            .execute(&mut *tx)
            .await?;
            inserted += r.rows_affected() as u32;
        }
    }

    let relations_path = root.join("relations.json");
    if relations_path.exists() {
        let rels: Vec<RelationRow> = serde_json::from_slice(&std::fs::read(relations_path)?)?;
        for rel in rels {
            let r = sqlx::query(&format!(
                "INSERT INTO brain_relations
                    (id, from_entity, to_entity, relation_type, strength,
                     bidirectional, project_id, created_at, provenance)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                 ON CONFLICT (from_entity, to_entity, relation_type) DO {}",
                if overwrite {
                    "UPDATE SET strength = EXCLUDED.strength, \
                     bidirectional = EXCLUDED.bidirectional, project_id = EXCLUDED.project_id, \
                     created_at = EXCLUDED.created_at, provenance = EXCLUDED.provenance"
                } else {
                    "NOTHING"
                }
            ))
            .bind(rel.id)
            .bind(rel.from_entity)
            .bind(rel.to_entity)
            .bind(&rel.relation_type)
            .bind(rel.strength)
            .bind(rel.bidirectional)
            .bind(rel.project_id)
            .bind(rel.created_at)
            .bind(&rel.provenance)
            .execute(&mut *tx)
            .await?;
            inserted += r.rows_affected() as u32;
        }
    }

    let mut embeddings_restored = 0u32;
    let blob_path = root.join("embeddings.bin.zst");
    if manifest.with_embeddings && blob_path.exists() {
        let compressed = std::fs::read(&blob_path)?;
        let raw = crate::sync::compressor::decompress(&compressed)?;
        let dim = manifest
            .embedding_dim
            .unwrap_or(crate::embeddings::onnx::EMBEDDING_DIM);
        let rec_size = embedding_record_size(dim).filter(|size| raw.len() % size == 0);
        if let Some(rec_size) = rec_size {
            for chunk in raw.chunks_exact(rec_size) {
                let id_bytes: [u8; 16] = chunk[..16]
                    .try_into()
                    .expect("embedding_record_size rejects any dim whose record is not longer than the 16-byte uuid");
                let id = Uuid::from_bytes(id_bytes);
                let mut floats = Vec::with_capacity(dim);
                for f_chunk in chunk[16..].chunks_exact(4) {
                    let arr: [u8; 4] = f_chunk
                        .try_into()
                        .expect("chunks_exact(4) yields exactly 4 bytes");
                    floats.push(f32::from_le_bytes(arr));
                }
                let v = pgvector::Vector::from(floats);
                let r = sqlx::query(
                    "UPDATE brain_observations SET embedding = $1::vector WHERE id = $2",
                )
                .bind(v)
                .bind(id)
                .execute(&mut *tx)
                .await?;
                embeddings_restored += r.rows_affected() as u32;
            }
        } else {
            tracing::warn!(
                "embeddings blob of {} bytes declares dim {} — skipping",
                raw.len(),
                dim
            );
        }
    }

    sqlx::query(
        "INSERT INTO brain_sync_state (manifest_hash, project_id, rows_inserted, source_path)
         VALUES ($1, $2, $3, $4) ON CONFLICT (manifest_hash) DO NOTHING",
    )
    .bind(&manifest.manifest_hash)
    .bind(manifest.project_id)
    .bind(inserted as i32)
    .bind(root.display().to_string())
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    let quarantine_note = (quarantined > 0).then(|| {
        format!(
            "{quarantined} imported rows look like they carry a credential and were written \
             with trust=quarantined: observations and episodes are withheld from cuba_faro and \
             errors from cuba_expediente. Read them with cuba_eco action=pending and accept \
             them with cuba_eco action=promote"
        )
    });

    let divergence_note = (!diverged.is_empty()).then(|| {
        format!(
            "{} observations already existed here with different content and were LEFT \
             UNCHANGED. conflict=merge is not a merge: it inserts what is missing and keeps \
             whatever was here first, so a correction made on the other machine after this \
             row was created does not arrive. Until per-row causality lands, the choices are \
             conflict=overwrite — which takes the incoming version and discards this one — or \
             reconciling these by hand. Ids: {}",
            diverged.len(),
            diverged
                .iter()
                .take(20)
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    });

    Ok(serde_json::json!({
        "action": "import",
        "manifest_hash": manifest.manifest_hash,
        "rows_inserted": inserted,
        "diverged": diverged.len(),
        "divergence_note": divergence_note,
        "embeddings_restored": embeddings_restored,
        "quarantined": quarantined,
        "quarantine_reasons": quarantine_reasons,
        "quarantine_note": quarantine_note,
        "from": root.display().to_string(),
    }))
}

async fn diff(pool: &PgPool, dir_arg: Option<&str>) -> Result<Value> {
    let root = resolve_dir(dir_arg)?;
    let project_id = crate::project::current_project_id(pool).await?;

    let mut on_disk: HashMap<Uuid, String> = HashMap::new();
    let entities_dir = root.join("entities");
    if entities_dir.exists() {
        for entry in std::fs::read_dir(&entities_dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let f: EntityFile = serde_json::from_slice(&std::fs::read(&path)?)?;
            on_disk.insert(f.id, f.name);
        }
    }

    let db_rows: Vec<(Uuid, String)> = sqlx::query_as(
        "SELECT id, name FROM brain_entities
         WHERE ($1::uuid IS NULL OR project_id = $1 OR project_id IS NULL)",
    )
    .bind(project_id)
    .fetch_all(pool)
    .await?;
    let in_db: HashMap<Uuid, String> = db_rows.into_iter().collect();

    let only_disk: Vec<Value> = on_disk
        .iter()
        .filter(|(id, _)| !in_db.contains_key(id))
        .map(|(id, name)| serde_json::json!({"id": id.to_string(), "name": name}))
        .collect();
    let only_db: Vec<Value> = in_db
        .iter()
        .filter(|(id, _)| !on_disk.contains_key(id))
        .map(|(id, name)| serde_json::json!({"id": id.to_string(), "name": name}))
        .collect();

    Ok(serde_json::json!({
        "action": "diff",
        "dir": root.display().to_string(),
        "only_in_disk": only_disk,
        "only_in_db": only_db,
        "common_count": on_disk.len() - only_disk.len(),
    }))
}

async fn status(pool: &PgPool, dir_arg: Option<&str>) -> Result<Value> {
    let root = resolve_dir(dir_arg)?;
    let manifest_path = root.join("manifest.json");

    let mut on_disk: Option<Manifest> = None;
    if manifest_path.exists() {
        on_disk = serde_json::from_slice(&std::fs::read(&manifest_path)?).ok();
    }

    let imported: Vec<(String, Option<Uuid>, chrono::DateTime<Utc>, i32)> = sqlx::query_as(
        "SELECT manifest_hash, project_id, imported_at, rows_inserted
         FROM brain_sync_state ORDER BY imported_at DESC LIMIT 20",
    )
    .fetch_all(pool)
    .await?;

    let imported_json: Vec<Value> = imported
        .iter()
        .map(|(h, p, ts, n)| {
            serde_json::json!({
                "manifest_hash": h,
                "project_id": p.map(|p| p.to_string()),
                "imported_at": ts.to_rfc3339(),
                "rows_inserted": n,
            })
        })
        .collect();

    let pending = match &on_disk {
        Some(m) => {
            let already: Option<(i32,)> = sqlx::query_as(
                "SELECT rows_inserted FROM brain_sync_state WHERE manifest_hash = $1",
            )
            .bind(&m.manifest_hash)
            .fetch_optional(pool)
            .await?;
            already.is_none()
        }
        None => false,
    };

    Ok(serde_json::json!({
        "action": "status",
        "dir": root.display().to_string(),
        "current_manifest": on_disk,
        "pending_import": pending,
        "recent_imports": imported_json,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cuba-sync-{label}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("the test needs a scratch directory");
        dir
    }

    fn write_json(dir: &Path, name: &str, body: Value) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&body).expect("a Value serialises"),
        )
        .expect("writing the fixture");
        path
    }

    #[test]
    fn a_project_scoped_export_never_prunes_a_file_owned_by_another_project() {
        let dir = scratch("prune-other-project");
        let mine = Uuid::new_v4();
        let theirs = Uuid::new_v4();

        let kept = write_json(
            &dir,
            "mine-kept.json",
            serde_json::json!({"project_id": mine}),
        );
        let stale = write_json(
            &dir,
            "mine-stale.json",
            serde_json::json!({"project_id": mine}),
        );
        let other = write_json(
            &dir,
            "theirs.json",
            serde_json::json!({"project_id": theirs}),
        );
        let global = write_json(&dir, "global.json", serde_json::json!({"project_id": null}));

        let keep: HashSet<PathBuf> = [kept.clone()].into_iter().collect();
        prune_stale_files(&dir, &keep, PruneScope::Project(mine)).expect("pruning must not fail");

        assert!(kept.exists(), "a file this export just wrote must survive");
        assert!(
            !stale.exists(),
            "a file of the exported project that the export no longer produced is genuinely \
             stale and pruning it is the whole point of the function"
        );
        assert!(
            other.exists(),
            "scope=project is the DEFAULT of cuba_sync export, and the keep set only ever holds \
             this project's files: deleting another project's export left the user with nothing \
             but whatever git happened to have committed"
        );
        assert!(
            global.exists(),
            "an entity with project_id NULL is shared by every project; a single project's \
             export has no standing to delete it"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_full_export_still_owns_and_prunes_the_whole_directory() {
        let dir = scratch("prune-everything");
        let kept = write_json(&dir, "kept.json", serde_json::json!({"project_id": null}));
        let stale = write_json(
            &dir,
            "stale.json",
            serde_json::json!({"project_id": Uuid::new_v4()}),
        );
        let opaque = dir.join("opaque.json");
        std::fs::write(&opaque, b"not json at all").expect("writing the fixture");

        let keep: HashSet<PathBuf> = [kept.clone()].into_iter().collect();
        prune_stale_files(&dir, &keep, PruneScope::Everything).expect("pruning must not fail");

        assert!(kept.exists());
        assert!(
            !stale.exists() && !opaque.exists(),
            "scope=all exports every row there is, so anything left over really was deleted from \
             the graph — narrowing this case would turn the export directory into an attic"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_that_does_not_declare_its_project_is_left_alone_by_a_project_export() {
        let dir = scratch("prune-unattributable");
        let legacy = write_json(&dir, "legacy.json", serde_json::json!({"id": "x"}));
        let corrupt = dir.join("corrupt.json");
        std::fs::write(&corrupt, b"{ truncated").expect("writing the fixture");

        prune_stale_files(&dir, &HashSet::new(), PruneScope::Project(Uuid::new_v4()))
            .expect("pruning must not fail");

        assert!(
            legacy.exists() && corrupt.exists(),
            "deletion needs positive evidence of ownership: a file written by an older version, \
             or one the user is mid-edit on, must not be destroyed because it failed to parse"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn episode_pruning_is_scoped_the_same_way_as_every_other_directory() {
        let dir = scratch("prune-episodes");
        let month = dir.join("2026-08");
        std::fs::create_dir_all(&month).expect("the test needs the month directory");
        let mine = Uuid::new_v4();
        let stale = write_json(&month, "mine.json", serde_json::json!({"project_id": mine}));
        let other = write_json(
            &month,
            "theirs.json",
            serde_json::json!({"project_id": Uuid::new_v4()}),
        );

        prune_stale_episode_files(&dir, &HashSet::new(), PruneScope::Project(mine))
            .expect("pruning must not fail");

        assert!(!stale.exists());
        assert!(
            other.exists(),
            "episodes live one directory deeper but the ownership rule is identical: a project \
             export deletes only what it produced"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_bundles_with_the_same_file_names_but_different_content_hash_differently() {
        let root = Path::new("/bundle");
        let project = Some(Uuid::new_v4());

        let mut before = BundleDigest::default();
        before.record(
            root,
            &root.join("entities/a.json"),
            b"{\"content\":\"original\"}",
        );
        before.record(root, &root.join("relations.json"), b"[]");

        let mut after = BundleDigest::default();
        after.record(
            root,
            &root.join("entities/a.json"),
            b"{\"content\":\"edited\"}",
        );
        after.record(root, &root.join("relations.json"), b"[]");

        assert_ne!(
            before.finish(project),
            after.finish(project),
            "the hash used to cover only the row COUNTS, so hand-editing a .json — the \
             git-friendly workflow the README sells — kept the same hash and the import on the \
             other machine answered skipped:true and discarded the edit in silence"
        );
    }

    #[test]
    fn the_bundle_hash_does_not_depend_on_the_order_the_files_were_written() {
        let root = Path::new("/bundle");
        let project = Some(Uuid::new_v4());

        let mut one = BundleDigest::default();
        one.record(root, &root.join("entities/a.json"), b"a");
        one.record(root, &root.join("entities/b.json"), b"b");

        let mut other = BundleDigest::default();
        other.record(root, &root.join("entities/b.json"), b"b");
        other.record(root, &root.join("entities/a.json"), b"a");

        assert_eq!(
            one.finish(project),
            other.finish(project),
            "row order comes out of Postgres and directory order comes out of the filesystem: if \
             either leaked into the hash, the same bundle would look new on every machine and \
             the already-imported check would never fire"
        );
    }

    #[test]
    fn the_bundle_hash_still_separates_a_project_export_from_a_full_one() {
        let root = Path::new("/bundle");
        let mut project_scoped = BundleDigest::default();
        project_scoped.record(root, &root.join("projects.json"), b"[]");
        let mut full = BundleDigest::default();
        full.record(root, &root.join("projects.json"), b"[]");

        assert_ne!(
            project_scoped.finish(Some(Uuid::new_v4())),
            full.finish(None),
            "two exports of identical files still describe different slices of the graph"
        );
    }

    #[test]
    fn a_hostile_embedding_dim_can_never_produce_a_record_shorter_than_the_uuid_it_starts_with() {
        const HOSTILE: usize = 4611686018427387903;

        assert_eq!(
            16usize.wrapping_add(HOSTILE.wrapping_mul(4)),
            12,
            "this is the arithmetic the import used to do on a manifest.json that arrives \
             through a shared git repo: a 12-byte record passed the `rec_size == 16` guard, and \
             then chunk[..16] panicked on a 12-byte slice"
        );
        assert_eq!(embedding_record_size(HOSTILE), None);
        assert_eq!(embedding_record_size(usize::MAX), None);
        assert_eq!(
            embedding_record_size(0),
            None,
            "dim 0 is the case the old `rec_size == 16` guard did catch, and it must stay caught"
        );
        assert_eq!(
            embedding_record_size(MAX_EMBEDDING_DIM + 1),
            None,
            "pgvector cannot store more than {MAX_EMBEDDING_DIM} dimensions, so a larger dim is \
             never a real export"
        );

        assert_eq!(embedding_record_size(384), Some(1552));
        assert_eq!(embedding_record_size(1024), Some(4112));
        for dim in [1, 2, 384, 1024, MAX_EMBEDDING_DIM] {
            let size = embedding_record_size(dim).expect("a real dimension is accepted");
            assert!(
                size > 16,
                "the expect() in the import reads chunk[..16] and states that records are longer \
                 than a uuid; dim {dim} produced {size}"
            );
        }
    }

    #[test]
    fn an_imported_observation_that_carries_a_credential_is_quarantined_instead_of_dropped() {
        let (trust, reason) = trust_for_imported("el deploy usa ghp_abcdefghijklmnop");
        assert_eq!(
            trust,
            crate::core::trust::QUARANTINED,
            "cuba_sync import reads JSON out of a repository anyone with push can write to, and \
             a git pull followed by an import puts it in the graph with nobody reading it"
        );
        assert_eq!(
            reason,
            Some("github token"),
            "the import has to report WHY a row was held back, or the user cannot judge whether \
             to promote it"
        );

        let (trust, reason) = trust_for_imported("el bug era que la password no se validaba");
        assert_eq!(
            trust,
            crate::core::trust::TRUSTED,
            "prose about credentials is not a credential; quarantining it would hide the ordinary \
             content of an import behind a review queue"
        );
        assert_eq!(reason, None);
    }
}
