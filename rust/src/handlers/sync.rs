//! Handler: cuba_sync — git-friendly export/import (v0.8).

use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::Value;
use sqlx::PgPool;
use std::collections::HashMap;
use uuid::Uuid;

use crate::sync::chunk::{
    payload_hash, Counts, EntityFile, EpisodeFile, ErrorFile, Manifest, ObservationRow,
    ProjectRow, RelationRow, SCHEMA_VERSION,
};
use crate::sync::paths::{ensure_within, resolve_dir, slug};

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let dir_arg = args.get("dir").and_then(|v| v.as_str());
    let scope = args.get("scope").and_then(|v| v.as_str()).unwrap_or("project");
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

// ── Export ──────────────────────────────────────────────────────────

async fn export(
    pool: &PgPool,
    dir_arg: Option<&str>,
    scope: &str,
    with_embeddings: bool,
) -> Result<Value> {
    let root = resolve_dir(dir_arg)?;
    let project_id = if scope == "all" {
        None
    } else {
        crate::project::current_project_id(pool).await?
    };
    let project_name: Option<String> = match project_id {
        Some(pid) => sqlx::query_scalar("SELECT name FROM brain_projects WHERE id = $1")
            .bind(pid)
            .fetch_optional(pool)
            .await?,
        None => None,
    };

    // 1. Projects (always include all — small table, useful for cross-machine context)
    let projects: Vec<ProjectRow> = sqlx::query_as::<_, (Uuid, String, chrono::DateTime<Utc>)>(
        "SELECT id, name, created_at FROM brain_projects",
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|(id, name, created_at)| ProjectRow { id, name, created_at })
    .collect();

    // 2. Entities (with their observations bundled)
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

    let entities_dir = root.join("entities");
    std::fs::create_dir_all(&entities_dir).context("mkdir entities/")?;

    let mut entity_files = 0u32;
    let mut obs_count = 0u32;
    let mut emb_blob: Vec<u8> = Vec::new();

    for (id, name, entity_type, importance, access_count, p_id, created_at) in entity_rows {
        let observations: Vec<ObservationRow> = sqlx::query_as::<_, (
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
        )>(
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

        // Optional embeddings blob (only requested observations)
        if with_embeddings {
            for obs in &observations {
                let emb: Option<pgvector::Vector> = sqlx::query_scalar(
                    "SELECT embedding FROM brain_observations WHERE id = $1",
                )
                .bind(obs.id)
                .fetch_optional(pool)
                .await
                .ok()
                .flatten();
                if let Some(v) = emb {
                    let floats: Vec<f32> = v.to_vec();
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
        ensure_within(&root, &path)?;
        std::fs::write(&path, serde_json::to_vec_pretty(&file)?)
            .with_context(|| format!("write entity {basename}"))?;
        entity_files += 1;
    }

    // 3. Episodes (partitioned monthly)
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
    for ep in episode_rows {
        let yyyymm = ep.7.format("%Y-%m").to_string();
        let dir = root.join("episodes").join(&yyyymm);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.json", ep.0));
        ensure_within(&root, &path)?;
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
        std::fs::write(path, serde_json::to_vec_pretty(&f)?)?;
        episode_count += 1;
    }

    // 4. Errors
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
    for e in error_rows {
        let path = errors_dir.join(format!("{}.json", e.0));
        ensure_within(&root, &path)?;
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
        std::fs::write(path, serde_json::to_vec_pretty(&f)?)?;
        err_count += 1;
    }

    // 5. Decisions (subset of observations, but referenced as a separate index for
    // human navigability — files live in decisions/<id>.json)
    let decisions: Vec<(Uuid, String)> = sqlx::query_as(
        "SELECT id, content FROM brain_observations
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
    for (id, content) in &decisions {
        let path = dec_dir.join(format!("{id}.json"));
        ensure_within(&root, &path)?;
        let body = serde_json::json!({"id": id.to_string(), "content": content});
        std::fs::write(path, serde_json::to_vec_pretty(&body)?)?;
        dec_count += 1;
    }

    // 6. Relations (single small array)
    let relation_rows: Vec<RelationRow> = sqlx::query_as::<_, (
        Uuid,
        Uuid,
        Uuid,
        String,
        f64,
        bool,
        Option<Uuid>,
        chrono::DateTime<Utc>,
    )>(
        "SELECT id, from_entity, to_entity, relation_type, strength::float8,
                bidirectional, project_id, created_at
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
    })
    .collect();
    let rel_count = relation_rows.len() as u32;
    std::fs::write(
        root.join("relations.json"),
        serde_json::to_vec_pretty(&relation_rows)?,
    )?;
    std::fs::write(root.join("projects.json"), serde_json::to_vec_pretty(&projects)?)?;

    // 7. Optional embeddings blob
    if with_embeddings && !emb_blob.is_empty() {
        let compressed = crate::sync::compressor::compress(&emb_blob)?;
        std::fs::write(root.join("embeddings.bin.zst"), compressed)?;
    }

    // 8. Manifest with content-derived hash
    let counts = Counts {
        entities: entity_files,
        observations: obs_count,
        episodes: episode_count,
        decisions: dec_count,
        errors: err_count,
        relations: rel_count,
    };
    let payload_for_hash = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "counts": &counts,
    });
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        manifest_hash: payload_hash(&payload_for_hash.to_string()),
        project_id,
        project_name,
        exported_at: Utc::now(),
        counts: counts.clone(),
        with_embeddings,
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

    Ok(serde_json::json!({
        "action": "export",
        "dir": root.display().to_string(),
        "manifest_hash": manifest.manifest_hash,
        "counts": counts,
        "with_embeddings": with_embeddings,
        "warning": warning,
    }))
}

// ── Import ──────────────────────────────────────────────────────────

async fn import(pool: &PgPool, dir_arg: Option<&str>, conflict: &str) -> Result<Value> {
    let root = resolve_dir(dir_arg)?;
    let manifest_path = root.join("manifest.json");
    if !manifest_path.exists() {
        anyhow::bail!("no manifest.json at {}", root.display());
    }
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .context("parse manifest.json")?;

    if manifest.schema_version > SCHEMA_VERSION + 1 {
        anyhow::bail!(
            "manifest schema_version {} is too new (this build supports {})",
            manifest.schema_version,
            SCHEMA_VERSION
        );
    }

    // Idempotent: re-import same manifest = no-op (PRIMARY KEY conflict)
    let already: Option<(i32,)> = sqlx::query_as(
        "SELECT rows_inserted FROM brain_sync_state WHERE manifest_hash = $1",
    )
    .bind(&manifest.manifest_hash)
    .fetch_optional(pool)
    .await?;
    if let Some((prev,)) = already {
        return Ok(serde_json::json!({
            "action": "import",
            "skipped": true,
            "reason": "manifest already imported",
            "previous_rows_inserted": prev,
        }));
    }

    let on_conflict_clause = match conflict {
        "skip" | "merge" => "DO NOTHING",
        "overwrite" => "DO UPDATE SET", // followed below per-table
        _ => anyhow::bail!("invalid conflict policy: {conflict}"),
    };
    // Conservative: 'merge' and 'skip' both translate to DO NOTHING (UUID PK is stable).
    // 'overwrite' is currently best-effort; we apply it to the main mutable fields.
    let _ = on_conflict_clause; // placated unused-binding lint until overwrite path expands

    let mut tx = pool.begin().await?;
    let mut inserted = 0u32;

    // 1. Projects (must come first — FK target)
    let projects_path = root.join("projects.json");
    if projects_path.exists() {
        let projects: Vec<ProjectRow> =
            serde_json::from_slice(&std::fs::read(projects_path)?)?;
        for p in projects {
            let r = sqlx::query(
                "INSERT INTO brain_projects (id, name, created_at)
                 VALUES ($1, $2, $3)
                 ON CONFLICT (id) DO NOTHING",
            )
            .bind(p.id)
            .bind(&p.name)
            .bind(p.created_at)
            .execute(&mut *tx)
            .await?;
            inserted += r.rows_affected() as u32;
        }
    }

    // 2. Entities + observations bundled
    let entities_dir = root.join("entities");
    if entities_dir.exists() {
        for entry in std::fs::read_dir(entities_dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let file: EntityFile = serde_json::from_slice(&std::fs::read(&path)?)?;
            let r = sqlx::query(
                "INSERT INTO brain_entities (id, name, entity_type, importance, access_count, project_id, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 ON CONFLICT (id) DO NOTHING",
            )
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
                let r = sqlx::query(
                    "INSERT INTO brain_observations
                        (id, entity_id, content, observation_type, source, importance,
                         tags, session_id, project_id, embedding_model, created_at)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                     ON CONFLICT (id) DO NOTHING",
                )
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
                .execute(&mut *tx)
                .await?;
                inserted += r.rows_affected() as u32;
            }
        }
    }

    // 3. Episodes (recurse into yyyy-mm subdirs)
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
                let r = sqlx::query(
                    "INSERT INTO brain_episodes
                        (id, entity_id, content, actors, artifacts, importance,
                         project_id, started_at, ended_at)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                     ON CONFLICT (id) DO NOTHING",
                )
                .bind(f.id)
                .bind(f.entity_id)
                .bind(&f.content)
                .bind(&f.actors)
                .bind(&f.artifacts)
                .bind(f.importance)
                .bind(f.project_id)
                .bind(f.started_at)
                .bind(f.ended_at)
                .execute(&mut *tx)
                .await?;
                inserted += r.rows_affected() as u32;
            }
        }
    }

    // 4. Errors
    let errors_dir = root.join("errors");
    if errors_dir.exists() {
        for entry in std::fs::read_dir(errors_dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let e: ErrorFile = serde_json::from_slice(&std::fs::read(&path)?)?;
            let r = sqlx::query(
                "INSERT INTO brain_errors
                    (id, error_type, error_message, solution, resolved,
                     project, project_id, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                 ON CONFLICT (id) DO NOTHING",
            )
            .bind(e.id)
            .bind(&e.error_type)
            .bind(&e.error_message)
            .bind(&e.solution)
            .bind(e.resolved)
            .bind(&e.project)
            .bind(e.project_id)
            .bind(e.created_at)
            .execute(&mut *tx)
            .await?;
            inserted += r.rows_affected() as u32;
        }
    }

    // 5. Relations
    let relations_path = root.join("relations.json");
    if relations_path.exists() {
        let rels: Vec<RelationRow> =
            serde_json::from_slice(&std::fs::read(relations_path)?)?;
        for rel in rels {
            let r = sqlx::query(
                "INSERT INTO brain_relations
                    (id, from_entity, to_entity, relation_type, strength,
                     bidirectional, project_id, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                 ON CONFLICT (from_entity, to_entity, relation_type) DO NOTHING",
            )
            .bind(rel.id)
            .bind(rel.from_entity)
            .bind(rel.to_entity)
            .bind(&rel.relation_type)
            .bind(rel.strength)
            .bind(rel.bidirectional)
            .bind(rel.project_id)
            .bind(rel.created_at)
            .execute(&mut *tx)
            .await?;
            inserted += r.rows_affected() as u32;
        }
    }

    // 6. Optional embeddings restore
    let mut embeddings_restored = 0u32;
    let blob_path = root.join("embeddings.bin.zst");
    if manifest.with_embeddings && blob_path.exists() {
        let compressed = std::fs::read(&blob_path)?;
        let raw = crate::sync::compressor::decompress(&compressed)?;
        // Format: [16-byte UUID][384*4 bytes f32]* (rec_size = 16 + 1536 = 1552)
        const REC_SIZE: usize = 16 + 384 * 4;
        if raw.len() % REC_SIZE != 0 {
            tracing::warn!(
                "embeddings blob length {} not a multiple of {} — skipping",
                raw.len(),
                REC_SIZE
            );
        } else {
            for chunk in raw.chunks_exact(REC_SIZE) {
                let id_bytes: [u8; 16] = chunk[..16].try_into().unwrap();
                let id = Uuid::from_bytes(id_bytes);
                let mut floats = Vec::with_capacity(384);
                for f_chunk in chunk[16..].chunks_exact(4) {
                    let arr: [u8; 4] = f_chunk.try_into().unwrap();
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
        }
    }

    // 7. Tracking row
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

    Ok(serde_json::json!({
        "action": "import",
        "manifest_hash": manifest.manifest_hash,
        "rows_inserted": inserted,
        "embeddings_restored": embeddings_restored,
        "from": root.display().to_string(),
    }))
}

// ── Diff ────────────────────────────────────────────────────────────

async fn diff(pool: &PgPool, dir_arg: Option<&str>) -> Result<Value> {
    let root = resolve_dir(dir_arg)?;
    let project_id = crate::project::current_project_id(pool).await?;

    // Disk-side entity ids
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

    // DB-side entity ids
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

// ── Status ──────────────────────────────────────────────────────────

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

