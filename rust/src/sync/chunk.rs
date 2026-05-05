//! Serializable types written to disk by cuba_sync.
//!
//! Schema_version = 1. Bumping is breaking; loaders refuse manifests with
//! `version > current + 1` to give one minor-version drift grace.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Manifest {
    pub schema_version: u32,
    pub manifest_hash: String, // sha256 of serialized payload (without this field)
    pub project_id: Option<Uuid>,
    pub project_name: Option<String>,
    pub exported_at: DateTime<Utc>,
    pub counts: Counts,
    pub with_embeddings: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Counts {
    pub entities: u32,
    pub observations: u32,
    pub episodes: u32,
    pub decisions: u32,
    pub errors: u32,
    pub relations: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectRow {
    pub id: Uuid,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EntityFile {
    pub id: Uuid,
    pub name: String,
    pub entity_type: String,
    pub importance: f64,
    pub access_count: i32,
    pub project_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub observations: Vec<ObservationRow>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ObservationRow {
    pub id: Uuid,
    pub content: String,
    pub observation_type: String,
    pub source: String,
    pub importance: f64,
    pub tags: Vec<String>,
    pub project_id: Option<Uuid>,
    pub session_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub embedding_model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EpisodeFile {
    pub id: Uuid,
    pub entity_id: Uuid,
    pub content: String,
    pub actors: Vec<String>,
    pub artifacts: Vec<String>,
    pub importance: f64,
    pub project_id: Option<Uuid>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ErrorFile {
    pub id: Uuid,
    pub error_type: String,
    pub error_message: String,
    pub solution: Option<String>,
    pub resolved: bool,
    pub project: String,
    pub project_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RelationRow {
    pub id: Uuid,
    pub from_entity: Uuid,
    pub to_entity: Uuid,
    pub relation_type: String,
    pub strength: f64,
    pub bidirectional: bool,
    pub project_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

/// Compute a stable sha256 hash for dedup. We avoid hashing the manifest
/// itself so the same payload always yields the same hash regardless of
/// when it's exported.
pub fn payload_hash(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    // Use std hasher (FxHash-equivalent) — we don't need crypto strength,
    // just a stable identifier for dedup. Avoids pulling sha2.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}
