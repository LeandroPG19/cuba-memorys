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
    /// Embedding dimension of the exported vectors. Import uses it to parse the
    /// blob; without it the record size is unknown for any model other than the
    /// original 384-d one. `#[serde(default)]` keeps old manifests loadable
    /// (they fall back to 384 on import).
    #[serde(default)]
    pub embedding_dim: Option<usize>,
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

/// Stable SHA-256 of a payload. The manifest itself is excluded from the input,
/// so the same content always yields the same hash regardless of when it was
/// exported.
///
/// This used to call `DefaultHasher` while its own docstring claimed sha256 —
/// and that is not a cosmetic lie. `std`'s hasher is explicitly **not stable
/// across Rust releases** ("the internal algorithm is not specified, and so it
/// and its hashes should not be relied upon over releases"), yet this hash is
/// persisted in `brain_sync_state` and compared *across machines*. Two laptops
/// whose binaries were built with different Rust versions could hash identical
/// content differently, and the sync would see phantom changes. Its 64 bits were
/// also far too few to name content by its hash.
///
/// `sha2` was already a dependency — the audit log hashes with it — so the
/// comment about "avoiding" it was wrong on both counts.
///
/// Changing the algorithm changes every `manifest_hash`, so the first sync after
/// upgrading re-imports once. Import is an idempotent upsert, so that is safe.
pub fn payload_hash(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_is_sha256_and_deterministic() {
        // Known vector: sha256("") — pins the algorithm, so a future refactor
        // cannot quietly swap it for something unstable again.
        assert_eq!(
            payload_hash(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(payload_hash("brain"), payload_hash("brain"));
        assert_ne!(payload_hash("a"), payload_hash("b"));
        assert_eq!(payload_hash("x").len(), 64);
    }
}
