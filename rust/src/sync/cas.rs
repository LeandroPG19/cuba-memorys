//! Content-addressed chunk store — sync across machines without merge conflicts.
//!
//! The existing exporter writes `entities/<slug>.json`: one mutable file per
//! entity, named after the entity. Edit the same entity on the laptop and on the
//! desktop, and git sees two versions of the same path. That is a merge conflict
//! in a JSON blob, resolved by hand, in a file nobody reads — and the loser's
//! observations are silently gone. With three machines it is not a question of
//! whether, only when.
//!
//! Content addressing removes the shared name. A chunk is stored at
//! `chunks/<hh>/<hash>.json`, where the hash is the SHA-256 of its bytes. Chunks
//! are therefore **immutable**: change an entity and you get a new chunk at a new
//! path; the old one is untouched. Two machines that edited the same entity
//! produce two chunks, both of which merge cleanly, because they never touch the
//! same file.
//!
//! What *can* still conflict is the index — the map from entity to current chunk.
//! That is the point: the conflict is now a single line naming two hashes, and
//! both versions of the content are still there to reconcile. A tiny, honest
//! conflict beats a silent overwrite.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::chunk::payload_hash;
use super::transport::Transport;

/// Index format version. A loader that does not understand the version must
/// refuse it rather than guess.
pub const INDEX_VERSION: u32 = 1;

/// Maps a logical name (entity slug) to the hash of its current chunk.
///
/// `BTreeMap` and not `HashMap`: the index is serialized into git, and a map
/// with nondeterministic iteration order would produce a different file on every
/// export, turning every sync into a diff.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Index {
    pub version: u32,
    /// entity slug → chunk hash
    pub entries: BTreeMap<String, String>,
}

impl Default for Index {
    fn default() -> Self {
        Self {
            version: INDEX_VERSION,
            entries: BTreeMap::new(),
        }
    }
}

impl Index {
    /// Merge another machine's index into this one.
    ///
    /// Where both name the same entity with different chunks, neither wins: the
    /// caller gets told. Resolving it needs the *content* of both chunks — which,
    /// crucially, both still exist on disk. Picking a side here, silently, is
    /// exactly the data loss this module exists to prevent.
    pub fn merge(&mut self, other: &Index) -> Vec<Divergence> {
        let mut divergences = Vec::new();
        for (name, their_hash) in &other.entries {
            match self.entries.get(name) {
                Some(our_hash) if our_hash != their_hash => {
                    divergences.push(Divergence {
                        name: name.clone(),
                        ours: our_hash.clone(),
                        theirs: their_hash.clone(),
                    });
                }
                Some(_) => {} // identical: nothing to do
                None => {
                    self.entries.insert(name.clone(), their_hash.clone());
                }
            }
        }
        divergences
    }
}

/// The same entity, pointing at two different chunks. Both chunks exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    pub name: String,
    pub ours: String,
    pub theirs: String,
}

/// Path of a chunk, sharded by the first two hex chars so no directory ends up
/// with tens of thousands of entries.
fn chunk_key(hash: &str) -> String {
    format!("chunks/{}/{}.json", &hash[..2], hash)
}

pub struct ChunkStore<'a> {
    transport: &'a dyn Transport,
}

impl<'a> ChunkStore<'a> {
    pub fn new(transport: &'a dyn Transport) -> Self {
        Self { transport }
    }

    /// Store a chunk, returning its hash. Writing the same content twice is a
    /// no-op, not a duplicate — that is what makes an export incremental for free.
    pub fn put(&self, payload: &str) -> Result<String> {
        let hash = payload_hash(payload);
        let key = chunk_key(&hash);
        if self.transport.exists(&key) {
            return Ok(hash); // already have these exact bytes
        }
        self.transport.put(&key, payload.as_bytes())?;
        Ok(hash)
    }

    /// Fetch a chunk **and verify it hashes to the name it was stored under**.
    ///
    /// The store may be a git checkout that someone rebased, or a directory
    /// synced by a tool that truncated a file. A content-addressed store that
    /// does not check the address is just a directory with confusing names.
    pub fn get(&self, hash: &str) -> Result<String> {
        let bytes = self.transport.get(&chunk_key(hash))?;
        let text = String::from_utf8(bytes).context("el chunk no es UTF-8 válido")?;
        let actual = payload_hash(&text);
        if actual != hash {
            bail!("chunk corrupto: se pidió {hash} pero el contenido hashea a {actual}");
        }
        Ok(text)
    }

    pub fn read_index(&self) -> Result<Index> {
        if !self.transport.exists("index.json") {
            return Ok(Index::default());
        }
        let bytes = self.transport.get("index.json")?;
        let index: Index = serde_json::from_slice(&bytes).context("index.json ilegible")?;
        if index.version > INDEX_VERSION {
            bail!(
                "index.json es de una versión más nueva ({}) que este binario ({INDEX_VERSION}) \
                 — actualizá cuba-memorys antes de importar",
                index.version
            );
        }
        Ok(index)
    }

    pub fn write_index(&self, index: &Index) -> Result<()> {
        let json = serde_json::to_vec_pretty(index)?;
        self.transport.put("index.json", &json)
    }

    /// Chunks in the store that no index entry points to. They are not deleted
    /// here: an unreferenced chunk is the only copy of a version someone may
    /// still want, and reclaiming space is not worth losing memory over.
    pub fn unreferenced(&self, index: &Index) -> Result<Vec<String>> {
        let referenced: std::collections::HashSet<&String> = index.entries.values().collect();
        let mut orphans = Vec::new();
        for key in self.transport.list("chunks")? {
            let Some(hash) = key.rsplit('/').next().and_then(|f| f.strip_suffix(".json")) else {
                continue;
            };
            if !referenced.contains(&hash.to_string()) {
                orphans.push(hash.to_string());
            }
        }
        Ok(orphans)
    }
}

#[cfg(test)]
mod tests {
    use super::super::transport::FsTransport;
    use super::*;

    fn store_dir(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("cuba-cas-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&p).ok();
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn identical_content_is_stored_once() {
        let dir = store_dir("dedup");
        let t = FsTransport::new(&dir);
        let store = ChunkStore::new(&t);

        let h1 = store.put(r#"{"name":"cuba"}"#).unwrap();
        let h2 = store.put(r#"{"name":"cuba"}"#).unwrap();
        assert_eq!(h1, h2, "el mismo contenido debe dar el mismo hash");
        assert_eq!(
            t.list("chunks").unwrap().len(),
            1,
            "no debe duplicar el blob"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn editing_an_entity_leaves_the_old_chunk_intact() {
        // The whole reason this module exists: no shared mutable path, so the
        // other machine's version cannot be clobbered.
        let dir = store_dir("immutable");
        let t = FsTransport::new(&dir);
        let store = ChunkStore::new(&t);

        let old = store.put(r#"{"v":1}"#).unwrap();
        let new = store.put(r#"{"v":2}"#).unwrap();
        assert_ne!(old, new);
        assert_eq!(
            store.get(&old).unwrap(),
            r#"{"v":1}"#,
            "la versión vieja sobrevive"
        );
        assert_eq!(store.get(&new).unwrap(), r#"{"v":2}"#);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_tampered_chunk_is_rejected_not_returned() {
        let dir = store_dir("corrupt");
        let t = FsTransport::new(&dir);
        let store = ChunkStore::new(&t);

        let hash = store.put(r#"{"real":true}"#).unwrap();
        // Simulate a bad rebase / truncated sync: right name, wrong bytes.
        t.put(&chunk_key(&hash), b"{\"real\":false}").unwrap();

        let err = store.get(&hash).unwrap_err().to_string();
        assert!(
            err.contains("corrupto"),
            "debería detectar la corrupción: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merging_two_machines_keeps_both_versions_and_reports_the_clash() {
        let mut laptop = Index::default();
        laptop.entries.insert("cuba-memorys".into(), "aaa".into());
        laptop.entries.insert("solo-laptop".into(), "bbb".into());

        let mut desktop = Index::default();
        desktop.entries.insert("cuba-memorys".into(), "zzz".into()); // edited on both
        desktop.entries.insert("solo-desktop".into(), "ccc".into());

        let divergences = laptop.merge(&desktop);

        // Non-conflicting entries merge silently — that is the common case.
        assert_eq!(laptop.entries.get("solo-desktop"), Some(&"ccc".to_string()));
        assert_eq!(laptop.entries.get("solo-laptop"), Some(&"bbb".to_string()));

        // The clash is REPORTED, not resolved: picking a winner here would
        // delete the other machine's work with no trace.
        assert_eq!(divergences.len(), 1);
        assert_eq!(divergences[0].name, "cuba-memorys");
        assert_eq!(divergences[0].ours, "aaa");
        assert_eq!(divergences[0].theirs, "zzz");
        assert_eq!(
            laptop.entries.get("cuba-memorys"),
            Some(&"aaa".to_string()),
            "no debe pisar nuestra versión a espaldas del usuario"
        );
    }

    #[test]
    fn the_index_serializes_deterministically() {
        // A HashMap here would reorder on every export and turn each sync into
        // a spurious diff.
        let mut idx = Index::default();
        for name in ["zeta", "alpha", "media"] {
            idx.entries.insert(name.into(), format!("hash-{name}"));
        }
        let once = serde_json::to_string(&idx).unwrap();
        let twice = serde_json::to_string(&idx).unwrap();
        assert_eq!(once, twice);
        assert!(once.find("alpha").unwrap() < once.find("zeta").unwrap());
    }

    #[test]
    fn unreferenced_chunks_are_listed_but_never_deleted() {
        let dir = store_dir("orphans");
        let t = FsTransport::new(&dir);
        let store = ChunkStore::new(&t);

        let kept = store.put(r#"{"keep":1}"#).unwrap();
        let orphan = store.put(r#"{"orphan":1}"#).unwrap();

        let mut idx = Index::default();
        idx.entries.insert("e".into(), kept.clone());

        let orphans = store.unreferenced(&idx).unwrap();
        assert_eq!(orphans, vec![orphan.clone()]);
        // Still readable: an orphan may be the only copy of a version someone wants.
        assert!(store.get(&orphan).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}
