//! Where a sync bundle lives, decoupled from what it contains.
//!
//! The exporter used to call `std::fs::write` directly, which welded the format
//! to the local filesystem. A [`Transport`] is the seam: the content-addressed
//! store (see [`super::cas`]) speaks in blobs and names, and something else
//! decides whether those land on disk, in a git checkout, or in object storage.
//!
//! Only [`FsTransport`] ships today, and that is on purpose — an S3 backend with
//! no user is worse than no S3 backend. The point of the trait is that adding one
//! later touches this file and nothing else.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub trait Transport: Send + Sync {
    /// Write a blob. Must be atomic enough that a crash never leaves a
    /// half-written object readable under its final name — a truncated chunk
    /// named by the hash of its complete form is a corrupt store that looks intact.
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;

    fn get(&self, key: &str) -> Result<Vec<u8>>;

    fn exists(&self, key: &str) -> bool;

    /// Keys under a prefix. Order is unspecified.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

/// Local filesystem, rooted at a directory.
pub struct FsTransport {
    root: PathBuf,
}

impl FsTransport {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        // Canonicalize up front: the traversal guard compares prefixes, and
        // `/tmp/x` vs `/private/tmp/x` would make an in-root path look foreign.
        let root = root.canonicalize().unwrap_or(root);
        Self { root }
    }

    /// Resolve a key to a path, refusing anything that climbs out of the root.
    /// Keys are hashes today, but an importer will one day read keys from an
    /// index file it did not write.
    fn path_for(&self, key: &str) -> Result<PathBuf> {
        let candidate = self.root.join(key);
        super::paths::ensure_within(&self.root, &candidate)
            .with_context(|| format!("clave fuera del directorio de sync: {key}"))?;
        Ok(candidate)
    }
}

impl Transport for FsTransport {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.path_for(key)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("no se pudo crear {}", parent.display()))?;
        }
        // Write to a sibling temp file, then rename: rename is atomic within a
        // filesystem, so a reader never observes a partial chunk under its
        // final, content-derived name.
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)
            .with_context(|| format!("no se pudo escribir {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("no se pudo mover a {}", path.display()))?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.path_for(key)?;
        std::fs::read(&path).with_context(|| format!("no se pudo leer {}", path.display()))
    }

    fn exists(&self, key: &str) -> bool {
        self.path_for(key).is_ok_and(|p| p.exists())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let base = self.path_for(prefix)?;
        if !base.exists() {
            return Ok(Vec::new());
        }
        let mut keys = Vec::new();
        collect(&base, &self.root, &mut keys)?;
        Ok(keys)
    }
}

fn collect(dir: &Path, root: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, root, out)?;
        } else if let Ok(rel) = path.strip_prefix(root)
            && path.extension().and_then(|e| e.to_str()) != Some("tmp")
        {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One directory per test. Naming them by pid alone made the three tests in
    /// this module share a directory, and the test runner is parallel: whichever
    /// finished first deleted the tree out from under the others.
    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("cuba-transport-{}-{tag}", std::process::id()));
        std::fs::remove_dir_all(&p).ok();
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn round_trips_a_blob() {
        let dir = tmpdir("roundtrip");
        let t = FsTransport::new(&dir);
        t.put("chunks/ab/cd.json", b"hola").unwrap();
        assert!(t.exists("chunks/ab/cd.json"));
        assert_eq!(t.get("chunks/ab/cd.json").unwrap(), b"hola");
        assert!(
            t.list("chunks")
                .unwrap()
                .contains(&"chunks/ab/cd.json".to_string())
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_to_escape_its_root() {
        let dir = tmpdir("escape");
        let t = FsTransport::new(&dir);
        // An index file is data, and data can be hostile.
        assert!(t.put("../../../etc/passwd", b"x").is_err());
        assert!(t.get("../../secret").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_half_written_chunk_survives_under_its_final_name() {
        let dir = tmpdir("atomic");
        let t = FsTransport::new(&dir);
        t.put("a/b.json", b"contenido completo").unwrap();
        // The temp file must not linger, or `list` would hand it to an importer.
        assert!(!t.list("a").unwrap().iter().any(|k| k.ends_with(".tmp")));
        std::fs::remove_dir_all(&dir).ok();
    }
}
