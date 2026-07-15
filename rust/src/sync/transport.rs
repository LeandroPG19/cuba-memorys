use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub trait Transport: Send + Sync {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;

    fn get(&self, key: &str) -> Result<Vec<u8>>;

    fn exists(&self, key: &str) -> bool;

    fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

pub struct FsTransport {
    root: PathBuf,
}

impl FsTransport {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root = root.canonicalize().unwrap_or(root);
        Self { root }
    }

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
        assert!(t.put("../../../etc/passwd", b"x").is_err());
        assert!(t.get("../../secret").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_half_written_chunk_survives_under_its_final_name() {
        let dir = tmpdir("atomic");
        let t = FsTransport::new(&dir);
        t.put("a/b.json", b"contenido completo").unwrap();
        assert!(!t.list("a").unwrap().iter().any(|k| k.ends_with(".tmp")));
        std::fs::remove_dir_all(&dir).ok();
    }
}
