pub mod python_lang;
pub mod rust_lang;

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Class,
    Module,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SymbolKind::Function => "function",
            SymbolKind::Struct => "struct",
            SymbolKind::Class => "class",
            SymbolKind::Module => "module",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Symbol {
    /// Qualified, stable identity: "relative/path.rs::name". Two files can each
    /// have a function called `new` — only the qualified name is unique.
    pub qualified_name: String,
    pub simple_name: String,
    pub kind: SymbolKind,
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: String,
    /// Simple names this symbol's body calls, exactly as written in source —
    /// resolution against the rest of the batch happens after every file is
    /// parsed, not per-file, so call target order never matters.
    pub calls: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Calls,
    Imports,
}

impl EdgeKind {
    pub fn as_relation_type(&self) -> &'static str {
        match self {
            EdgeKind::Calls => "uses",
            EdgeKind::Imports => "depends_on",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone)]
pub struct ModuleImports {
    pub file: String,
    /// Raw import paths as written (e.g. `crate::db::create_pool`, `os.path`) —
    /// not resolved to a symbol, since that needs a full module-resolution pass
    /// this extractor deliberately doesn't attempt.
    pub paths: Vec<String>,
}

#[derive(Debug, Default)]
pub struct ExtractionResult {
    pub symbols: Vec<Symbol>,
    pub imports: Vec<ModuleImports>,
    pub files_parsed: usize,
    pub files_skipped: Vec<(String, String)>,
}

pub fn extract_dir(root: &Path, extensions: &[&str]) -> Result<ExtractionResult> {
    let mut result = ExtractionResult::default();
    walk(root, root, extensions, &mut result)?;
    Ok(result)
}

fn walk(root: &Path, dir: &Path, extensions: &[&str], out: &mut ExtractionResult) -> Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();

        if path.is_dir() {
            if matches!(
                name.as_ref(),
                "target" | "node_modules" | ".git" | "dist" | "build" | "__pycache__" | ".venv"
            ) {
                continue;
            }
            walk(root, &path, extensions, out)?;
            continue;
        }

        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !extensions.contains(&ext) {
            continue;
        }

        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                out.files_skipped.push((rel, e.to_string()));
                continue;
            }
        };

        let parsed = match ext {
            "rs" => rust_lang::extract(&rel, &source),
            "py" => python_lang::extract(&rel, &source),
            _ => continue,
        };

        match parsed {
            Ok((symbols, imports)) => {
                out.symbols.extend(symbols);
                if !imports.is_empty() {
                    out.imports.push(ModuleImports {
                        file: rel,
                        paths: imports,
                    });
                }
                out.files_parsed += 1;
            }
            Err(e) => out.files_skipped.push((rel, e.to_string())),
        }
    }
    Ok(())
}

/// Resolve `calls` (simple names) into `Calls` edges. A call only becomes an
/// edge when its simple name matches EXACTLY ONE symbol in the whole batch —
/// ambiguous (0 or 2+ candidates) names are dropped rather than guessed, so
/// every edge this produces is something the AST actually supports, not a
/// heuristic pretending to be one.
pub fn resolve_call_edges(symbols: &[Symbol]) -> Vec<Edge> {
    let mut by_simple_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in symbols {
        by_simple_name
            .entry(s.simple_name.as_str())
            .or_default()
            .push(s.qualified_name.as_str());
    }

    let mut edges = Vec::new();
    for s in symbols {
        for callee in &s.calls {
            if let Some(candidates) = by_simple_name.get(callee.as_str())
                && candidates.len() == 1
                && candidates[0] != s.qualified_name
            {
                edges.push(Edge {
                    from: s.qualified_name.clone(),
                    to: candidates[0].to_string(),
                    kind: EdgeKind::Calls,
                });
            }
        }
    }
    edges
}

pub fn line_of_byte(source: &str, byte_offset: usize) -> usize {
    source[..byte_offset.min(source.len())]
        .bytes()
        .filter(|&b| b == b'\n')
        .count()
        + 1
}

pub fn default_extensions_for(langs: &[String]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for l in langs {
        match l.as_str() {
            "rust" => out.push("rs"),
            "python" => out.push("py"),
            _ => {}
        }
    }
    if out.is_empty() {
        out = vec!["rs", "py"];
    }
    out
}

pub fn resolve_path(path_arg: Option<&str>) -> PathBuf {
    match path_arg {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from("."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbol(qualified: &str, simple: &str, calls: &[&str]) -> Symbol {
        Symbol {
            qualified_name: qualified.to_string(),
            simple_name: simple.to_string(),
            kind: SymbolKind::Function,
            file: "f.rs".to_string(),
            line_start: 1,
            line_end: 1,
            signature: String::new(),
            calls: calls.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn an_unambiguous_call_becomes_an_edge() {
        let symbols = vec![symbol("f.rs::a", "a", &["b"]), symbol("f.rs::b", "b", &[])];
        let edges = resolve_call_edges(&symbols);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, "f.rs::a");
        assert_eq!(edges[0].to, "f.rs::b");
    }

    #[test]
    fn an_ambiguous_callee_name_is_dropped_not_guessed() {
        let symbols = vec![
            symbol("f.rs::a", "a", &["new"]),
            symbol("f.rs::Foo::new", "new", &[]),
            symbol("f.rs::Bar::new", "new", &[]),
        ];
        let edges = resolve_call_edges(&symbols);
        assert!(
            edges.is_empty(),
            "two candidates named `new` — guessing either would be a fabricated edge"
        );
    }

    #[test]
    fn a_call_to_an_unknown_name_produces_no_edge() {
        let symbols = vec![symbol("f.rs::a", "a", &["println"])];
        assert!(resolve_call_edges(&symbols).is_empty());
    }

    #[test]
    fn self_recursive_calls_do_not_create_a_self_loop() {
        let symbols = vec![symbol("f.rs::a", "a", &["a"])];
        assert!(resolve_call_edges(&symbols).is_empty());
    }
}
