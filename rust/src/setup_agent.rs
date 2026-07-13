//! `cuba-memorys setup` — declarative MCP client registration.
//!
//! This exists because of a bug that cost real recall. `~/.mcp.json` had
//! `ONNX_MODEL_PATH` set; `~/.claude.json` did not. Every MCP process spawned
//! from the second config ran with no embedding model, so the vector branch of
//! the hybrid search returned an empty vec — silently. Retrieval degraded to
//! lexical-only and nothing in any response said so.
//!
//! Hand-maintaining N copies of the same env block across N clients is how that
//! happens. So this module does two things:
//!
//!   `setup check`  — audits every client config it can find and reports the
//!                    ones that are missing vars, or that disagree with each other.
//!   `setup <target>` — writes the correct block, with the binary's own absolute
//!                    path, plan-first, backing the file up before touching it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

/// Env vars an MCP process needs to run cuba-memorys at full strength.
/// Missing `ONNX_MODEL_PATH` or `ORT_DYLIB_PATH` does not fail — it degrades.
const REQUIRED_ENV: [&str; 3] = ["DATABASE_URL", "ONNX_MODEL_PATH", "ORT_DYLIB_PATH"];

/// Vars that must AGREE across every config, even when absent from some.
///
/// The embedding dimension is the one that matters most and the one this audit
/// originally missed. Two clients pointing at the same database with different
/// `CUBA_EMBEDDING_DIM` is not a cosmetic inconsistency: the one whose dimension
/// disagrees with the column cannot run a vector comparison at all, so its hybrid
/// search silently becomes lexical. It happened — a stale project-level
/// `.mcp.json` kept spawning 384-d servers against a 1024-d column while the
/// audit reported "all consistent", because it was not looking at this var, and
/// not looking in that file.
const MUST_AGREE: [&str; 2] = ["CUBA_EMBEDDING_DIM", "CUBA_EMBED_MODEL"];

fn home() -> PathBuf {
    std::env::var("HOME").map_or_else(|_| PathBuf::from("."), PathBuf::from)
}

/// Client configs we know how to read and write.
fn known_targets() -> Vec<(&'static str, PathBuf)> {
    vec![
        ("claude", home().join(".claude.json")),
        ("mcp", home().join(".mcp.json")),
        ("cursor", home().join(".cursor").join("mcp.json")),
    ]
}

/// Project-level `.mcp.json` files, which spawn servers just like the global
/// ones and were invisible to this audit.
///
/// Bounded on purpose: a shallow walk of the usual project roots rather than a
/// scan of the whole home directory. Missing a config in an unusual place is a
/// gap; hanging for a minute on every `setup check` is a bug.
fn project_configs() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let roots = [home().join("proyectos"), home().join("projects")];

    for root in roots.iter().filter(|r| r.is_dir()) {
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten().take(200) {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            // The project itself, and one level of nesting (monorepos, MCP/ dirs).
            let mut candidates = vec![dir.join(".mcp.json")];
            if let Ok(subs) = std::fs::read_dir(&dir) {
                for sub in subs.flatten().take(100) {
                    if sub.path().is_dir() {
                        candidates.push(sub.path().join(".mcp.json"));
                    }
                }
            }
            for c in candidates {
                if c.is_file()
                    && read_json(&c).as_ref().and_then(cuba_block).is_some()
                {
                    let label = c
                        .parent()
                        .and_then(|p| p.file_name())
                        .map_or_else(|| "proyecto".to_string(), |n| n.to_string_lossy().to_string());
                    out.push((label, c));
                }
            }
        }
    }
    out
}

/// The block a client needs to spawn this exact binary, correctly.
fn desired_config() -> Result<Value> {
    // The binary's own path — not `which`, not a relative path. A client that
    // resolves the command differently is how you end up running a stale binary.
    let exe = std::env::current_exe().context("no se pudo resolver la ruta del binario")?;

    let db = std::env::var("DATABASE_URL").unwrap_or_default();
    let onnx = std::env::var("ONNX_MODEL_PATH")
        .unwrap_or_else(|_| home().join(".cache/cuba-memorys/models").display().to_string());
    let ort = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| {
        home()
            .join(".cache/cuba-memorys/onnxruntime/libonnxruntime.so")
            .display()
            .to_string()
    });

    Ok(json!({
        "command": exe.display().to_string(),
        "args": [],
        "env": {
            "DATABASE_URL": db,
            "ONNX_MODEL_PATH": onnx,
            "ORT_DYLIB_PATH": ort,
        }
    }))
}

fn read_json(path: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn cuba_block(cfg: &Value) -> Option<&Value> {
    cfg.get("mcpServers")?.get("cuba-memorys")
}

// ---------------------------------------------------------------------------
// check — the audit that would have caught the dead vector branch
// ---------------------------------------------------------------------------

fn run_check() -> Result<()> {
    println!("Auditoría de configuraciones de cliente MCP\n");

    let mut found = 0;
    let mut problems = 0;
    // env var → set of distinct values seen across configs (divergence detector)
    let mut seen: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();

    // Global configs AND project-level ones. A project `.mcp.json` spawns a server
    // exactly like a global config does, and leaving them out of the audit is how
    // a stale one kept serving degraded results while this command said "all fine".
    let targets: Vec<(String, PathBuf)> = known_targets()
        .into_iter()
        .map(|(n, p)| (n.to_string(), p))
        .chain(project_configs())
        .collect();

    for (name, path) in targets {
        let Some(cfg) = read_json(&path) else {
            continue;
        };
        let Some(block) = cuba_block(&cfg) else {
            continue;
        };
        found += 1;

        println!("── {name}  ({})", path.display());
        let command = block.get("command").and_then(Value::as_str).unwrap_or("");
        println!("   command: {command}");
        if !command.is_empty() && !Path::new(command).exists() {
            println!("   PROBLEMA: ese binario no existe");
            problems += 1;
        }

        let env = block.get("env").and_then(Value::as_object);
        for key in REQUIRED_ENV {
            match env.and_then(|e| e.get(key)).and_then(Value::as_str) {
                Some(v) if !v.is_empty() => {
                    seen.entry(key.to_string()).or_default().insert(v.to_string());
                    // A path that points nowhere is as bad as an absent one.
                    if key != "DATABASE_URL" && !Path::new(v).exists() {
                        println!("   PROBLEMA: {key} apunta a una ruta inexistente ({v})");
                        problems += 1;
                    } else {
                        println!("   {key}: ok");
                    }
                }
                _ => {
                    println!("   PROBLEMA: falta {key}");
                    if key == "ONNX_MODEL_PATH" || key == "ORT_DYLIB_PATH" {
                        println!(
                            "     → sin esto la rama vectorial devuelve vacío EN SILENCIO: \
                             la búsqueda queda solo léxica"
                        );
                    }
                    problems += 1;
                }
            }
        }

        // Absence is a value here, not a gap to skip: a config with no
        // CUBA_EMBEDDING_DIM runs at the 384-d default, and that DISAGREES with a
        // config that sets 1024. Recording it as "(default)" is what makes the
        // divergence check below able to see it at all.
        for key in MUST_AGREE {
            let value = env
                .and_then(|e| e.get(key))
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
                .map_or_else(
                    || match key {
                        "CUBA_EMBEDDING_DIM" => "384 (default)".to_string(),
                        _ => "multilingual-e5-small (default)".to_string(),
                    },
                    std::string::ToString::to_string,
                );
            println!("   {key}: {value}");
            seen.entry(key.to_string()).or_default().insert(value);
        }
        println!();
    }

    if found == 0 {
        println!("No encontré ninguna config con un bloque «cuba-memorys».");
        println!("Generá una con:  cuba-memorys setup print");
        return Ok(());
    }

    // The actual bug: two configs, same server, different env.
    for (key, values) in &seen {
        if values.len() > 1 {
            println!("DIVERGENCIA en {key}: los clientes no coinciden");
            for v in values {
                println!("   - {v}");
            }
            if key == "CUBA_EMBEDDING_DIM" || key == "CUBA_EMBED_MODEL" {
                // This one is not a wobble, it is a hard break.
                println!(
                    "   → GRAVE. Los clientes hablan con la MISMA base con modelos de dimensiones\n\
                     \x20    distintas. El que no coincida con la columna no puede comparar vectores:\n\
                     \x20    su búsqueda híbrida se vuelve LÉXICA sin avisar, y devuelve peores\n\
                     \x20    resultados con la misma confianza. Alineá todas las configs.\n"
                );
            } else {
                println!(
                    "   → los procesos MCP se comportan distinto según quién los lance. \
                     Esto es exactamente el bug que mató el recall vectorial.\n"
                );
            }
            problems += 1;
        }
    }

    if problems == 0 {
        println!("{found} config(s) revisada(s): todas completas y coherentes entre sí.");
    } else {
        println!("{found} config(s) revisada(s): {problems} problema(s).");
        println!("Arreglalo con:  cuba-memorys setup <cliente> --apply");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// write — plan first, backup, then merge
// ---------------------------------------------------------------------------

fn run_write(target: &str, apply: bool) -> Result<()> {
    let path = known_targets()
        .into_iter()
        .find(|(n, _)| *n == target)
        .map(|(_, p)| p)
        .with_context(|| format!("cliente desconocido: {target} (claude | mcp | cursor)"))?;

    let desired = desired_config()?;

    // Warn about what `check` would flag, before writing it in.
    if let Some(env) = desired.get("env").and_then(Value::as_object) {
        for key in ["ONNX_MODEL_PATH", "ORT_DYLIB_PATH"] {
            if let Some(v) = env.get(key).and_then(Value::as_str)
                && !Path::new(v).exists()
            {
                println!("AVISO: {key} apunta a {v}, que no existe todavía.");
                println!("       El servidor arrancará, pero sin búsqueda vectorial.\n");
            }
        }
    }

    println!("Se escribiría en {}:\n", path.display());
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({ "mcpServers": { "cuba-memorys": desired } }))?
    );
    println!();

    if !apply {
        println!("Esto fue un plan — no se tocó ningún archivo.");
        println!("Para aplicarlo:  cuba-memorys setup {target} --apply");
        return Ok(());
    }

    // Merge into whatever is already there: these files hold the user's other
    // MCP servers and unrelated settings. Overwriting them wholesale would be
    // destroying config to install config.
    let mut cfg = read_json(&path).unwrap_or_else(|| json!({}));
    if !cfg.is_object() {
        bail!("{} no contiene un objeto JSON en la raíz", path.display());
    }

    if path.exists() {
        let backup = path.with_extension(format!(
            "json.bak-{}",
            chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
        ));
        std::fs::copy(&path, &backup)
            .with_context(|| format!("no se pudo respaldar {}", path.display()))?;
        println!("Backup: {}", backup.display());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    cfg.as_object_mut()
        .expect("checked above")
        .entry("mcpServers")
        .or_insert_with(|| json!({}));
    cfg["mcpServers"]["cuba-memorys"] = desired;

    std::fs::write(&path, serde_json::to_string_pretty(&cfg)?)
        .with_context(|| format!("no se pudo escribir {}", path.display()))?;

    println!("Escrito en {}", path.display());
    println!("Reiniciá el cliente MCP para que tome el binario y el entorno nuevos.");
    Ok(())
}

pub fn run_cli(args: &[String]) -> Result<()> {
    let mut target: Option<String> = None;
    let mut apply = false;

    for a in args {
        match a.as_str() {
            "--apply" => apply = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys setup <check | print | claude | mcp | cursor> [--apply]\n\n\
                     check   audita las configs existentes: variables faltantes, rutas muertas,\n\
                             y divergencias entre clientes (el bug que mató el recall vectorial).\n\
                     print   imprime el bloque correcto para pegarlo donde haga falta.\n\
                     hook    instala un SessionStart que inyecta la memoria automáticamente.\n\
                     claude  ~/.claude.json     mcp  ~/.mcp.json     cursor  ~/.cursor/mcp.json\n\n\
                     Sin --apply, solo muestra el plan. Con --apply, respalda el archivo y mergea."
                );
                return Ok(());
            }
            other => target = Some(other.to_string()),
        }
    }

    match target.as_deref() {
        Some("hook") => run_hook(apply),
        Some("check") | None => run_check(),
        Some("print") => {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({ "mcpServers": { "cuba-memorys": desired_config()? } })
                )?
            );
            Ok(())
        }
        Some(t) => run_write(t, apply),
    }
}

// ---------------------------------------------------------------------------
// hook — make the memory load itself
// ---------------------------------------------------------------------------

/// Install a `SessionStart` hook that injects the brain's context.
///
/// The CLAUDE.md in this setup already *asks* the agent to start every session
/// with `cuba_faro`, and the same file lists forgetting to do it as anti-pattern
/// AP2. Asking is the problem. A hook cannot forget.
fn run_hook(apply: bool) -> Result<()> {
    let exe = std::env::current_exe().context("no se pudo resolver la ruta del binario")?;
    let path = home().join(".claude").join("settings.json");

    // `--quiet` so a project with no memory prints nothing at all rather than a
    // header over an empty result: an empty section injected into every session
    // is pure cost.
    let command = format!("{} recall --quiet", exe.display());
    let hook = json!({
        "matcher": "*",
        "hooks": [{ "type": "command", "command": command }]
    });

    println!("Se añadiría a {} un hook SessionStart:\n", path.display());
    println!("{}\n", serde_json::to_string_pretty(&hook)?);
    println!("Inyecta la última sesión, los errores sin resolver y las decisiones ya tomadas");
    println!("— unos 300 tokens — antes de que el agente escriba nada.\n");

    if !apply {
        println!("Esto fue un plan — no se tocó ningún archivo.");
        println!("Para aplicarlo:  cuba-memorys setup hook --apply");
        return Ok(());
    }

    let mut cfg = read_json(&path).unwrap_or_else(|| json!({}));
    if !cfg.is_object() {
        bail!("{} no contiene un objeto JSON en la raíz", path.display());
    }

    if path.exists() {
        let backup = path.with_extension(format!(
            "json.bak-{}",
            chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
        ));
        std::fs::copy(&path, &backup)
            .with_context(|| format!("no se pudo respaldar {}", path.display()))?;
        println!("Backup: {}", backup.display());
    }

    let obj = cfg.as_object_mut().expect("checked above");
    obj.entry("hooks").or_insert_with(|| json!({}));
    let hooks = cfg["hooks"]
        .as_object_mut()
        .context("hooks no es un objeto")?;
    hooks.entry("SessionStart").or_insert_with(|| json!([]));
    let list = cfg["hooks"]["SessionStart"]
        .as_array_mut()
        .context("SessionStart no es una lista")?;

    // Idempotent: re-running must not stack a second copy of the same hook.
    let already = list.iter().any(|h| {
        h.get("hooks")
            .and_then(Value::as_array)
            .is_some_and(|inner| {
                inner.iter().any(|i| {
                    i.get("command")
                        .and_then(Value::as_str)
                        .is_some_and(|c| c.contains("cuba-memorys") && c.contains("recall"))
                })
            })
    });
    if already {
        println!("El hook ya estaba instalado — no se duplicó.");
        return Ok(());
    }

    list.push(hook);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, serde_json::to_string_pretty(&cfg)?)
        .with_context(|| format!("no se pudo escribir {}", path.display()))?;

    println!("Instalado. La próxima sesión arrancará con la memoria ya cargada,");
    println!("obedezca el modelo su CLAUDE.md o no.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_carries_the_vars_whose_absence_is_silent() {
        let cfg = desired_config().expect("current_exe resolves under test");
        let env = cfg.get("env").and_then(Value::as_object).expect("env block");
        for key in REQUIRED_ENV {
            assert!(env.contains_key(key), "falta {key} en el bloque generado");
        }
        // An absolute path: a client resolving `cuba-memorys` off $PATH is how
        // you silently run yesterday's binary.
        let command = cfg.get("command").and_then(Value::as_str).unwrap_or("");
        assert!(Path::new(command).is_absolute(), "el command debe ser absoluto");
    }

    /// The divergence that actually broke production, in test form.
    ///
    /// A config with no CUBA_EMBEDDING_DIM runs at the 384-d default. Treating
    /// "absent" as "nothing to compare" is what made the audit report
    /// "all consistent" while a stale project config kept spawning 384-d servers
    /// against a 1024-d column.
    #[test]
    fn an_absent_dimension_is_a_value_that_can_disagree() {
        use std::collections::HashSet;
        let mut seen: HashSet<String> = HashSet::new();

        // Config A sets it explicitly.
        seen.insert("1024".to_string());
        // Config B omits it — which is NOT the same as agreeing.
        seen.insert("384 (default)".to_string());

        assert_eq!(seen.len(), 2, "una config sin la var diverge de una que la fija");
    }

    #[test]
    fn the_vars_that_must_agree_include_the_dimension() {
        // The audit originally checked ONNX paths but not the dimension — and the
        // dimension is the one whose mismatch silently kills the vector branch.
        assert!(MUST_AGREE.contains(&"CUBA_EMBEDDING_DIM"));
        assert!(MUST_AGREE.contains(&"CUBA_EMBED_MODEL"));
    }

    #[test]
    fn finds_the_cuba_block_only_when_present() {
        let with = json!({"mcpServers": {"cuba-memorys": {"command": "/bin/x"}}});
        let without = json!({"mcpServers": {"otro": {"command": "/bin/y"}}});
        assert!(cuba_block(&with).is_some());
        assert!(cuba_block(&without).is_none());
        assert!(cuba_block(&json!({})).is_none());
    }
}
