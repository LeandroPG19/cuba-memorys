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

    for (name, path) in known_targets() {
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
            println!(
                "   → los procesos MCP se comportan distinto según quién los lance. \
                 Esto es exactamente el bug que mató el recall vectorial.\n"
            );
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

    #[test]
    fn finds_the_cuba_block_only_when_present() {
        let with = json!({"mcpServers": {"cuba-memorys": {"command": "/bin/x"}}});
        let without = json!({"mcpServers": {"otro": {"command": "/bin/y"}}});
        assert!(cuba_block(&with).is_some());
        assert!(cuba_block(&without).is_none());
        assert!(cuba_block(&json!({})).is_none());
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
