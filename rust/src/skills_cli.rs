use anyhow::{Context, Result};

use crate::handlers::receta;

const TRUSTED: f64 = 0.5;

fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "procedimiento".to_string()
    } else {
        trimmed
    }
}

fn yaml_quote(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', " ")
    )
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let mut dir: Option<String> = None;
    let mut min_reliability = 0.0_f64;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--min-reliability" => {
                min_reliability = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .context("--min-reliability needs a float in [0,1]")?
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: cuba-memorys skills <dir> [--min-reliability 0.0]\n\n\
                     Exports procedural memory as Claude Code Skills — one folder with a\n\
                     SKILL.md each. The agent loads them lazily: it sees the name and the\n\
                     description, and reads the body only when the task matches.\n\n\
                     Each Skill carries its own track record, so the agent knows what it is\n\
                     trusting. Typical target: ~/.claude/skills"
                );
                return Ok(());
            }
            other => dir = Some(other.to_string()),
        }
    }
    let Some(dir) = dir else {
        anyhow::bail!("falta el directorio — uso: cuba-memorys skills ~/.claude/skills");
    };

    let url = crate::setup::resolve_database_url().await;
    let pool = crate::db::create_pool(&url)
        .await
        .context("connecting to database for skills export")?;

    let procedures = receta::all_for_export(&pool).await?;
    if procedures.is_empty() {
        println!("No hay procedimientos todavía.");
        println!("Guardá el primero con la tool cuba_receta (action=add), o dejá que el agente");
        println!("lo haga la próxima vez que descubra cómo se levanta algo.");
        return Ok(());
    }

    let root = std::path::Path::new(&dir);
    let mut written = 0usize;
    let mut skipped = 0usize;

    for p in &procedures {
        let reliability = p["reliability"].as_f64().unwrap_or(0.0);
        if reliability < min_reliability {
            skipped += 1;
            continue;
        }
        let name = p["name"].as_str().unwrap_or("procedimiento");
        let trigger = p["trigger"].as_str().unwrap_or("");
        let markdown = p["markdown"].as_str().unwrap_or("");
        let preconditions = p["preconditions"].as_str().unwrap_or("");
        let verification = p["verification"].as_str().unwrap_or("");
        let successes = p["successes"].as_i64().unwrap_or(0);
        let failures = p["failures"].as_i64().unwrap_or(0);

        let folder = root.join(slug(name));
        std::fs::create_dir_all(&folder)
            .with_context(|| format!("no se pudo crear {}", folder.display()))?;

        let description = if trigger.is_empty() {
            format!("Cómo: {name}.")
        } else {
            format!("Usar {trigger}. Procedimiento verificado: {name}.")
        };

        let mut body = String::new();
        body.push_str("---\n");
        body.push_str(&format!("name: {}\n", yaml_quote(name)));
        body.push_str(&format!("description: {}\n", yaml_quote(&description)));
        body.push_str("---\n\n");
        body.push_str(&format!("# {name}\n\n"));

        let total = successes + failures;
        if total == 0 {
            body.push_str(
                "> **Sin historial.** Este procedimiento nunca se ejecutó, así que no hay evidencia \
                 de que funcione. Verificá cada paso y reportá el resultado con \
                 `cuba_receta action=outcome`.\n\n",
            );
        } else {
            let rate = successes as f64 / total as f64;
            let pct = rate * 100.0;

            let (mark, caveat) = if reliability >= TRUSTED {
                ("Probado", None)
            } else if rate < 0.5 {
                (
                    "POCO FIABLE",
                    Some(
                        "> Falla más veces de las que funciona. Tratalo como una hipótesis, \
                         no como una receta: probablemente esté desactualizado.\n\n",
                    ),
                )
            } else {
                (
                    "Sin evidencia suficiente",
                    Some(
                        "> Ha funcionado más de lo que ha fallado, pero se ejecutó demasiado \
                         pocas veces como para prometer nada. Verificá los pasos y reportá el \
                         resultado — con unas pocas ejecuciones más, esto se vuelve fiable.\n\n",
                    ),
                )
            };

            body.push_str(&format!(
                "> **{mark}: {successes} de {total} veces** ({pct:.0}%, fiabilidad {reliability:.2}).\n\n"
            ));
            if let Some(c) = caveat {
                body.push_str(c);
            }
        }

        if !preconditions.is_empty() {
            body.push_str(&format!("## Antes de empezar\n\n{preconditions}\n\n"));
        }
        body.push_str(&format!("## Pasos\n\n{markdown}\n"));
        if !verification.is_empty() {
            body.push_str(&format!("## Cómo saber que funcionó\n\n{verification}\n\n"));
        }
        body.push_str(
            "---\n_Generado desde la memoria procedimental de cuba-memorys. \
             Después de ejecutarlo, reportá el resultado con `cuba_receta action=outcome` — \
             si no, la memoria no aprende nada._\n",
        );

        std::fs::write(folder.join("SKILL.md"), body)
            .with_context(|| format!("no se pudo escribir la Skill de «{name}»"))?;
        written += 1;
    }

    println!("{written} Skill(s) escritas en {dir}");
    if skipped > 0 {
        println!("{skipped} omitida(s) por fiabilidad < {min_reliability:.2}");
    }
    println!("\nEl agente las carga solas: ve el nombre y la descripción, y lee el cuerpo");
    println!("solo cuando la tarea coincide. No gastan contexto hasta que hacen falta.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_survive_real_names() {
        assert_eq!(
            slug("Levantar el entorno de desarrollo"),
            "levantar-el-entorno-de-desarrollo"
        );
        assert_eq!(slug("migrar a bge-m3 (1024-d)"), "migrar-a-bge-m3-1024-d");
        assert_eq!(slug("   "), "procedimiento");
        assert_eq!(slug("¿cómo?"), "c-mo");
    }

    #[test]
    fn low_confidence_is_not_the_same_as_a_bad_procedure() {
        fn verdict(successes: i64, failures: i64) -> &'static str {
            let total = successes + failures;
            if total == 0 {
                return "sin historial";
            }
            let reliability = crate::handlers::receta::wilson_lower_bound(successes, failures);
            let rate = successes as f64 / total as f64;
            if reliability >= TRUSTED {
                "probado"
            } else if rate < 0.5 {
                "poco fiable"
            } else {
                "sin evidencia suficiente"
            }
        }

        assert_eq!(verdict(3, 1), "sin evidencia suficiente");
        assert_eq!(verdict(1, 9), "poco fiable");
        assert_eq!(verdict(0, 0), "sin historial");
        assert_eq!(verdict(47, 3), "probado");
    }

    #[test]
    fn frontmatter_cannot_be_broken_by_a_procedure_name() {
        assert_eq!(yaml_quote("simple"), "\"simple\"");
        assert_eq!(yaml_quote("con \"comillas\""), "\"con \\\"comillas\\\"\"");
        assert_eq!(yaml_quote("dos\nlíneas"), "\"dos líneas\"");
    }
}
