use anyhow::{Context, Result, bail};
use std::io::Write;
use std::path::{Path, PathBuf};

const ORT_VERSION: &str = "1.26.0";

struct ModelSpec {
    dir: &'static str,
    hf_repo: &'static str,
    files: &'static [(&'static str, &'static str)],
    env_hint: &'static str,
}

fn spec(name: &str) -> Option<ModelSpec> {
    match name {
        "embed" | "embeddings" => Some(ModelSpec {
            dir: "models",
            hf_repo: "Xenova/multilingual-e5-small",
            files: &[
                ("onnx/model_quantized.onnx", "model_quantized.onnx"),
                ("tokenizer.json", "tokenizer.json"),
            ],
            env_hint: "ONNX_MODEL_PATH",
        }),
        "nli" => Some(ModelSpec {
            dir: "models-nli",
            hf_repo: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            files: &[
                ("onnx/model.onnx", "model.onnx"),
                ("tokenizer.json", "tokenizer.json"),
                ("config.json", "config.json"),
                ("tokenizer_config.json", "tokenizer_config.json"),
            ],
            env_hint: "CUBA_NLI_PATH",
        }),
        "reranker" => Some(ModelSpec {
            dir: "reranker",
            hf_repo: "celinehoang/bge-reranker-v2-m3-onnx",
            files: &[
                ("model.onnx", "model.onnx"),
                ("tokenizer.json", "tokenizer.json"),
                ("config.json", "config.json"),
            ],
            env_hint: "CUBA_RERANKER_PATH",
        }),
        _ => None,
    }
}

fn cache_root() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("no HOME/USERPROFILE — no sé dónde está la caché")?;
    Ok(PathBuf::from(home).join(".cache").join("cuba-memorys"))
}

pub async fn run_cli(args: &[String]) -> Result<()> {
    let what = args.first().map(String::as_str).unwrap_or("");
    match what {
        "embed" | "embeddings" | "nli" | "reranker" => {
            download_model(&spec(what).unwrap()).await?;
        }
        "runtime" => {
            download_runtime().await?;
        }
        "all" => {
            download_runtime().await?;
            for m in ["embed", "nli", "reranker"] {
                download_model(&spec(m).unwrap()).await?;
            }
        }
        "" | "-h" | "--help" | "help" => {
            print_help();
        }
        other => {
            bail!(
                "modelo desconocido `{other}`. Usá: embed | nli | reranker | runtime | all\n\
                 (`cuba-memorys models help` para más detalle)"
            );
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "cuba-memorys models <embed|nli|reranker|runtime|all>\n\n\
         Descarga los modelos ONNX y el runtime a ~/.cache/cuba-memorys/, en cualquier\n\
         sistema. cuba-memorys los encuentra ahí solo — no hace falta setear env vars.\n\n\
           embed      multilingual-e5-small (384-d) — búsqueda semántica. ~113 MB\n\
           nli        mDeBERTa-v3-xnli — verify sin LLM. ~1.1 GB\n\
           reranker   bge-reranker-v2-m3 — reordena candidatos. ~1.1 GB\n\
           runtime    libonnxruntime para tu plataforma. ~15 MB\n\
           all        el runtime + los tres modelos\n\n\
         Verificá con:  cuba-memorys doctor"
    );
}

async fn download_model(spec: &ModelSpec) -> Result<()> {
    let dir = cache_root()?.join(spec.dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creando {}", dir.display()))?;

    println!(
        "→ {} ({} archivos) en {}",
        spec.hf_repo,
        spec.files.len(),
        dir.display()
    );
    for (remote, local) in spec.files {
        let dest = dir.join(local);
        if dest.exists() && std::fs::metadata(&dest)?.len() > 0 {
            println!("  ✓ {local} (ya está)");
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{remote}",
            spec.hf_repo
        );
        print!("  ↓ {local} ... ");
        std::io::stdout().flush().ok();
        download_to(&url, &dest)
            .await
            .with_context(|| format!("descargando {local}"))?;
        let mb = std::fs::metadata(&dest)?.len() / 1_048_576;
        println!("{mb} MB");
    }
    println!(
        "  listo. cuba-memorys lo encuentra en {} (o {}=<ruta>)",
        dir.display(),
        spec.env_hint
    );
    Ok(())
}

fn runtime_target() -> Result<(String, &'static str, &'static str)> {
    let v = ORT_VERSION;
    let (archive, ext, lib) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => (
            format!("onnxruntime-linux-x64-{v}"),
            "tgz",
            "libonnxruntime.so",
        ),
        ("linux", "aarch64") => (
            format!("onnxruntime-linux-aarch64-{v}"),
            "tgz",
            "libonnxruntime.so",
        ),
        ("macos", "x86_64") => (
            format!("onnxruntime-osx-x86_64-{v}"),
            "tgz",
            "libonnxruntime.dylib",
        ),
        ("macos", "aarch64") => (
            format!("onnxruntime-osx-arm64-{v}"),
            "tgz",
            "libonnxruntime.dylib",
        ),
        ("windows", "x86_64") => (format!("onnxruntime-win-x64-{v}"), "zip", "onnxruntime.dll"),
        (os, arch) => bail!(
            "no tengo el runtime prearmado para {os}/{arch}. Instalá onnxruntime {v} a mano \
             y apuntá ORT_DYLIB_PATH a la librería."
        ),
    };
    let ext: &'static str = ext;
    let lib: &'static str = lib;
    Ok((archive, ext, lib))
}

async fn download_runtime() -> Result<()> {
    let (archive_stem, ext, lib_name) = runtime_target()?;
    let dir = cache_root()?.join("onnxruntime");
    std::fs::create_dir_all(&dir)?;
    let lib_dest = dir.join(lib_name);

    if lib_dest.exists() && std::fs::metadata(&lib_dest)?.len() > 0 {
        println!("→ runtime: {lib_name} ya está en {}", dir.display());
        return Ok(());
    }

    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{archive_stem}.{ext}"
    );
    println!(
        "→ runtime ({}/{}) desde {url}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let archive_path = dir.join(format!("{archive_stem}.{ext}"));
    print!("  ↓ descargando ... ");
    std::io::stdout().flush().ok();
    download_to(&url, &archive_path)
        .await
        .context("descargando el runtime")?;
    println!("{} MB", std::fs::metadata(&archive_path)?.len() / 1_048_576);

    print!("  ⇢ extrayendo {lib_name} ... ");
    std::io::stdout().flush().ok();
    extract_library(&archive_path, ext, lib_name, &lib_dest)
        .with_context(|| format!("extrayendo {lib_name} de {}", archive_path.display()))?;
    std::fs::remove_file(&archive_path).ok();
    println!("ok");
    println!(
        "  listo. cuba-memorys lo encuentra en {} (o ORT_DYLIB_PATH=<ruta>)",
        lib_dest.display()
    );
    Ok(())
}

fn extract_library(archive: &Path, ext: &str, lib_name: &str, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension("part");
    match ext {
        "tgz" => {
            let f = std::fs::File::open(archive)?;
            let gz = flate2::read::GzDecoder::new(f);
            let mut tar = tar::Archive::new(gz);
            let mut best: Option<(u64, PathBuf)> = None;
            let scratch = dest.with_file_name("cuba-ort-extract");
            std::fs::create_dir_all(&scratch)?;
            for entry in tar.entries()? {
                let mut entry = entry?;
                if entry.header().entry_type() != tar::EntryType::Regular {
                    continue;
                }
                let name = match entry.path()?.file_name().and_then(|n| n.to_str()) {
                    Some(n) if n.starts_with(lib_name) => n.to_string(),
                    _ => continue,
                };
                let size = entry.header().size().unwrap_or(0);
                let staged = scratch.join(&name);
                let mut out = std::fs::File::create(&staged)?;
                std::io::copy(&mut entry, &mut out)?;
                if best.as_ref().is_none_or(|(s, _)| size > *s) {
                    best = Some((size, staged));
                }
            }
            match best {
                Some((_, real)) => {
                    std::fs::rename(&real, dest)?;
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
                    }
                    std::fs::remove_dir_all(&scratch).ok();
                    Ok(())
                }
                None => {
                    std::fs::remove_dir_all(&scratch).ok();
                    bail!("no encontré {lib_name} dentro del .tgz");
                }
            }
        }
        "zip" => {
            let f = std::fs::File::open(archive)?;
            let mut zip = zip::ZipArchive::new(f)?;
            for i in 0..zip.len() {
                let mut file = zip.by_index(i)?;
                let is_lib = file
                    .enclosed_name()
                    .and_then(|p| p.file_name().map(|n| n.to_os_string()))
                    .is_some_and(|n| n == lib_name);
                if is_lib {
                    let mut out = std::fs::File::create(&tmp)?;
                    std::io::copy(&mut file, &mut out)?;
                    drop(out);
                    std::fs::rename(&tmp, dest)?;
                    return Ok(());
                }
            }
            bail!("no encontré {lib_name} dentro del .zip");
        }
        other => bail!("formato de archivo desconocido: {other}"),
    }
}

async fn download_to(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("cuba-memorys/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let resp = client.get(url).send().await?.error_for_status()?;
    let expected = resp.content_length();

    let tmp = dest.with_extension("part");
    let mut file =
        std::fs::File::create(&tmp).with_context(|| format!("creando {}", tmp.display()))?;
    let mut got: u64 = 0;
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        got += chunk.len() as u64;
        file.write_all(&chunk)?;
    }
    file.flush()?;
    drop(file);

    if let Some(exp) = expected
        && got != exp
    {
        std::fs::remove_file(&tmp).ok();
        bail!("descarga truncada: {got} de {exp} bytes");
    }
    std::fs::rename(&tmp, dest)?;
    Ok(())
}
