//! `cuba-memorys models <embed|nli|reranker|runtime|all>` — get the ONNX models and
//! the runtime, on any platform, with one command.
//!
//! # Why this replaces the shell scripts
//!
//! Onboarding used to be three `.sh` files that told you to `curl` a model and `export`
//! an env var. On Windows that is nothing at all: bash does not run, the scripts do not
//! ship in the npm package, and there is no `libonnxruntime` to point at. A real field
//! install on Windows came up with hash-based embeddings — semantically inert — and no
//! way to fix it. This subcommand is the fix: it is the binary itself, so it runs
//! wherever the binary runs, and it fetches both the weights and the runtime.
//!
//! Downloads are atomic (temp file, then rename) and size-checked against the server's
//! Content-Length, because a truncated model that `ls` reports as present is the exact
//! silent-degradation failure this project keeps hitting.

use anyhow::{Context, Result, bail};
use std::io::Write;
use std::path::{Path, PathBuf};

/// onnxruntime version matching `ort = 2.0.0-rc.12`. Bump both together.
///
/// This is 1.26.0, not the 1.21.0 the old shell scripts named. It is not a free
/// choice: with 1.21.0 the runtime loads, connects, and then HANGS on the first
/// session — ort 2.0-rc expects a newer ABI, and the mismatch does not error, it
/// wedges. The number that works was read straight off the runtime this project
/// already ships (`strings libonnxruntime.so | grep '^1\.'`). Bump it in lockstep
/// with the `ort` dependency, and verify a real inference after, because the failure
/// mode is a silent hang, not a link error.
const ORT_VERSION: &str = "1.26.0";

struct ModelSpec {
    /// Cache subdirectory under ~/.cache/cuba-memorys/.
    dir: &'static str,
    hf_repo: &'static str,
    /// (path-in-repo, local-filename)
    files: &'static [(&'static str, &'static str)],
    env_hint: &'static str,
}

fn spec(name: &str) -> Option<ModelSpec> {
    match name {
        "embed" | "embeddings" => Some(ModelSpec {
            dir: "models",
            // Xenova, not Teradata. Teradata's `onnx/model_int8.onnx` is a different
            // int8 export whose ops HANG the ONNX runtime on load — it connects, then
            // never returns, the exact silent hang this project has fought before.
            // Xenova is the canonical Transformers.js export, battle-tested, and the
            // fallback the old download_model.sh already reached for. Verified to load.
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
            // BAAI/bge-reranker-v2-m3 ships only PyTorch; the ONNX export lives in a
            // community repo. This one has model.onnx (fp16, ~1.1 GB) at the root, the
            // layout the loader wants. NOT a *-quant repo: int8 wrecks this architecture
            // the same way it wrecked the NLI judge.
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
        .or_else(|_| std::env::var("USERPROFILE")) // Windows
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

    println!("→ {} ({} archivos) en {}", spec.hf_repo, spec.files.len(), dir.display());
    for (remote, local) in spec.files {
        let dest = dir.join(local);
        if dest.exists() && std::fs::metadata(&dest)?.len() > 0 {
            println!("  ✓ {local} (ya está)");
            continue;
        }
        let url = format!("https://huggingface.co/{}/resolve/main/{remote}", spec.hf_repo);
        print!("  ↓ {local} ... ");
        std::io::stdout().flush().ok();
        download_to(&url, &dest).await.with_context(|| format!("descargando {local}"))?;
        let mb = std::fs::metadata(&dest)?.len() / 1_048_576;
        println!("{mb} MB");
    }
    println!("  listo. cuba-memorys lo encuentra en {} (o {}=<ruta>)", dir.display(), spec.env_hint);
    Ok(())
}

/// The onnxruntime archive name and the library inside it, for this platform.
fn runtime_target() -> Result<(String, &'static str, &'static str)> {
    let v = ORT_VERSION;
    // (archive filename, extension, library filename inside lib/)
    let (archive, ext, lib) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => (format!("onnxruntime-linux-x64-{v}"), "tgz", "libonnxruntime.so"),
        ("linux", "aarch64") => {
            (format!("onnxruntime-linux-aarch64-{v}"), "tgz", "libonnxruntime.so")
        }
        ("macos", "x86_64") => (format!("onnxruntime-osx-x86_64-{v}"), "tgz", "libonnxruntime.dylib"),
        ("macos", "aarch64") => (format!("onnxruntime-osx-arm64-{v}"), "tgz", "libonnxruntime.dylib"),
        ("windows", "x86_64") => (format!("onnxruntime-win-x64-{v}"), "zip", "onnxruntime.dll"),
        (os, arch) => bail!(
            "no tengo el runtime prearmado para {os}/{arch}. Instalá onnxruntime {v} a mano \
             y apuntá ORT_DYLIB_PATH a la librería."
        ),
    };
    // leaked into 'static via the caller? No — return owned archive, borrowed consts.
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
    println!("→ runtime ({}/{}) desde {url}", std::env::consts::OS, std::env::consts::ARCH);

    let archive_path = dir.join(format!("{archive_stem}.{ext}"));
    print!("  ↓ descargando ... ");
    std::io::stdout().flush().ok();
    download_to(&url, &archive_path).await.context("descargando el runtime")?;
    println!("{} MB", std::fs::metadata(&archive_path)?.len() / 1_048_576);

    print!("  ⇢ extrayendo {lib_name} ... ");
    std::io::stdout().flush().ok();
    extract_library(&archive_path, ext, lib_name, &lib_dest)
        .with_context(|| format!("extrayendo {lib_name} de {}", archive_path.display()))?;
    std::fs::remove_file(&archive_path).ok();
    println!("ok");
    println!("  listo. cuba-memorys lo encuentra en {} (o ORT_DYLIB_PATH=<ruta>)", lib_dest.display());
    Ok(())
}

/// Pull the single shared library out of the runtime archive, wherever `lib/` sits.
fn extract_library(archive: &Path, ext: &str, lib_name: &str, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension("part");
    match ext {
        "tgz" => {
            // onnxruntime ships `lib/libonnxruntime.so` as a SYMLINK to the real
            // `libonnxruntime.so.1.21.0`. A symlink entry in a tar has zero content, so
            // matching the exact name and copying gives an empty file — which is what it
            // did, and an empty .so is a runtime that loads as nothing. So we take the
            // largest REGULAR entry whose name starts with the base lib name (the real
            // versioned object), and write it under the unversioned name the loader wants.
            let f = std::fs::File::open(archive)?;
            let gz = flate2::read::GzDecoder::new(f);
            let mut tar = tar::Archive::new(gz);
            let mut best: Option<(u64, PathBuf)> = None;
            let scratch = dest.with_file_name("cuba-ort-extract");
            std::fs::create_dir_all(&scratch)?;
            for entry in tar.entries()? {
                let mut entry = entry?;
                if entry.header().entry_type() != tar::EntryType::Regular {
                    continue; // skip the symlink and directories
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

/// Download a URL to a file: streamed, atomic, and size-checked. Shared by models and
/// runtime so both fail the same honest way on a short read.
async fn download_to(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("cuba-memorys/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let resp = client.get(url).send().await?.error_for_status()?;
    let expected = resp.content_length();

    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp).with_context(|| format!("creando {}", tmp.display()))?;
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
