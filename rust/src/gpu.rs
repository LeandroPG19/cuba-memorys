use anyhow::Result;
use ort::session::builder::SessionBuilder;
#[cfg(any(feature = "cuda", feature = "directml"))]
use std::path::PathBuf;

pub fn configure(builder: SessionBuilder) -> Result<SessionBuilder> {
    let providers: Vec<ort::ep::ExecutionProviderDispatch> = [
        #[cfg(feature = "cuda")]
        ort::ep::CUDA::default().build(),
        #[cfg(feature = "directml")]
        ort::ep::DirectML::default().build(),
    ]
    .into_iter()
    .collect();

    if providers.is_empty() {
        return Ok(builder);
    }

    builder
        .with_execution_providers(providers)
        .map_err(|e| anyhow::anyhow!("registrando execution providers GPU: {e}"))
}

pub struct GpuStatus {
    pub degraded: bool,
    pub detail: String,
    pub hint: Option<String>,
}

pub fn status() -> GpuStatus {
    #[cfg(any(feature = "cuda", feature = "directml"))]
    {
        let provider = if cfg!(feature = "cuda") {
            "cuda"
        } else {
            "directml"
        };
        let runtime_gpu = runtime_has_gpu_provider(provider);
        let gpu_device = provider != "cuda" || nvidia_present();

        if runtime_gpu && gpu_device {
            return GpuStatus {
                degraded: false,
                detail: format!(
                    "{provider} — runtime GPU y GPU detectados (confirmá el uso real con nvidia-smi durante una consulta)"
                ),
                hint: None,
            };
        }
        if !runtime_gpu {
            return GpuStatus {
                degraded: true,
                detail: format!(
                    "compilado con {provider}, pero el runtime instalado es el de CPU → corriendo en CPU"
                ),
                hint: Some("cuba-memorys models runtime --gpu".to_string()),
            };
        }
        GpuStatus {
            degraded: true,
            detail: format!(
                "compilado con {provider}, pero no detecté GPU NVIDIA → corriendo en CPU"
            ),
            hint: Some(
                "revisá el driver (nvidia-smi); sin GPU, esta build igual corre en CPU".to_string(),
            ),
        }
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "directml")))]
    {
        GpuStatus {
            degraded: false,
            detail: "cpu (compilado sin soporte GPU)".to_string(),
            hint: None,
        }
    }
}

pub fn active_provider() -> String {
    status().detail
}

#[cfg(any(feature = "cuda", feature = "directml"))]
fn runtime_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        return PathBuf::from(p).parent().map(|p| p.to_path_buf());
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    Some(PathBuf::from(home).join(".cache/cuba-memorys/onnxruntime"))
}

#[cfg(any(feature = "cuda", feature = "directml"))]
fn runtime_has_gpu_provider(provider: &str) -> bool {
    let Some(dir) = runtime_dir() else {
        return false;
    };
    let candidates: [&str; 4] = match provider {
        "cuda" => [
            "libonnxruntime_providers_cuda.so",
            "onnxruntime_providers_cuda.dll",
            "libonnxruntime_providers_cuda.dylib",
            "onnxruntime_providers_cuda.so",
        ],
        _ => [
            "onnxruntime_providers_dml.dll",
            "DirectML.dll",
            "libonnxruntime_providers_dml.so",
            "onnxruntime_providers_dml.so",
        ],
    };
    candidates.iter().any(|name| dir.join(name).exists())
}

#[cfg(feature = "cuda")]
fn nvidia_present() -> bool {
    if std::path::Path::new("/proc/driver/nvidia/version").exists() {
        return true;
    }
    let exe = if cfg!(windows) {
        "nvidia-smi.exe"
    } else {
        "nvidia-smi"
    };
    std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).any(|p| p.join(exe).exists()))
        .unwrap_or(false)
}
