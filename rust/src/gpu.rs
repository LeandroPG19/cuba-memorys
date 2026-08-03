use anyhow::Result;
use ort::session::builder::SessionBuilder;
#[cfg(any(feature = "cuda", feature = "directml"))]
use std::path::PathBuf;

/// Which model is asking for a device.
///
/// Placement is decided per model, not once per process, because the three
/// models cannot share the same hardware profitably:
///
/// * **Embedder** — INT8 dynamically quantised: 96 `DynamicQuantizeLinear`
///   feeding 144 `MatMulInteger`. The CUDA provider registers no kernel for
///   either, so ONNX Runtime partitions those nodes onto the CPU regardless.
///   Registering CUDA for it only reserved an arena the model never computed in
///   (measured: 374 MiB of VRAM held while all 544 MB of weights sat in host
///   RAM) and paid a host↔device copy at every partition boundary.
/// * **Reranker** — genuine FP16, and the heaviest of the three (24 layers over
///   up to 50 candidates). This is the one the GPU actually accelerates.
/// * **NLI** — FP32, and stuck there: mDeBERTa is documented upstream as not
///   supporting FP16, and this project already measured the INT8 build handing
///   back confident false entailments. It is also invoked rarely and tolerates
///   latency, so CPU costs ~150-400 ms per verdict and buys back over a
///   gigabyte of VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    Embedder,
    Reranker,
    Nli,
}

impl Workload {
    /// The environment variable that overrides this model's placement, so a
    /// placement can be A/B tested without a rebuild.
    fn device_var(self) -> &'static str {
        match self {
            Self::Embedder => "CUBA_EMBED_DEVICE",
            Self::Reranker => "CUBA_RERANK_DEVICE",
            Self::Nli => "CUBA_NLI_DEVICE",
        }
    }

    /// Only the reranker gains anything from the GPU — see the type docs.
    fn gpu_by_default(self) -> bool {
        matches!(self, Self::Reranker)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Embedder => "embedder",
            Self::Reranker => "reranker",
            Self::Nli => "nli",
        }
    }
}

/// Whether this model runs on the GPU. Accepts `gpu`/`cuda` or `cpu`.
pub fn wants_gpu(workload: Workload) -> bool {
    if !cfg!(any(feature = "cuda", feature = "directml")) {
        return false;
    }
    match std::env::var(workload.device_var()) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "gpu" | "cuda" | "directml" => true,
            "cpu" => false,
            other => {
                tracing::warn!(
                    var = workload.device_var(),
                    value = other,
                    default_gpu = workload.gpu_by_default(),
                    "dispositivo desconocido — uso el default del modelo"
                );
                workload.gpu_by_default()
            }
        },
        Err(_) => workload.gpu_by_default(),
    }
}

pub fn configure(builder: SessionBuilder, workload: Workload) -> Result<SessionBuilder> {
    if !wants_gpu(workload) {
        return configure_cpu(builder, workload);
    }

    let providers: Vec<ort::ep::ExecutionProviderDispatch> = [
        #[cfg(feature = "cuda")]
        cuda_provider(),
        #[cfg(feature = "directml")]
        ort::ep::DirectML::default().build(),
    ]
    .into_iter()
    .collect();

    // Compiled without a GPU feature: `wants_gpu` already returned false, so
    // this is only reachable if that ever changes. Fall back rather than
    // silently hand back a builder with no provider registered.
    if providers.is_empty() {
        return configure_cpu(builder, workload);
    }

    tracing::info!(model = workload.label(), "sesión ONNX en GPU");
    builder
        .with_execution_providers(providers)
        .map_err(|e| anyhow::anyhow!("registrando execution providers GPU: {e}"))
}

/// The CPU provider is the implicit fallback, so registering it explicitly is
/// only how `ort` exposes the arena switch. The embedder answers every search
/// and keeps its arena to reuse activation buffers between queries; the NLI
/// fires occasionally, and dropping its arena returns that memory between
/// verdicts instead of holding it for the life of the daemon.
fn configure_cpu(builder: SessionBuilder, workload: Workload) -> Result<SessionBuilder> {
    let use_arena = !matches!(workload, Workload::Nli);
    tracing::info!(
        model = workload.label(),
        arena = use_arena,
        "sesión ONNX en CPU"
    );
    builder
        .with_execution_providers([ort::ep::CPU::default()
            .with_arena_allocator(use_arena)
            .build()])
        .map_err(|e| anyhow::anyhow!("registrando CPU execution provider: {e}"))
}

/// `CUDA::default()` leaves the arena on `ArenaExtendStrategy::NextPowerOfTwo`,
/// which doubles its reservation on every growth instead of taking the size a
/// session actually asked for — that is how ~1.65 GB of model weights ballooned
/// into 5+ GB of VRAM on a 6 GB card. `SameAsRequested` plus an explicit cap
/// makes the footprint match the real working set.
///
/// The cap is per session, not global (`gpu_mem_limit` in ONNX Runtime terms),
/// which is survivable only because exactly one workload asks for CUDA — see
/// `Workload`. Sending a second model to the GPU means budgeting both against
/// the card, because two sessions would each take this much.
#[cfg(feature = "cuda")]
fn cuda_provider() -> ort::ep::ExecutionProviderDispatch {
    let limit_mb: usize = std::env::var("CUBA_GPU_MEM_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);

    ort::ep::CUDA::default()
        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
        .with_memory_limit(limit_mb * 1024 * 1024)
        .build()
}

pub struct GpuStatus {
    pub degraded: bool,
    pub detail: String,
    pub hint: Option<String>,
}

/// Where each model actually runs, for `doctor` — a GPU being present says
/// nothing about which sessions use it.
pub fn placement_summary() -> String {
    [Workload::Embedder, Workload::Reranker, Workload::Nli]
        .iter()
        .map(|&w| {
            let dev = if wants_gpu(w) { "gpu" } else { "cpu" };
            format!("{}={dev}", w.label())
        })
        .collect::<Vec<_>>()
        .join(" ")
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
                    "{provider} — runtime GPU y GPU detectados · colocación: {}",
                    placement_summary()
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
        if nvidia_driver_present() {
            return GpuStatus {
                degraded: true,
                detail: "hay una GPU NVIDIA en esta máquina pero el binario se compiló sin \
                         soporte — el embebedor corre en CPU y cada búsqueda cuesta ~9x más"
                    .to_string(),
                hint: Some("./scripts/build-gpu.sh".to_string()),
            };
        }
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

#[cfg_attr(any(feature = "cuda", feature = "directml"), allow(dead_code))]
fn nvidia_driver_present() -> bool {
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
