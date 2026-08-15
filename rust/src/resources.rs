use std::path::{Path, PathBuf};

const MIB: u64 = 1024 * 1024;

const HEADROOM_MB: u64 = 768;
const BASE_MB: u64 = 220;
const EMBEDDER_MB: u64 = 900;
const RERANKER_MB: u64 = 1100;
const NLI_MB: u64 = 1100;
const RERANKER_VRAM_MB: u64 = 1600;
const RERANKER_VRAM_WIDE_MB: u64 = 3000;
const GPU_VRAM_RESERVE_MB: u64 = 512;

const MAX_EMBED_INTRA_THREADS: usize = 4;
const MAX_RERANK_INTRA_THREADS: usize = 8;
const MAX_NLI_INTRA_THREADS: usize = 4;
const GPU_RERANK_INTRA_THREADS: usize = 2;
const MAX_RERANK_CHUNK: usize = 16;
const NARROW_GPU_RERANK_CHUNK: usize = 8;
const CPU_RERANK_CHUNK: usize = 4;
const MAX_WORKER_THREADS: usize = 4;
const MIN_WORKER_THREADS: usize = 2;
const MAX_BLOCKING_THREADS: usize = 16;
const MIN_BLOCKING_THREADS: usize = 2;
const MAX_DB_CONNECTIONS: u32 = 4;
const MIN_DB_CONNECTIONS: u32 = 2;
const MAX_OOD_FIT_LIMIT: i64 = 5000;
const MIN_OOD_FIT_LIMIT: i64 = 500;
const DEFAULT_GPU_MEM_LIMIT_MB: u64 = 2048;

const OOD_BYTES_PER_SAMPLE: u64 = 1024 * 8 * 2;
const OOD_BUDGET_DIVISOR: u64 = 20;

const DISABLED_MODEL_DIR: &str = "cuba-memorys-disabled-by-resource-plan";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Machine {
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub swap_total_mb: u64,
    pub cgroup_limit_mb: Option<u64>,
    pub cores_logical: usize,
    pub cores_physical: usize,
    pub vram_free_mb: Option<u64>,
}

impl Machine {
    pub fn budget_mb(&self) -> u64 {
        let ceiling = match self.cgroup_limit_mb {
            Some(limit) => limit.min(self.ram_available_mb),
            None => self.ram_available_mb,
        };
        ceiling.saturating_sub(HEADROOM_MB)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Minimal,
    Lean,
    Standard,
    Full,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Minimal => "minimal",
            Tier::Lean => "lean",
            Tier::Standard => "standard",
            Tier::Full => "full",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Tier::Minimal => {
                "sin modelos: embeddings por hash, búsqueda léxica (BM25 + full-text + trigram)"
            }
            Tier::Lean => "embebedor semántico; sin reranker ni NLI",
            Tier::Standard => "embebedor semántico + un modelo de apoyo",
            Tier::Full => "embebedor + reranker + NLI",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub tier: Tier,
    pub embedder: bool,
    pub reranker: bool,
    pub reranker_on_gpu: bool,
    pub nli: bool,
    pub embed_intra_threads: usize,
    pub rerank_intra_threads: usize,
    pub nli_intra_threads: usize,
    pub rerank_chunk: usize,
    pub gpu_mem_limit_mb: Option<u64>,
    pub worker_threads: usize,
    pub max_blocking_threads: usize,
    pub db_max_connections: u32,
    pub ood_fit_limit: i64,
    pub budget_mb: u64,
    pub committed_mb: u64,
}

impl Plan {
    pub fn describe(&self) -> String {
        format!(
            "nivel={} ({}) · presupuesto={} MiB · reserva={} MiB · modelos: embebedor={} \
             reranker={}{} nli={} · hilos: embed={} rerank={} nli={} · tokio: workers={} \
             blocking={} · pool={} · chunk={} · ood_fit={}",
            self.tier.as_str(),
            self.tier.describe(),
            self.budget_mb,
            self.committed_mb,
            self.embedder,
            self.reranker,
            if self.reranker_on_gpu { " (GPU)" } else { "" },
            self.nli,
            self.embed_intra_threads,
            self.rerank_intra_threads,
            self.nli_intra_threads,
            self.worker_threads,
            self.max_blocking_threads,
            self.db_max_connections,
            self.rerank_chunk,
            self.ood_fit_limit,
        )
    }
}

pub fn plan(m: &Machine) -> Plan {
    let budget_mb = m.budget_mb();
    let vram_free_mb = m.vram_free_mb.unwrap_or(0);

    let embedder = budget_mb >= BASE_MB + EMBEDDER_MB;
    let mut committed_mb = BASE_MB + if embedder { EMBEDDER_MB } else { 0 };

    let reranker_on_gpu = vram_free_mb >= RERANKER_VRAM_MB;
    let reranker = embedder && budget_mb >= committed_mb + RERANKER_MB;
    if reranker {
        committed_mb += RERANKER_MB;
    }

    let nli = embedder && budget_mb >= committed_mb + NLI_MB;
    if nli {
        committed_mb += NLI_MB;
    }

    let tier = match [embedder, reranker, nli].iter().filter(|on| **on).count() {
        0 => Tier::Minimal,
        1 => Tier::Lean,
        2 => Tier::Standard,
        _ => Tier::Full,
    };

    let half_cores = (m.cores_logical / 2).max(1);
    let rerank_intra_threads = if reranker_on_gpu {
        GPU_RERANK_INTRA_THREADS
    } else {
        half_cores.min(MAX_RERANK_INTRA_THREADS)
    };

    let rerank_chunk = if !reranker {
        MAX_RERANK_CHUNK
    } else if !reranker_on_gpu {
        CPU_RERANK_CHUNK
    } else if vram_free_mb >= RERANKER_VRAM_WIDE_MB {
        MAX_RERANK_CHUNK
    } else {
        NARROW_GPU_RERANK_CHUNK
    };

    let gpu_mem_limit_mb = (reranker && reranker_on_gpu)
        .then(|| DEFAULT_GPU_MEM_LIMIT_MB.min(vram_free_mb.saturating_sub(GPU_VRAM_RESERVE_MB)));

    let ood_fit_limit = ((budget_mb * MIB / OOD_BUDGET_DIVISOR) / OOD_BYTES_PER_SAMPLE)
        .min(MAX_OOD_FIT_LIMIT as u64) as i64;

    Plan {
        tier,
        embedder,
        reranker,
        reranker_on_gpu,
        nli,
        embed_intra_threads: half_cores.min(MAX_EMBED_INTRA_THREADS),
        rerank_intra_threads,
        nli_intra_threads: half_cores.min(MAX_NLI_INTRA_THREADS),
        rerank_chunk,
        gpu_mem_limit_mb,
        worker_threads: m
            .cores_logical
            .clamp(MIN_WORKER_THREADS, MAX_WORKER_THREADS),
        max_blocking_threads: (m.cores_logical * 2)
            .clamp(MIN_BLOCKING_THREADS, MAX_BLOCKING_THREADS),
        db_max_connections: (m.cores_logical as u32).clamp(MIN_DB_CONNECTIONS, MAX_DB_CONNECTIONS),
        ood_fit_limit: ood_fit_limit.max(MIN_OOD_FIT_LIMIT),
        budget_mb,
        committed_mb,
    }
}

fn read_cgroup_bytes(path: &Path) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    let raw = raw.trim();
    if raw == "max" {
        return None;
    }
    raw.parse().ok()
}

fn tighten(best: &mut Option<u64>, dir: &Path) {
    for file in ["memory.max", "memory.high"] {
        if let Some(value) = read_cgroup_bytes(&dir.join(file)) {
            *best = Some(best.map_or(value, |current: u64| current.min(value)));
        }
    }
}

fn cgroup_limit_mb() -> Option<u64> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = cgroup.lines().find_map(|l| l.strip_prefix("0::"))?.trim();

    let mut dir = PathBuf::from("/sys/fs/cgroup");
    let mut best = None;
    tighten(&mut best, &dir);
    for segment in relative.split('/').filter(|s| !s.is_empty()) {
        dir.push(segment);
        tighten(&mut best, &dir);
    }
    best.map(|bytes| bytes / MIB)
}

fn vram_free_mb() -> Option<u64> {
    if !cfg!(any(feature = "cuda", feature = "directml")) {
        return None;
    }
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

pub fn probe() -> Machine {
    let mut system = sysinfo::System::new();
    system.refresh_memory();

    let cores_logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let cores_physical = sysinfo::System::physical_core_count()
        .unwrap_or(cores_logical.div_ceil(2))
        .max(1);

    Machine {
        ram_total_mb: system.total_memory() / MIB,
        ram_available_mb: system.available_memory() / MIB,
        swap_total_mb: system.total_swap() / MIB,
        cgroup_limit_mb: cgroup_limit_mb(),
        cores_logical,
        cores_physical,
        vram_free_mb: vram_free_mb(),
    }
}

fn set_if_absent(key: &str, value: &str) {
    if std::env::var_os(key).is_some() {
        return;
    }
    unsafe { std::env::set_var(key, value) }
}

pub fn disabled_model_path() -> String {
    std::env::temp_dir()
        .join(DISABLED_MODEL_DIR)
        .to_string_lossy()
        .into_owned()
}

pub fn apply(p: &Plan) {
    set_if_absent(
        "CUBA_EMBED_INTRA_THREADS",
        &p.embed_intra_threads.to_string(),
    );
    set_if_absent(
        "CUBA_RERANK_INTRA_THREADS",
        &p.rerank_intra_threads.to_string(),
    );
    set_if_absent("CUBA_NLI_INTRA_THREADS", &p.nli_intra_threads.to_string());
    set_if_absent("CUBA_RERANK_CHUNK", &p.rerank_chunk.to_string());
    set_if_absent("CUBA_DB_MAX_CONNECTIONS", &p.db_max_connections.to_string());
    set_if_absent("CUBA_OOD_FIT_LIMIT", &p.ood_fit_limit.to_string());

    if let Some(limit) = p.gpu_mem_limit_mb {
        set_if_absent("CUBA_GPU_MEM_LIMIT_MB", &limit.to_string());
    }
    if !p.reranker {
        set_if_absent("CUBA_RERANKER_PATH", &disabled_model_path());
    }
    if !p.nli {
        set_if_absent("CUBA_NLI_PATH", &disabled_model_path());
    }
}

pub fn ood_fit_limit() -> i64 {
    std::env::var("CUBA_OOD_FIT_LIMIT")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(MAX_OOD_FIT_LIMIT)
}

pub fn db_max_connections() -> u32 {
    std::env::var("CUBA_DB_MAX_CONNECTIONS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(MAX_DB_CONNECTIONS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laptop_4gb_no_gpu() -> Machine {
        Machine {
            ram_total_mb: 3900,
            ram_available_mb: 2800,
            swap_total_mb: 2048,
            cgroup_limit_mb: None,
            cores_logical: 4,
            cores_physical: 2,
            vram_free_mb: None,
        }
    }

    fn container_512mb() -> Machine {
        Machine {
            ram_total_mb: 16000,
            ram_available_mb: 12000,
            swap_total_mb: 0,
            cgroup_limit_mb: Some(512),
            cores_logical: 8,
            cores_physical: 4,
            vram_free_mb: None,
        }
    }

    fn desktop_16gb_with_gpu() -> Machine {
        Machine {
            ram_total_mb: 15900,
            ram_available_mb: 7160,
            swap_total_mb: 16384,
            cgroup_limit_mb: Some(4500),
            cores_logical: 12,
            cores_physical: 6,
            vram_free_mb: Some(5900),
        }
    }

    fn server_64gb_no_gpu() -> Machine {
        Machine {
            ram_total_mb: 65536,
            ram_available_mb: 60000,
            swap_total_mb: 0,
            cgroup_limit_mb: None,
            cores_logical: 32,
            cores_physical: 16,
            vram_free_mb: None,
        }
    }

    fn workstation_8gb_narrow_gpu() -> Machine {
        Machine {
            ram_total_mb: 8000,
            ram_available_mb: 6200,
            swap_total_mb: 4096,
            cgroup_limit_mb: None,
            cores_logical: 8,
            cores_physical: 4,
            vram_free_mb: Some(2100),
        }
    }

    fn every_machine() -> Vec<(&'static str, Machine)> {
        vec![
            ("laptop 4 GB sin GPU", laptop_4gb_no_gpu()),
            ("contenedor de 512 MB", container_512mb()),
            ("escritorio 16 GB con GPU", desktop_16gb_with_gpu()),
            ("servidor 64 GB sin GPU", server_64gb_no_gpu()),
            (
                "estación 8 GB con GPU estrecha",
                workstation_8gb_narrow_gpu(),
            ),
        ]
    }

    #[test]
    fn a_512mb_container_loads_no_model_at_all() {
        let p = plan(&container_512mb());

        assert_eq!(p.tier, Tier::Minimal, "{}", p.describe());
        assert!(!p.embedder, "2,6 GiB de pesos no caben en 512 MB");
        assert!(!p.reranker);
        assert!(!p.nli);
        assert_eq!(p.budget_mb, 0, "512 MB menos la reserva no deja nada");
    }

    #[test]
    fn the_cgroup_limit_wins_over_what_proc_meminfo_reports() {
        let m = container_512mb();

        assert!(
            m.ram_available_mb > 10_000,
            "el sistema dice que hay memoria de sobra"
        );
        assert_eq!(
            m.budget_mb(),
            0,
            "el presupuesto lo fija el cgroup, no /proc/meminfo: creerle al sistema haría \
             cargar 2,6 GiB de pesos dentro de un límite de 512 MB y el kernel mataría el proceso"
        );
    }

    #[test]
    fn a_4gb_laptop_without_gpu_still_gets_semantic_search() {
        let p = plan(&laptop_4gb_no_gpu());

        assert!(
            p.embedder,
            "el objetivo es que funcione, más lento, no que caiga a búsqueda léxica: {}",
            p.describe()
        );
        assert!(
            !p.reranker,
            "el cross-encoder en 2 núcleos físicos gasta su presupuesto y tira las puntuaciones"
        );
        assert!(!p.nli, "no caben 1,1 GiB más: {}", p.describe());
        assert_eq!(p.tier, Tier::Lean);
    }

    #[test]
    fn a_machine_with_room_for_everything_reproduces_todays_hardcoded_values() {
        let p = plan(&desktop_16gb_with_gpu());

        assert_eq!(p.tier, Tier::Full, "{}", p.describe());
        assert_eq!(p.embed_intra_threads, 4);
        assert_eq!(p.rerank_intra_threads, GPU_RERANK_INTRA_THREADS);
        assert_eq!(p.nli_intra_threads, 4);
        assert_eq!(p.rerank_chunk, MAX_RERANK_CHUNK);
        assert_eq!(p.gpu_mem_limit_mb, Some(DEFAULT_GPU_MEM_LIMIT_MB));
        assert_eq!(p.worker_threads, MAX_WORKER_THREADS);
        assert_eq!(p.db_max_connections, MAX_DB_CONNECTIONS);
        assert_eq!(p.ood_fit_limit, MAX_OOD_FIT_LIMIT);
    }

    #[test]
    fn the_plan_never_asks_for_more_than_the_daemon_already_took() {
        for (name, m) in every_machine() {
            let p = plan(&m);
            assert!(p.embed_intra_threads <= MAX_EMBED_INTRA_THREADS, "{name}");
            assert!(p.rerank_intra_threads <= MAX_RERANK_INTRA_THREADS, "{name}");
            assert!(p.nli_intra_threads <= MAX_NLI_INTRA_THREADS, "{name}");
            assert!(p.rerank_chunk <= MAX_RERANK_CHUNK, "{name}");
            assert!(p.worker_threads <= MAX_WORKER_THREADS, "{name}");
            assert!(p.max_blocking_threads <= MAX_BLOCKING_THREADS, "{name}");
            assert!(p.db_max_connections <= MAX_DB_CONNECTIONS, "{name}");
            assert!(p.ood_fit_limit <= MAX_OOD_FIT_LIMIT, "{name}");
            assert!(
                p.gpu_mem_limit_mb.unwrap_or(0) <= DEFAULT_GPU_MEM_LIMIT_MB,
                "{name}"
            );
        }
    }

    #[test]
    fn nothing_the_plan_hands_out_can_be_zero() {
        for (name, m) in every_machine() {
            let p = plan(&m);
            assert!(p.embed_intra_threads >= 1, "{name}");
            assert!(p.rerank_intra_threads >= 1, "{name}");
            assert!(p.nli_intra_threads >= 1, "{name}");
            assert!(p.rerank_chunk >= 1, "{name}");
            assert!(p.worker_threads >= 1, "{name}");
            assert!(p.max_blocking_threads >= 1, "{name}");
            assert!(p.db_max_connections >= 1, "{name}");
            assert!(p.ood_fit_limit >= MIN_OOD_FIT_LIMIT, "{name}");
        }
    }

    #[test]
    fn what_the_plan_loads_always_fits_the_budget_it_measured() {
        for (name, m) in every_machine() {
            let p = plan(&m);
            assert!(
                p.committed_mb <= p.budget_mb.max(BASE_MB),
                "{name} compromete {} MiB con un presupuesto de {} MiB",
                p.committed_mb,
                p.budget_mb
            );
        }
    }

    #[test]
    fn only_memory_decides_the_reranker_never_a_guess_about_cpu_speed() {
        let mut roomy_but_narrow = server_64gb_no_gpu();
        roomy_but_narrow.cores_physical = 4;

        let p = plan(&roomy_but_narrow);

        assert!(
            p.reranker,
            "el reranker mide +93% nDCG y su lentitud en CPU ya la corta CUBA_RERANK_TIMEOUT_SECS; \
             apagarlo por un umbral de núcleos sin medir sería cambiar comportamiento a ciegas: {}",
            p.describe()
        );
        assert_eq!(p.tier, Tier::Full);

        let laptop = plan(&laptop_4gb_no_gpu());
        assert!(
            !laptop.reranker,
            "en el portátil lo excluye el presupuesto, que sí está medido: 1,1 GiB más sobre \
             los 1,12 no entran en {} MiB",
            laptop.budget_mb
        );
    }

    #[test]
    fn a_narrow_gpu_keeps_the_reranker_but_halves_its_batch() {
        let p = plan(&workstation_8gb_narrow_gpu());

        assert!(p.reranker_on_gpu, "{}", p.describe());
        assert_eq!(
            p.rerank_chunk, NARROW_GPU_RERANK_CHUNK,
            "las activaciones escalan con chunk × tokens; con 2,1 GiB libres la tanda entera \
             no entra y partirla es lo que evita el fallo de asignación"
        );
        assert!(
            p.gpu_mem_limit_mb.unwrap() < DEFAULT_GPU_MEM_LIMIT_MB,
            "la arena no puede pedir más VRAM de la que hay libre"
        );
    }

    #[test]
    fn the_reranker_is_preferred_over_nli_when_only_one_fits() {
        let mut tight = desktop_16gb_with_gpu();
        tight.cgroup_limit_mb = Some(3400);

        let p = plan(&tight);

        assert!(
            p.reranker,
            "medido +93% nDCG: si solo cabe un modelo de apoyo, es este: {}",
            p.describe()
        );
        assert!(!p.nli, "{}", p.describe());
        assert_eq!(p.tier, Tier::Standard);
    }

    #[test]
    fn the_ood_fit_shrinks_only_where_its_80mb_actually_hurt() {
        let squeezed = plan(&container_512mb()).ood_fit_limit;
        let laptop = plan(&laptop_4gb_no_gpu()).ood_fit_limit;
        let roomy = plan(&desktop_16gb_with_gpu()).ood_fit_limit;

        assert!(
            squeezed < roomy,
            "la muestra retiene n×1024 f64 dos veces: 80 MiB dentro de un límite de 512 MB \
             es la sexta parte del contenedor y hay que recortarla"
        );
        assert_eq!(squeezed, MIN_OOD_FIT_LIMIT);
        assert_eq!(
            laptop, MAX_OOD_FIT_LIMIT,
            "esos mismos 80 MiB son el 4% de un presupuesto de 2 GiB: recortar aquí sería \
             perder precisión de abstención sin ganar nada, y el regulador no está para eso"
        );
        assert_eq!(roomy, MAX_OOD_FIT_LIMIT);
    }

    #[test]
    fn an_explicit_environment_variable_beats_the_plan() {
        let key = "CUBA_RESOURCES_TEST_KNOB";
        unsafe { std::env::set_var(key, "explicit") };

        set_if_absent(key, "from-plan");

        assert_eq!(
            std::env::var(key).as_deref(),
            Ok("explicit"),
            "un valor puesto a mano gana siempre: el regulador rellena huecos, no pisa decisiones"
        );
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn disabling_a_model_points_it_at_a_path_that_cannot_resolve() {
        let path = PathBuf::from(disabled_model_path());

        assert!(
            !path.join("model.onnx").exists(),
            "apagar un modelo reutiliza la degradación ya probada — ruta que no resuelve — \
             en vez de inventar un modo nuevo"
        );
        assert!(!path.exists());
    }

    #[test]
    fn a_cgroup_file_saying_max_is_not_a_limit() {
        let root = std::env::temp_dir().join("cuba-memorys-cgroup-probe");
        std::fs::remove_dir_all(&root).ok();

        let capped = root.join("capped");
        std::fs::create_dir_all(&capped).unwrap();
        std::fs::write(capped.join("memory.max"), "max\n").unwrap();
        std::fs::write(capped.join("memory.high"), "4718592000\n").unwrap();

        let mut best = None;
        tighten(&mut best, &capped);
        assert_eq!(
            best,
            Some(4_718_592_000),
            "memory.high es el que hace que el kernel empiece a reclamar, y reclamar aquí \
             significa paginar a swap: cuenta como límite aunque memory.max diga `max`"
        );

        let unlimited = root.join("unlimited");
        std::fs::create_dir_all(&unlimited).unwrap();
        std::fs::write(unlimited.join("memory.max"), "max\n").unwrap();
        std::fs::write(unlimited.join("memory.high"), "max\n").unwrap();

        let mut none = None;
        tighten(&mut none, &unlimited);
        assert_eq!(
            none, None,
            "sin límite el presupuesto lo tiene que fijar la memoria disponible del sistema; \
             leer `max` como un número deja un techo que no existe"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn probing_this_machine_yields_a_usable_plan() {
        let m = probe();

        assert!(m.ram_total_mb > 0, "{m:?}");
        assert!(m.cores_logical >= 1, "{m:?}");
        assert!(m.cores_physical >= 1, "{m:?}");
        assert!(
            m.cores_physical <= m.cores_logical,
            "no puede haber más núcleos físicos que lógicos: {m:?}"
        );

        let p = plan(&m);
        assert!(p.worker_threads >= 1, "{}", p.describe());
    }
}
