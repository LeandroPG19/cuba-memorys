//! Natural language inference — does this evidence ENTAIL this claim?
//!
//! # Why this exists
//!
//! v0.11.2 fixed `cuba_faro mode=verify` by sending its evidence to an LLM judge, and
//! that worked: the false claim went from 0.61 confidence to `contradicted`. But it
//! costs a model round-trip — ~20 s through the `claude` CLI — and without a CLI or an
//! MCP client that supports sampling, verification degrades to `unknown`.
//!
//! Entailment is a *classification* problem, and a cross-encoder answers it in ~50 ms.
//! That is what this is: a local verdict, in Spanish, for free, offline.
//!
//! # Why NOT the model cuba-thinking used
//!
//! cuba-thinking has an NLI contradiction detector, and porting it was the original
//! plan. Two things killed that, and both were found by checking instead of assuming:
//!
//! 1. **It is BART-large-MNLI, which is English-only.** Of the 1,487 observations in
//!    the corpus this was built for, **1,108 are in Spanish** (75%). A verifier that
//!    silently fails on three quarters of the memories is worse than no verifier — it
//!    is the exact class of bug this project spent a week eliminating.
//! 2. **rust-bert pulls in libtorch: 8.1 GB.** cuba-memorys ships a 13 MB binary.
//!
//! So: **mDeBERTa-v3-base-xnli** (100 languages, 87.1% on XNLI) exported to ONNX, run
//! through the `ort` runtime this crate already links. No libtorch, and it reads
//! Spanish.
//!
//! # Three things measurement overturned
//!
//! Every one of these looked obviously right, and every one of them was wrong. They are
//! recorded because the next person to "improve" this module will reach for the same
//! three.
//!
//! **1. The quantized checkpoint (323 MB) is unusable.** Not merely less accurate —
//! wrong in the one direction that matters. It read evidence saying the reranker "is
//! disabled by default" as SUPPORTING the claim that it is *enabled*, at 0.62. The
//! fp32 export (1.1 GB) says `contradicts` at 0.995. DeBERTa-v3's disentangled
//! attention does not survive int8, and it bought nothing anyway: 48 ms per verdict
//! quantized against 53 ms at full precision. It was paying in accuracy for a speed-up
//! that does not exist on CPU.
//!
//! **2. Decomposing the premise into clauses makes it worse, not better.** See
//! [`MAX_TOKENS`]. It got three of five real cases wrong, and CONTRADICTED true claims.
//!
//! **3. An argmax over the three-way head is not a verdict.** See [`decide`]. The model
//! is capable of scoring `entailment` highest on a claim that is flatly false, and an
//! argmax publishes that as `supports`.
//!
//! # The lesson from the reranker, applied
//!
//! The cross-encoder in `search::rerank` never produced a single score for its entire
//! life, because it fed `token_type_ids` to a model that has none and read `f16`
//! logits as `f32` — and the caller swallowed the error. So this module does not
//! assume the model's interface. It **asks**: which inputs does the graph declare,
//! and what precision are the outputs. Both are discovered at load time.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::PathBuf;
use std::sync::OnceLock;

static NLI_SESSION: OnceLock<std::sync::Mutex<Session>> = OnceLock::new();
static NLI_TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();
static NLI_STATUS: OnceLock<bool> = OnceLock::new();
static NLI_SEMAPHORE: OnceLock<tokio::sync::Semaphore> = OnceLock::new();
/// Whether the loaded graph actually declares `token_type_ids`. Discovered, not
/// assumed — see module docs.
static WANTS_TYPE_IDS: OnceLock<bool> = OnceLock::new();

/// What the evidence says about the claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entailment {
    /// The evidence asserts the claim.
    Supports,
    /// The evidence asserts something incompatible with it.
    Contradicts,
    /// The evidence neither confirms nor denies it — INCLUDING when it is about the
    /// same topic and simply says nothing about what the claim asserts. Being
    /// on-topic is not support, and conflating the two is what broke verify.
    Neutral,
}

impl Entailment {
    /// The judge's vocabulary, so an NLI verdict is interchangeable with an LLM one.
    pub fn as_verdict(self) -> &'static str {
        match self {
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::Neutral => "unrelated",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NliVerdict {
    pub label: Entailment,
    /// Softmax probability of the winning label.
    pub confidence: f64,
    /// Did the model actually decide, or is `label` a shrug?
    ///
    /// The distinction only matters for `Neutral`, and there it matters a lot.
    /// `Neutral + decisive` means *the evidence says nothing about this claim* — a
    /// real finding. `Neutral + !decisive` means *the evidence is plainly about this
    /// claim and I cannot tell which way it cuts* — no finding at all, and callers
    /// must escalate instead of recording a verdict that was never reached.
    pub decisive: bool,
    /// The full distribution, `[entailment, neutral, contradiction]`. Kept because the
    /// winning label alone hides the difference between a decision and a coin-flip.
    pub probs: [f64; 3],
}

/// ORT threads for one NLI pass. Half the machine, never more than 4.
///
/// The evidence a claim is checked against is paragraphs, and attention is quadratic in
/// sequence length — the compute has to come from somewhere. Since the session is
/// serialized behind a Mutex anyway (see [`semaphore`]), the choice is between one pass
/// using the cores properly and several passes fighting over them.
///
/// Half, not all: cuba-memorys runs on the machine its user is working on.
fn intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 4))
        .unwrap_or(2)
}

fn semaphore() -> &'static tokio::sync::Semaphore {
    // ONE permit, because one is the truth.
    //
    // `Session::run` takes `&mut self`, so the session lives behind a Mutex and exactly
    // one pass executes at a time no matter what this number says. It said 2, and the
    // second permit bought nothing but a thread parked on a mutex — while each pass ran
    // on half the cores it could have used. Admitting the serialization and giving the
    // one real pass more threads is strictly better than pretending there are two.
    NLI_SEMAPHORE.get_or_init(|| tokio::sync::Semaphore::new(1))
}

/// Where the model lives. `CUBA_NLI_PATH`, or the cache directory the download
/// script writes to.
fn model_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CUBA_NLI_PATH") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home)
        .join(".cache")
        .join("cuba-memorys")
        .join("models-nli");
    p.exists().then_some(p)
}

/// Is there a model on disk to load? A filesystem check, nothing more.
///
/// Separate from [`enabled`] on purpose. `resolve_judge()` runs on paths that never
/// ask for entailment — `cuba_contradiccion`, dedupe — and making the *choice* of
/// judge pull 1.1 GB into RSS would tax every one of them for a model they will not
/// call. So selection asks this, and the model loads on first use.
pub fn available() -> bool {
    model_dir().is_some()
}

/// True when a local NLI model is loaded and usable. **Loads the model** on first
/// call — see [`available`] for the cheap check.
pub fn enabled() -> bool {
    *NLI_STATUS.get_or_init(|| match model_dir() {
        Some(dir) => match init(&dir) {
            Ok(()) => {
                tracing::info!(path = %dir.display(), "NLI (mDeBERTa-xnli) cargado — entailment local");
                true
            }
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "NLI no pudo cargarse — se usará el juez LLM");
                false
            }
        },
        None => false,
    })
}

fn init(dir: &std::path::Path) -> Result<()> {
    // Before touching `ort`: is the runtime even there? With `load-dynamic`, a
    // missing libonnxruntime.so does not raise — in v0.11.2 it HUNG the whole MCP
    // server, with no error and no timeout, because the client was still waiting on
    // a handshake that would never come. So we look first, and bail with a sentence
    // that says what to install.
    if crate::embeddings::onnx::locate_onnxruntime().is_none() {
        anyhow::bail!(
            "hay un modelo NLI en {dir:?} pero no encuentro libonnxruntime.so — \
             instalá onnxruntime o apuntá ORT_DYLIB_PATH a la librería"
        );
    }

    // Full precision FIRST, and the int8 export only as a last resort.
    //
    // The quantized checkpoint is not merely less accurate, it is wrong in the one
    // direction that matters. Measured on the same pairs: it read "the reranker …is
    // disabled by default" as SUPPORTING "the reranker is enabled by default" (0.62),
    // and it let "cuba-memorys is written in Java" pass as `unrelated` against
    // evidence saying Rust. The fp32 export answers both correctly at 0.995+.
    // DeBERTa-v3's disentangled attention does not survive int8, and a verifier that
    // confirms false claims is worse than no verifier.
    //
    // It costs nothing to fix: 53 ms per verdict at fp32 versus 48 ms quantized — int8
    // buys no speed on CPU here, it only pays in accuracy.
    let full = dir.join("model.onnx");
    let quantized = dir.join("model_quantized.onnx");
    let model_file = if full.exists() {
        full
    } else if quantized.exists() {
        tracing::warn!(
            path = %quantized.display(),
            "usando el NLI CUANTIZADO: da entailments falsas con confianza \
             (mide 0.62 de 'supports' para una contradicción evidente). Descargá \
             model.onnx (fp32) — cuesta lo mismo por veredicto"
        );
        quantized
    } else {
        anyhow::bail!("no hay model.onnx ni model_quantized.onnx en {dir:?}");
    };

    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
        .with_intra_threads(intra_threads())
        .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?
        .commit_from_file(&model_file)
        .map_err(|e| anyhow::anyhow!("cargando {model_file:?}: {e}"))?;

    // Ask the graph what it takes. The reranker assumed, and never ran.
    let wants = session
        .inputs()
        .iter()
        .any(|i| i.name() == "token_type_ids");
    let _ = WANTS_TYPE_IDS.set(wants);
    tracing::debug!(token_type_ids = wants, "NLI: inputs del grafo");

    NLI_SESSION
        .set(std::sync::Mutex::new(session))
        .map_err(|_| anyhow::anyhow!("sesión NLI ya inicializada"))?;

    let tok_path = dir.join("tokenizer.json");
    if !tok_path.exists() {
        anyhow::bail!("falta tokenizer.json en {dir:?}");
    }
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("truncation: {e}"))?;
    NLI_TOKENIZER
        .set(tokenizer)
        .map_err(|_| anyhow::anyhow!("tokenizer NLI ya inicializado"))?;

    Ok(())
}

/// Confirming a claim takes more evidence than doubting one, and the thresholds say
/// so out loud.
///
/// This is not a tuned constant, it is a cost asymmetry. cuba-memorys exists to stop a
/// model from believing things its memory never said; a false `supports` is that exact
/// failure, restored. A false `contradicts` merely sends a true claim back for a second
/// look. The two errors are not worth the same, so they do not cost the same.
///
/// The corpus agrees with the principle. Every genuine entailment measured scored ≥0.95
/// — "escrito en Rust" against evidence saying so, 0.998; "eating fruit" from "eating an
/// apple", 0.995. The one spurious entailment found, where the model read "the reranker
/// is a cross-encoder based on XLM-RoBERTa" as supporting "the reranker is a bi-encoder"
/// — jargon whose mutual exclusivity it has no way to know — scored 0.693. Real support
/// does not live down there.
///
/// This threshold is the single thing standing between that 0.693 and a confirmed false
/// claim, which is the bug this whole subsystem exists to kill.
const SUPPORT_FLOOR: f64 = 0.80;

/// Doubt is cheaper than confirmation, so it clears a lower bar. The weakest true
/// contradiction measured — port 5488 against port 5491 — scored 0.731.
const CONTRA_FLOOR: f64 = 0.60;

/// Above this, "the evidence says nothing about this claim" is itself a finding rather
/// than a shrug. "escrito en Rust" against a claim about paella scores 0.887 neutral —
/// that is not indecision, it is a clear report of irrelevance.
const NEUTRAL_FLOOR: f64 = 0.50;

/// The premise is scored WHOLE. It is not cut into clauses, and that is a decision
/// paid for in blood.
///
/// Decomposing looked obviously right: XNLI premises are single sentences, real
/// memories are paragraphs, so hand the model one sentence at a time and max-pool.
/// Measured against the corpus, it got **three of five cases wrong**, and wrong in the
/// worst direction — it CONTRADICTED true claims:
///
/// ```text
///   evidence: "cuba-memorys es un servidor MCP escrito en Rust. Usa PostgreSQL…"
///   claim:    "cuba-memorys está escrito en Rust"        (TRUE)
///     decomposed → contradicts 0.986   ✗
///     whole      → supports    0.998   ✓
/// ```
///
/// The reason is a mismatch between what NLI *is* and what fact-checking *needs*. NLI
/// is trained on scenes, where two predicates about one subject are alternatives: "a
/// man is playing guitar" genuinely contradicts "a man is playing piano". A knowledge
/// base is not a scene. "cuba-memorys uses PostgreSQL" and "cuba-memorys is written in
/// Rust" are both true, and the model — shown them in isolation — rates them a
/// contradiction at **0.993**. Every clause describing a *different attribute of the
/// same entity* becomes a vote against the claim. Decomposition did not sharpen the
/// signal; it manufactured conflict.
///
/// Given the whole paragraph, the model sees the premise simply covers more ground
/// than the claim, and says so. The dilution that decomposition was invented to fix is
/// handled instead by [`decide`], which reads the directed classes against each other
/// rather than taking an argmax — and that rule, it turns out, was always sufficient
/// on its own. Decomposition was solving a problem that had already been solved.
///
/// The tokenizer truncates at 512 tokens. A memory longer than that is read to the cut
/// and no further, and [`entails`] says so out loud rather than reporting "nothing
/// found" for a contradiction it never looked at.
const MAX_TOKENS: usize = 512;

/// Does `premise` (the stored evidence) entail `hypothesis` (the claim)?
///
/// The order matters and is not symmetric: NLI asks whether a reader of the premise
/// would accept the hypothesis. Evidence is the premise; the claim is the hypothesis.
pub async fn entails(premise: &str, hypothesis: &str) -> Result<NliVerdict> {
    if !enabled() {
        anyhow::bail!("no hay modelo NLI cargado");
    }
    let premise = premise.trim();
    if premise.is_empty() {
        anyhow::bail!("la evidencia está vacía: no hay nada que pueda apoyar ni contradecir");
    }
    let (p, h) = (premise.to_string(), hypothesis.to_string());

    let _permit = semaphore()
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("semáforo NLI cerrado"))?;

    tokio::task::spawn_blocking(move || {
        let (probs, truncated) = classify(&p, &h)?;
        if truncated {
            // Never a silent cut. A contradiction living past the 512th token comes
            // back as "no contradiction found", and the caller has to be able to tell
            // that apart from evidence we read to the end.
            tracing::warn!(
                chars = p.chars().count(),
                "evidencia recortada a {MAX_TOKENS} tokens: si contradice la afirmación                  más allá del corte, NO se detectará"
            );
        }
        Ok(decide(probs))
    })
    .await
    .context("la tarea NLI hizo panic")?
}

/// Turn the model's distribution into a verdict — or into an admission that there
/// isn't one.
///
/// This is *not* an argmax. Each class must clear its own floor, and the floors differ
/// on purpose: confirming a claim is held to a higher standard than doubting one,
/// because confirming a false memory is the failure this project exists to prevent and
/// doubting a true one merely costs a second look.
///
/// When nothing clears its floor, the model is telling us it does not know. It says so,
/// rather than handing back whichever number happened to come out largest — an argmax
/// over [0.69 · 0.10 · 0.21] would confidently report `supports` for a claim that is
/// flatly false.
fn decide(probs: [f64; 3]) -> NliVerdict {
    let (e, n, c) = (probs[0], probs[1], probs[2]);

    let (label, confidence, decisive) = if e >= SUPPORT_FLOOR && e > c {
        (Entailment::Supports, e, true)
    } else if c >= CONTRA_FLOOR && c > e {
        (Entailment::Contradicts, c, true)
    } else if n >= NEUTRAL_FLOOR && n > e && n > c {
        // Silent, and confidently so. "The evidence does not speak to this claim" is a
        // real finding, and it counts for NEITHER side.
        (Entailment::Neutral, n, true)
    } else {
        // No class earned it. Not a verdict — and callers must not treat it as one.
        (Entailment::Neutral, n, false)
    };

    NliVerdict {
        label,
        confidence,
        decisive,
        probs,
    }
}

/// One forward pass over the pair. Returns the raw distribution
/// `[entailment, neutral, contradiction]` and whether the premise hit the token limit.
/// Deciding what it *means* is [`decide`]'s job, and lives in exactly one place.
fn classify(premise: &str, hypothesis: &str) -> Result<([f64; 3], bool)> {
    let tokenizer = NLI_TOKENIZER
        .get()
        .context("tokenizer NLI no inicializado")?;
    let session_lock = NLI_SESSION.get().context("sesión NLI no inicializada")?;

    // A sentence PAIR: the model sees `premise [SEP] hypothesis` and judges the relation
    // between them. Encoding the two separately would ask a different question.
    let encoding = tokenizer
        .encode((premise, hypothesis), true)
        .map_err(|e| anyhow::anyhow!("tokenizando el par: {e}"))?;

    let ids_raw = encoding.get_ids();
    let truncated = ids_raw.len() >= MAX_TOKENS;

    let ids: Vec<i64> = ids_raw.iter().map(|&i| i as i64).collect();
    let mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();
    let types: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

    let shape = vec![1i64, ids.len() as i64];
    let ids_t = ort::value::Tensor::from_array((shape.clone(), ids)).context("tensor input_ids")?;
    let mask_t =
        ort::value::Tensor::from_array((shape.clone(), mask)).context("tensor attention_mask")?;

    let mut session = session_lock
        .lock()
        .map_err(|e| anyhow::anyhow!("lock NLI envenenado: {e}"))?;

    let wants_types = *WANTS_TYPE_IDS.get().unwrap_or(&false);
    let outputs = if wants_types {
        let types_t =
            ort::value::Tensor::from_array((shape, types)).context("tensor token_type_ids")?;
        session.run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
            "token_type_ids" => types_t,
        })
    } else {
        session.run(ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        })
    }
    .map_err(|e| anyhow::anyhow!("inferencia NLI: {e}"))?;

    if outputs.len() == 0 {
        anyhow::bail!("el modelo NLI no devolvió salidas");
    }

    // f32 or f16, depending on the export. The reranker only asked for f32 and threw
    // on every pair for its entire existence.
    let logits: Vec<f32> = match outputs[0].try_extract_tensor::<f32>() {
        Ok((_, d)) => d.to_vec(),
        Err(_) => {
            let (_, d) = outputs[0]
                .try_extract_tensor::<half::f16>()
                .map_err(|e| anyhow::anyhow!("extrayendo logits (ni f32 ni f16): {e}"))?;
            d.iter().map(|h| h.to_f32()).collect()
        }
    };

    if logits.len() < 3 {
        anyhow::bail!(
            "esperaba 3 logits (entailment/neutral/contradiction), llegaron {}",
            logits.len()
        );
    }

    // XNLI label order, from the model's own config.json:
    //   0 = entailment · 1 = neutral · 2 = contradiction
    let max = logits[..3].iter().cloned().fold(f32::MIN, f32::max);
    let exps: [f64; 3] = [
        ((logits[0] - max) as f64).exp(),
        ((logits[1] - max) as f64).exp(),
        ((logits[2] - max) as f64).exp(),
    ];
    let sum: f64 = exps.iter().sum();

    Ok(([exps[0] / sum, exps[1] / sum, exps[2] / sum], truncated))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distribuciones REALES medidas con el modelo fp32 sobre el corpus. Si el modelo
    /// cambia, estos números cambian — y entonces hay que volver a medir, no a suponer.
    #[test]
    fn real_distributions_get_the_right_verdict() {
        // «cuba-memorys está escrito en Java» contra la evidencia que dice Rust.
        // EL bug original: el coseno le daba 0.61, más que a la afirmación verdadera.
        let v = decide([0.000, 0.001, 0.999]);
        assert_eq!(v.label, Entailment::Contradicts);
        assert!(v.decisive);

        // La misma evidencia, la afirmación verdadera.
        let v = decide([0.998, 0.001, 0.001]);
        assert_eq!(v.label, Entailment::Supports);
        assert!(v.decisive);

        // Evidencia que no habla del tema. No vota — ni a favor ni en contra.
        let v = decide([0.016, 0.887, 0.097]);
        assert_eq!(v.label, Entailment::Neutral);
        assert!(
            v.decisive,
            "no es indecisión: la evidencia dice claramente que no habla de esto"
        );

        // La contradicción verdadera más DÉBIL que encontré: puerto 5488 vs 5491.
        // Marca el listón de CONTRA_FLOOR — si sube, esto deja de detectarse.
        let v = decide([0.002, 0.267, 0.731]);
        assert_eq!(v.label, Entailment::Contradicts);
        assert!(v.decisive);
    }

    /// The asymmetry that keeps false claims from being confirmed: a spurious entailment
    /// at 0.69 must NOT become `supports`, while genuine ones at 0.95+ must.
    #[test]
    fn weak_entailment_never_confirms() {
        let spurious = decide([0.693, 0.100, 0.207]);
        assert!(
            !spurious.decisive,
            "0.69 de entailment no puede confirmar nada: {:?}",
            spurious
        );
        assert_eq!(spurious.label, Entailment::Neutral);

        let genuine = decide([0.952, 0.036, 0.011]);
        assert_eq!(genuine.label, Entailment::Supports);
        assert!(genuine.decisive);
    }

    #[test]
    fn verdicts_speak_the_judge_vocabulary() {
        // The NLI verdict has to drop into the same slot an LLM verdict would, or
        // `compute_grounding_judged` cannot read it.
        assert_eq!(Entailment::Supports.as_verdict(), "supports");
        assert_eq!(Entailment::Contradicts.as_verdict(), "contradicts");
        assert_eq!(
            Entailment::Neutral.as_verdict(),
            "unrelated",
            "neutral must map to `unrelated`: it counts for NEITHER side, which is the \
             whole repair — being on-topic is not support"
        );
    }
}
