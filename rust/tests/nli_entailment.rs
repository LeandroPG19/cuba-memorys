//! The NLI judge, measured against the exact failure that created it.
//!
//! `cuba_faro mode=verify` used to score the FALSE claim ("cuba-memorys está escrito
//! en Java") at 0.61 and the TRUE one at 0.59 — because it was measuring cosine
//! similarity, which says what a text is *about*, not what it *asserts*. v0.11.2
//! fixed that with an LLM judge at ~20 s a call. This suite asserts the local model
//! gets the same three answers, and gets them in milliseconds.
//!
//! Skipped, not failed, when no model is installed: a contributor without a 323 MB
//! download must still be able to run `cargo test`.

use cuba_memorys::cognitive::nli::{self, Entailment};

const EVIDENCE_RUST: &str = "cuba-memorys es un servidor MCP de memoria escrito en Rust, \
                             con PostgreSQL y pgvector para la búsqueda semántica.";

fn skip_unless_model() -> bool {
    if nli::available() && nli::enabled() {
        return false;
    }
    eprintln!(
        "SKIP: no hay modelo NLI instalado (CUBA_NLI_PATH o ~/.cache/cuba-memorys/models-nli)"
    );
    true
}

#[tokio::test]
async fn true_claim_is_supported() {
    if skip_unless_model() {
        return;
    }
    let v = nli::entails(EVIDENCE_RUST, "cuba-memorys está escrito en Rust")
        .await
        .expect("el NLI debe emitir un veredicto, no un error");

    assert_eq!(
        v.label,
        Entailment::Supports,
        "la evidencia dice literalmente que está en Rust; p={:.3}",
        v.confidence
    );
    assert!(
        v.confidence > 0.5,
        "confianza demasiado baja: {}",
        v.confidence
    );
}

#[tokio::test]
async fn false_claim_is_contradicted() {
    if skip_unless_model() {
        return;
    }
    // THE regression test. Cosine gave this 0.61 — higher than the true claim —
    // because "escrito en Java" and "escrito en Rust" are nearly the same vector:
    // same subject, same shape, one word apart. No threshold separates them. Only
    // something that READS can.
    let v = nli::entails(EVIDENCE_RUST, "cuba-memorys está escrito en Java")
        .await
        .expect("el NLI debe emitir un veredicto, no un error");

    assert_eq!(
        v.label,
        Entailment::Contradicts,
        "Rust y Java se excluyen; el coseno no lo veía. p={:.3}",
        v.confidence
    );
}

#[tokio::test]
async fn unrelated_claim_is_neutral() {
    if skip_unless_model() {
        return;
    }
    // The third failure mode, and the subtlest: evidence that is silent on a claim
    // must NOT count as support. `unrelated` is not a weak yes — it is not a vote.
    let v = nli::entails(EVIDENCE_RUST, "la paella es un plato valenciano")
        .await
        .expect("el NLI debe emitir un veredicto, no un error");

    assert_eq!(
        v.label,
        Entailment::Neutral,
        "la evidencia no dice nada sobre paella; p={:.3}",
        v.confidence
    );
}

#[tokio::test]
async fn english_still_works() {
    if skip_unless_model() {
        return;
    }
    // We chose mDeBERTa over BART-MNLI *for* Spanish. Confirm we did not lose English
    // on the way — 23% of the corpus is English.
    let v = nli::entails(
        "The reranker runs as a separate ONNX session and is disabled by default.",
        "The reranker is enabled by default.",
    )
    .await
    .expect("el NLI debe emitir un veredicto, no un error");

    assert_eq!(v.label, Entailment::Contradicts, "p={:.3}", v.confidence);
}

/// The premise the corpus actually stores is a paragraph, not an XNLI sentence.
///
/// Undecomposed, this exact pair scored [0.19 entail · 0.41 neutral · 0.40 contra] —
/// the false claim escaped as "unrelated" because three distractor clauses about
/// PostgreSQL and pgvector diluted the one clause that mattered. Clause-level scoring
/// is not a tuning knob; it is what makes the verdict survive a realistic premise.
#[tokio::test]
async fn distractor_clauses_do_not_dilute_the_verdict() {
    if skip_unless_model() {
        return;
    }
    let v = nli::entails(EVIDENCE_RUST, "cuba-memorys está escrito en Java")
        .await
        .expect("veredicto");

    assert_eq!(
        v.label,
        Entailment::Contradicts,
        "probs={:?} — la contradicción (0.40) le gana a la entailment (0.19) más de \
         dos a uno; si esto vuelve a dar Neutral es que se leyó el argmax de una \
         cabeza de 3 clases y la masa de 'neutral' se tragó la señal",
        v.probs
    );
    assert!(
        v.decisive,
        "y debe ser un veredicto, no un empate: {:?}",
        v.probs
    );
}

/// The model's one measured blind spot, and the guarantee that contains it.
///
/// Knowing that a cross-encoder and a bi-encoder are mutually exclusive is domain
/// knowledge, not linguistic inference, and XNLI cannot supply it. Shown this pair,
/// the model does not merely hesitate — it scores *entailment* at 0.693, the exact
/// opposite of the truth, and an argmax would have published that as `supports`.
///
/// It does not, because entailment must clear `SUPPORT_FLOOR` (0.80) and 0.693 does
/// not. Every genuine entailment we measured scored ≥0.95; spurious ones live in the
/// 0.6s. So the claim comes back undecided and `NliJudge` escalates to an LLM that
/// has read more than sentence pairs.
///
/// **This test is the guardrail.** If it ever fails with `decisive == true` and
/// `Supports`, a false claim is being confirmed by the store — the precise bug this
/// whole subsystem exists to prevent.
#[tokio::test]
async fn a_false_claim_is_never_confirmed_on_weak_entailment() {
    if skip_unless_model() {
        return;
    }
    let v = nli::entails(
        "The reranker is a cross-encoder based on XLM-RoBERTa.",
        "The reranker is a bi-encoder.",
    )
    .await
    .expect("veredicto");

    assert_ne!(
        (v.label, v.decisive),
        (Entailment::Supports, true),
        "SE ESTÁ CONFIRMANDO UNA AFIRMACIÓN FALSA. probs={:?}. La entailment espuria \
         (0.69) se coló por debajo de SUPPORT_FLOOR y el juez la publicó como \
         `supports` en vez de escalar.",
        v.probs
    );

    if !v.decisive {
        assert_eq!(
            v.label,
            Entailment::Neutral,
            "sin veredicto, la etiqueta no puede ser dirigida: {:?}",
            v.probs
        );
    }
}

/// The flip side: abstaining must stay *rare*. A verifier that shrugs at everything
/// is safe and useless, and would pass the test above trivially.
///
/// Substituting one technical term for another inside the same frame is the shape
/// almost every real false memory takes, and the model must call these — decisively.
#[tokio::test]
async fn technical_substitutions_are_caught_decisively() {
    if skip_unless_model() {
        return;
    }
    let cases = [
        ("El servidor usa PostgreSQL.", "El servidor usa MySQL."),
        ("El índice es HNSW.", "El índice es IVFFlat."),
        (
            "El servidor corre en el puerto 5488.",
            "El servidor corre en el puerto 5491.",
        ),
        (
            "cuba-memorys está escrito en Rust.",
            "cuba-memorys está escrito en Python.",
        ),
        (
            "The reranker runs as a separate ONNX session and is disabled by default.",
            "The reranker is enabled by default.",
        ),
    ];

    for (premise, claim) in cases {
        let v = nli::entails(premise, claim).await.expect("veredicto");
        assert_eq!(
            (v.label, v.decisive),
            (Entailment::Contradicts, true),
            "«{claim}» contradice «{premise}» y debe detectarse sin titubeos; probs={:?}",
            v.probs
        );
    }
}

/// The whole point of doing this locally: the LLM judge took ~20 s per claim, which
/// made `verify` unusable on more than a couple of pieces of evidence. If the local
/// model is not orders of magnitude faster, it bought nothing.
#[tokio::test]
async fn a_verdict_costs_milliseconds_not_seconds() {
    if skip_unless_model() {
        return;
    }
    // Warm the session: the first call pays the 323 MB load, which is not what we
    // are measuring — verify runs many claims against one loaded model.
    let _ = nli::entails(EVIDENCE_RUST, "cuba-memorys usa PostgreSQL").await;

    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        nli::entails(EVIDENCE_RUST, "cuba-memorys está escrito en Rust")
            .await
            .expect("veredicto");
    }
    let per_call = t0.elapsed() / 5;

    eprintln!("NLI: {:?} por veredicto", per_call);
    assert!(
        per_call < std::time::Duration::from_millis(500),
        "un veredicto tardó {per_call:?}; el juez LLM al que reemplaza tardaba ~20 s, \
         así que <500 ms es el listón mínimo para que esto valga la pena"
    );
}
