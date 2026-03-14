//! Dual-Strength Model unified with FSRS-6 (ADR-002 + Gemini Deep Research 2026-03-14).
//!
//! V1: Original Bjork & Bjork 1992 — separate SS/RS.
//! V2: SAC unification (Raaijmakers & Shiffrin → Pavlik & Anderson 2005).
//!     SS ↔ FSRS Stability (S), RS ↔ FSRS Retrievability (R).
//!     Update formula: ΔSS = α * (SS_max - SS) * e^(-β * RS)
//!     Parameters: α=0.5 (learning rate), β=0.5 (retrieval inhibition).
//!     V3: Agent-optimized (Deep Research 2026-03-14): agents don't need
//!         "desirable difficulty" — faster consolidation via higher α,
//!         lower retrieval penalty via lower β.
//! VF2: Testing Effect (Roediger & Karpicke 2006).

use crate::constants::{
    RETRIEVAL_DECAY_FACTOR, RETRIEVAL_SEARCH_BOOST, STORAGE_STRENGTH_INCREMENT,
};
use anyhow::Result;
use sqlx::PgPool;

// ── SAC Parameters (Gemini Deep Research 2026-03-14) ────────────
/// SAC learning rate: controls how fast storage strength approaches maximum.
/// V3: 0.2→0.5 (agent-optimized — faster consolidation).
const SAC_ALPHA: f64 = 0.5;
/// SAC retrieval inhibition: higher RS → smaller ΔSS (diminishing returns).
/// V3: 1.5→0.5 (agent-optimized — less penalty for high retrieval).
const SAC_BETA: f64 = 0.5;
/// Maximum storage strength (ceiling).
const SS_MAX: f64 = 1.0;

/// Update storage strength using SAC formula (V2 — unified with FSRS).
///
/// ΔSS = α × (SS_max − SS) × e^(−β × RS)
///
/// Key insight: When RS is high (memory easily accessible), ΔSS is small
/// (no need to strengthen encoding). When RS is low (hard to retrieve),
/// ΔSS is larger (desirable difficulty effect — Bjork 1994).
///
/// This replaces the simpler `SS + 0.1 * (1 - SS)` formula from V1.
pub fn increment_storage(current_ss: f64, retrieval_strength: f64) -> f64 {
    let rs = retrieval_strength.clamp(0.0, 1.0);
    let ss = current_ss.clamp(0.0, SS_MAX);
    let delta = SAC_ALPHA * (SS_MAX - ss) * (-SAC_BETA * rs).exp();
    (ss + delta).min(SS_MAX)
}

/// V1 compatibility: increment_storage without RS (assumes RS=0 → maximum delta).
pub fn increment_storage_simple(current: f64) -> f64 {
    let delta = STORAGE_STRENGTH_INCREMENT * (1.0 - current);
    (current + delta).min(1.0)
}

/// Decay retrieval strength based on elapsed time.
///
/// Formula: RS_new = RS * 0.95^days
pub fn decay_retrieval(current: f64, elapsed_days: f64) -> f64 {
    if elapsed_days <= 0.0 {
        return current;
    }
    (current * RETRIEVAL_DECAY_FACTOR.powf(elapsed_days)).max(0.0)
}

/// VF2: Testing Effect — search boosts retrieval more than creation.
///
/// Retrieval (actively finding a memory) strengthens it more
/// than passive study (creating/updating) — Roediger & Karpicke 2006.
pub fn search_boost_retrieval(current: f64) -> f64 {
    (current + RETRIEVAL_SEARCH_BOOST).min(1.0)
}

/// Memory state classification based on dual-strength values.
///
/// VF4: Active(≥70%), Dormant(40-70%), Silent(10-40%), Unavailable(<10%).
pub fn memory_state(storage: f64, retrieval: f64) -> &'static str {
    let composite = 0.5 * retrieval + 0.3 * storage + 0.2 * retrieval.powf(0.5);
    if composite >= 0.70 {
        "active"
    } else if composite >= 0.40 {
        "dormant"
    } else if composite >= 0.10 {
        "silent"
    } else {
        "unavailable"
    }
}

/// Batch update dual-strength on entity access (V2: SAC formula in SQL).
///
/// SAC: ΔSS = α × (1 − SS) × e^(−β × RS)
pub async fn on_entity_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        r#"
        UPDATE brain_observations SET
            storage_strength = LEAST(
                storage_strength + $1 * (1.0 - storage_strength) * EXP(-$2 * retrieval_strength),
                1.0
            ),
            retrieval_strength = 1.0,
            last_accessed = NOW()
        WHERE entity_id = $3
          AND observation_type != 'superseded'
        "#,
    )
    .bind(SAC_ALPHA)
    .bind(SAC_BETA)
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// VF2: Boost retrieval on search match (Testing Effect).
pub async fn on_search_match(pool: &PgPool, observation_ids: &[uuid::Uuid]) -> Result<()> {
    if observation_ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        r#"
        UPDATE brain_observations SET
            retrieval_strength = LEAST(retrieval_strength + $1, 1.0),
            last_accessed = NOW()
        WHERE id = ANY($2)
        "#,
    )
    .bind(RETRIEVAL_SEARCH_BOOST)
    .bind(observation_ids)
    .execute(pool)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sac_storage_increases() {
        // Low RS → large ΔSS (desirable difficulty)
        let s = increment_storage(0.5, 0.1);
        assert!(s > 0.5, "SAC should increase SS: got {s}");
        assert!(s <= 1.0);
    }

    #[test]
    fn test_sac_desirable_difficulty() {
        // Key SAC insight: low RS yields bigger ΔSS than high RS
        let delta_low_rs = increment_storage(0.5, 0.1) - 0.5;
        let delta_high_rs = increment_storage(0.5, 0.9) - 0.5;
        assert!(
            delta_low_rs > delta_high_rs,
            "low RS should give bigger delta: {delta_low_rs} vs {delta_high_rs}"
        );
    }

    #[test]
    fn test_sac_diminishing_returns() {
        // Near ceiling → small delta regardless of RS
        let s = increment_storage(0.99, 0.0);
        assert!(s - 0.99 < 0.01, "near ceiling delta should be tiny");
    }

    #[test]
    fn test_sac_parameters_calibrated() {
        // V3: α=0.5, β=0.5: at SS=0, RS=0 → ΔSS = 0.5 * 1.0 * e^0 = 0.5
        let s = increment_storage(0.0, 0.0);
        assert!((s - 0.5).abs() < 0.001, "SS=0,RS=0 → ΔSS should be α: got {s}");

        // At SS=0, RS=1 → ΔSS = 0.5 * 1.0 * e^(-0.5) ≈ 0.5 * 0.6065 ≈ 0.303
        let s = increment_storage(0.0, 1.0);
        assert!(s > 0.29 && s < 0.32, "SS=0,RS=1 → ΔSS ≈ 0.303: got {s}");
    }

    #[test]
    fn test_storage_simple_backward_compat() {
        let s = increment_storage_simple(0.5);
        assert!(s > 0.5);
        assert!(s <= 1.0);
    }

    #[test]
    fn test_retrieval_decays() {
        let r = decay_retrieval(1.0, 10.0);
        assert!(r < 1.0);
        assert!(r > 0.0);
    }

    #[test]
    fn test_memory_states() {
        assert_eq!(memory_state(1.0, 1.0), "active");
        assert_eq!(memory_state(0.5, 0.3), "dormant");
        assert_eq!(memory_state(0.2, 0.15), "silent");
        assert_eq!(memory_state(0.0, 0.0), "unavailable");
    }
}
