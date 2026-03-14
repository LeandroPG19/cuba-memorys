//! Custom FSRS-6 implementation (ADR-001: ~50 LOC core, zero deps).
//!
//! Based on Ye 2024 paper, 21 parameters.
//! V4: FSRS-6 upgrade from FSRS-4 (21 params instead of 4).
//! V5: Personalizable decay rate w20 (Expertium 2025) — per-entity.
//! V6: Topological Inertia (Gemini Deep Research 2026-03-14).
//!     Hub nodes with high PageRank retain stability longer:
//!     S'_graph = S' × (1 + γ * ln(1 + PR(n)))
//!     γ = 0.5 (default, tunable). Prevents "structural amnesia"
//!     where critical hub nodes are pruned despite high connectivity.

use crate::constants::{DECAY_THRESHOLD, DEFAULT_DECAY_RATE, FSRS6_DEFAULT_PARAMS};
use anyhow::Result;
use sqlx::PgPool;

// ── Topological Inertia Parameters (Gemini Deep Research 2026-03-14) ─
/// Topological inertia coefficient: scales PageRank influence on stability.
/// Higher γ → hub nodes retain stability more aggressively.
/// Range: [0.0, 2.0]. Default: 0.5. Set to 0.0 to disable.
/// V3: Configurable via CUBA_TOPO_GAMMA env var.
const TOPO_INERTIA_GAMMA_DEFAULT: f64 = 0.5;

/// Read γ from env (cached with OnceLock for single-read).
fn topo_gamma() -> f64 {
    use std::sync::OnceLock;
    static GAMMA: OnceLock<f64> = OnceLock::new();
    *GAMMA.get_or_init(|| {
        std::env::var("CUBA_TOPO_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(TOPO_INERTIA_GAMMA_DEFAULT)
    })
}

/// Calculate retrievability from stability and elapsed days.
///
/// Formula: R(t, S) = (1 + (19/81) * t / S)^(-decay_rate)
///
/// V5: decay_rate is now personalizable per entity (w20 in FSRS-6).
pub fn retrievability(stability: f64, elapsed_days: f64) -> f64 {
    retrievability_with_decay(stability, elapsed_days, DEFAULT_DECAY_RATE)
}

/// Retrievability with custom decay rate (FSRS-6 w20).
pub fn retrievability_with_decay(stability: f64, elapsed_days: f64, decay_rate: f64) -> f64 {
    if stability <= 0.0 || elapsed_days < 0.0 {
        return 0.0;
    }
    let factor = 19.0_f64 / 81.0;
    let clamped_decay = decay_rate.clamp(0.1, 0.8);
    (1.0 + factor * elapsed_days / stability).powf(-clamped_decay)
}

/// Compute adaptive decay rate from access_count (FSRS-6 w20).
///
/// Uses sigmoid mapping: decay = 0.1 + 0.7 / (1 + e^(0.05 * (count - 30)))
pub fn adaptive_decay_rate(access_count: i32) -> f64 {
    let sigmoid = 1.0 / (1.0 + (0.05 * (access_count as f64 - 30.0)).exp());
    (0.1 + 0.7 * sigmoid).clamp(0.1, 0.8)
}

/// Update stability after a recall event.
///
/// Uses FSRS-6 formula with 21 default parameters.
pub fn update_stability(
    current_stability: f64,
    difficulty: f64,
    retrievability: f64,
    rating: u8,
) -> f64 {
    let w = &FSRS6_DEFAULT_PARAMS;
    let d = difficulty.clamp(1.0, 10.0);
    let r = retrievability.clamp(0.0, 1.0);
    let s = current_stability.max(0.01);

    match rating {
        0 => {
            w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0)
                * (w[14] * (1.0 - r)).exp()
        }
        _ => {
            let rating_bonus = match rating {
                1 => w[15],
                2 => 1.0,
                3 => w[16],
                _ => 1.0,
            };
            s * (1.0 + (w[8] * d.powf(-w[9]) * (s.powf(-w[10]) - 1.0) * rating_bonus).exp())
        }
    }
}

/// V6: Apply Topological Inertia to FSRS stability.
///
/// S'_graph = S' × (1 + γ × ln(1 + PR(n)))
///
/// Hub nodes (high PageRank) get higher effective stability,
/// preventing "structural amnesia" where important knowledge hubs
/// are forgotten despite being well-connected.
///
/// Scientific basis: Scale-free network theory (Barabási 1999).
/// Hub nodes serve as "memory anchors" — losing them fragments
/// the knowledge graph disproportionately.
pub fn apply_topological_inertia(base_stability: f64, pagerank: f64) -> f64 {
    let pr = pagerank.max(0.0);
    base_stability * (1.0 + topo_gamma() * (1.0 + pr).ln())
}

/// Update difficulty based on rating.
pub fn update_difficulty(current_difficulty: f64, rating: u8) -> f64 {
    let w = &FSRS6_DEFAULT_PARAMS;
    let d = current_difficulty;
    let delta = -w[6] * (rating as f64 - 3.0);
    let new_d = d + delta;
    let mean_reversion = w[7] * (w[4] - new_d);
    (new_d + mean_reversion).clamp(1.0, 10.0)
}

/// Batch decay — apply FSRS retrievability check on all observations.
///
/// V6: Incorporates topological inertia via PageRank proportional factor.
/// Entities with high PageRank resist decay (stability multiplied by
/// topological inertia factor in the threshold comparison).
pub async fn batch_decay(pool: &PgPool, protected: &[uuid::Uuid]) -> Result<usize> {
    let result = sqlx::query(
        r#"
        UPDATE brain_observations o SET
            stability = stability * 0.95,
            retrieval_strength = retrieval_strength * 0.95,
            updated_at = NOW()
        FROM brain_entities e
        WHERE o.entity_id = e.id
          AND o.entity_id NOT IN (SELECT UNNEST($1::uuid[]))
          AND o.observation_type != 'superseded'
          AND (
            -- V6: Topological inertia — multiply stability by (1 + γ * ln(1 + PR))
            -- High-PR entities have effective higher stability → resist decay
            POWER(
                1.0 + (19.0/81.0) * EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0
                    / GREATEST(
                        o.stability * (1.0 + $3 * LN(1.0 + COALESCE(e.importance, 0.0))),
                        0.01
                    ),
                -LEAST(0.8, GREATEST(0.1, 0.1 + 0.7 / (1.0 + EXP(0.05 * (e.access_count - 30)))))
            ) < $2
          )
        "#,
    )
    .bind(protected)
    .bind(DECAY_THRESHOLD)
    .bind(topo_gamma())
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrievability_fresh() {
        let r = retrievability(1.0, 0.0);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_retrievability_decays() {
        let r = retrievability(1.0, 10.0);
        assert!(r < 0.6, "R should decay: got {r}");
        assert!(r > 0.4, "R should not collapse: got {r}");
    }

    #[test]
    fn test_retrievability_high_stability() {
        let r = retrievability(30.0, 10.0);
        assert!(r > 0.8);
    }

    #[test]
    fn test_retrievability_custom_decay_slow() {
        let r_slow = retrievability_with_decay(1.0, 10.0, 0.1);
        let r_default = retrievability(1.0, 10.0);
        assert!(r_slow > r_default, "slow decay should retain more: {r_slow} vs {r_default}");
    }

    #[test]
    fn test_retrievability_custom_decay_fast() {
        let r_fast = retrievability_with_decay(1.0, 10.0, 0.8);
        let r_default = retrievability(1.0, 10.0);
        assert!(r_fast < r_default, "fast decay should retain less: {r_fast} vs {r_default}");
    }

    #[test]
    fn test_adaptive_decay_rate_low_access() {
        let rate = adaptive_decay_rate(0);
        assert!(rate > 0.5, "low access should have high decay: {rate}");
    }

    #[test]
    fn test_adaptive_decay_rate_high_access() {
        let rate = adaptive_decay_rate(100);
        assert!(rate < 0.2, "high access should have low decay: {rate}");
    }

    #[test]
    fn test_adaptive_decay_rate_bounds() {
        for count in [0, 1, 5, 10, 30, 50, 100, 500, 1000] {
            let rate = adaptive_decay_rate(count);
            assert!((0.1..=0.8).contains(&rate), "out of bounds at count={count}: {rate}");
        }
    }

    #[test]
    fn test_update_stability_recall() {
        let new_s = update_stability(1.0, 5.0, 0.9, 2);
        assert!(new_s > 1.0, "stability should increase on recall");
    }

    #[test]
    fn test_update_stability_forget() {
        let new_s = update_stability(10.0, 5.0, 0.3, 0);
        assert!(new_s < 10.0, "stability should decrease on forget, got {new_s}");
        assert!(new_s > 0.0, "stability should remain positive");
    }

    #[test]
    fn test_difficulty_bounds() {
        let d = update_difficulty(5.0, 0);
        assert!((1.0..=10.0).contains(&d));
    }

    // ── V6: Topological Inertia Tests ─────────────────────────────

    #[test]
    fn test_topological_inertia_hub_boost() {
        // High PageRank → stability multiplied up
        let base_s = 10.0;
        let s_hub = apply_topological_inertia(base_s, 0.5);
        assert!(s_hub > base_s, "hub should have higher stability: {s_hub} vs {base_s}");
    }

    #[test]
    fn test_topological_inertia_leaf_minimal() {
        // Very low PageRank → minimal boost (near 1.0 multiplier)
        let base_s = 10.0;
        let s_leaf = apply_topological_inertia(base_s, 0.001);
        assert!(
            (s_leaf - base_s).abs() < 0.1,
            "leaf node should have near-base stability: {s_leaf} vs {base_s}"
        );
    }

    #[test]
    fn test_topological_inertia_zero_pr() {
        // PageRank=0 → ln(1+0) = 0 → no boost
        let base_s = 10.0;
        let s = apply_topological_inertia(base_s, 0.0);
        assert!((s - base_s).abs() < 1e-10, "zero PR should have no boost: {s}");
    }

    #[test]
    fn test_topological_inertia_monotonic() {
        // Higher PR → higher effective stability (monotonic)
        let s1 = apply_topological_inertia(10.0, 0.1);
        let s2 = apply_topological_inertia(10.0, 0.5);
        let s3 = apply_topological_inertia(10.0, 1.0);
        assert!(s1 < s2, "monotonic: s1 < s2");
        assert!(s2 < s3, "monotonic: s2 < s3");
    }
}
