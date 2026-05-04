//! Hebbian learning — Oja's rule with BCM metaplastic throttling.
//!
//! V1: BCM Theory (Bienenstock-Cooper-Munro, 1982): static threshold θ_M = 50.0.
//! V2: Dynamic sliding threshold (Gemini Deep Research 2026-03-14).
//! V3: Persistent EMA (Deep Research Calibration 2026-03-14).
//!     θ_M = max(θ_min, (1-α_ema) × θ_prev + α_ema × access_count)
//!     α_ema = 0.15, θ_M persisted in brain_entities.bcm_theta column.
//!     Prevents saturation of central nodes during bursty access.
//!
//! When a node is accessed frequently, θ_M rises → boost decreases.
//! When a node is idle, θ_M normalizes → boost recovers.

use anyhow::Result;
use sqlx::PgPool;

use crate::constants::{BCM_THROTTLE_SCALE, HEBBIAN_ACCESS_BOOST};

// ── BCM V3: EMA Parameters ────────────────────────────────────────
/// Minimum BCM threshold floor — prevents division by near-zero.
const BCM_THETA_MIN: f64 = 10.0;

/// V3: EMA smoothing factor for θ_M updates.
/// Lower = smoother (more memory), higher = more reactive.
const BCM_EMA_ALPHA: f64 = 0.15;

/// V0.9: Δt time-constant (seconds) for the burst-suppression factor.
/// τ=600 means a re-access after 10 min recovers ~63% of the boost; after
/// 1 hour, ~99.8%. Inspired by STDP triplet rules (Pfister-Gerstner 2006)
/// without requiring spike-train modeling.
const HEBBIAN_TAU_SECS: f64 = 600.0;

/// Boost entity importance on access with V3 EMA BCM throttling.
///
/// FIX R-003: Single atomic UPDATE — no read-modify-write race.
/// V3: θ_M = max(10, EMA(θ_prev, access_count)) persisted in bcm_theta.
/// V0.9: Δt-aware burst suppression — `boost *= (1 - exp(-Δt/τ))`.
///   Re-access in same second → factor 0 (anti-saturation, prevents
///   amplification of burst access patterns). Δt > 1h → factor ≈ 1
///   (normal access fully boosts). Implements asymmetric pre-post timing
///   (cf. STDP triplet rules, Pfister-Gerstner 2006) at lookup time.
pub async fn boost_on_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_entities SET
            bcm_theta = GREATEST($5, (1.0 - $6) * COALESCE(bcm_theta, $5) + $6 * access_count::float8),
            importance = LEAST(
                importance + $1
                  * GREATEST(0.1, 1.0 - (access_count::float8 / GREATEST(COALESCE(bcm_theta, $2), access_count::float8)) * $3)
                  * (1.0 - EXP(-LEAST(EXTRACT(EPOCH FROM (NOW() - updated_at)), 86400.0) / $7)),
                1.0
            ),
            access_count = access_count + 1,
            updated_at = NOW()
         WHERE id = $4"
    )
    .bind(HEBBIAN_ACCESS_BOOST)  // $1
    .bind(BCM_THETA_MIN)         // $2
    .bind(BCM_THROTTLE_SCALE)    // $3
    .bind(entity_id)             // $4
    .bind(BCM_THETA_MIN)         // $5 (floor)
    .bind(BCM_EMA_ALPHA)         // $6
    .bind(HEBBIAN_TAU_SECS)      // $7
    .execute(pool)
    .await?;
    Ok(())
}

/// Boost neighbors via 1-hop diffusion (Collins & Loftus 1975).
///
/// V0.7 (Mejora 6): Weighted by MAX(relation strength) per neighbor.
/// Previously all neighbors received uniform boost regardless of edge weight.
/// Collins & Loftus explicitly specify activation proportional to link strength.
///
/// Note: GROUP BY repeats the full CASE expression (not the alias `neighbor_id`)
/// because PostgreSQL processes GROUP BY before SELECT, so SELECT aliases are
/// not yet visible at GROUP BY evaluation time (SQL standard §7.9).
pub async fn boost_neighbors(pool: &PgPool, entity_id: uuid::Uuid) -> Result<usize> {
    let result = sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(importance + $1 * 0.5 * sub.max_strength, 1.0),
            updated_at = NOW()
         FROM (
             SELECT
                 CASE WHEN from_entity = $2 THEN to_entity ELSE from_entity END AS neighbor_id,
                 MAX(strength) AS max_strength
             FROM brain_relations
             WHERE from_entity = $2 OR to_entity = $2
             GROUP BY CASE WHEN from_entity = $2 THEN to_entity ELSE from_entity END
         ) sub
         WHERE brain_entities.id = sub.neighbor_id",
    )
    .bind(HEBBIAN_ACCESS_BOOST)
    .bind(entity_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}
