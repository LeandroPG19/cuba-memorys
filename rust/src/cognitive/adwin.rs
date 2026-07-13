//! ADWIN — Adaptive Windowing for Concept Drift Detection.
//!
//! Bifet, A., & Gavaldà, R. (2007).
//! "Learning from Time-Changing Data with Adaptive Windowing." SDM 2007.
//!
//! Maintains a sliding window of recent observations and detects when the
//! distribution shifts (concept drift) using a Hoeffding-bound based test.
//! When drift is detected, the older portion of the window is discarded.
//!
//! ## Why this matters for cuba-memorys
//!
//! - **Embedding distribution drift**: when the project topic shifts, the
//!   centroid of recent observations moves. ADWIN over the cosine-distance
//!   distribution flags it → trigger re-tagging or re-embedding.
//! - **Calibration drift in `brain_verify_log`**: outcomes start swinging
//!   towards "incorrect" → the judge model is degrading or the corpus has
//!   shifted under it.
//! - **Access-pattern drift**: a previously hot entity goes cold → schedule
//!   it for decay-eligible status.
//!
//! Hoeffding bound (one-sided, Theorem 3.1 of Bifet-Gavaldà 2007):
//!   ε_cut = sqrt((1/(2·m)) · ln(4/δ))
//! where m = harmonic mean of |W0|, |W1|, and δ is the confidence
//! parameter (default 0.002 → ~99.8% confidence).

use std::collections::VecDeque;

const DEFAULT_DELTA: f64 = 0.002;

/// Single-feature ADWIN tracker. Generic over f64 measurements (typical
/// uses: cosine distance, calibration outcome 0/1, access count delta).
#[derive(Debug, Clone)]
pub struct Adwin {
    window: VecDeque<f64>,
    delta: f64,
    max_window: usize,
}

impl Adwin {
    pub fn new(delta: f64, max_window: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_window),
            delta: delta.clamp(1e-6, 0.5),
            max_window: max_window.max(8),
        }
    }

    pub fn with_default() -> Self {
        Self::new(DEFAULT_DELTA, 1024)
    }

    /// Push a new measurement and return true if drift was detected.
    /// On drift, the older portion of the window is automatically pruned.
    pub fn add(&mut self, value: f64) -> bool {
        self.window.push_back(value);
        if self.window.len() > self.max_window {
            self.window.pop_front();
        }
        self.detect_and_shrink()
    }

    /// Current window length.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Mean of the current window.
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().sum::<f64>() / self.window.len() as f64
    }

    /// Try every cut point in [4, len-4] and return true if any cut shows
    /// |μ_left - μ_right| > ε_cut. When found, drop everything before the
    /// cut. Implements Theorem 3.1 of Bifet-Gavaldà 2007.
    fn detect_and_shrink(&mut self) -> bool {
        let n = self.window.len();
        if n < 8 {
            return false;
        }

        // Precompute prefix sums for O(1) mean queries
        let mut prefix = Vec::with_capacity(n + 1);
        prefix.push(0.0);
        // Carry the running sum instead of re-reading the vector's tail: same
        // result, no unwrap, and it does not depend on `prefix` being non-empty.
        let mut running = 0.0;
        for &v in &self.window {
            running += v;
            prefix.push(running);
        }
        let total = prefix[n];

        // `cut` is used to index multiple arrays + arithmetic; iterator form
        // would produce a less readable pattern, so we explicitly opt out.
        // The loop index IS the quantity of interest — `cut` is the split point
        // whose statistic we are testing, and both prefix[cut] and prefix[n]-prefix[cut]
        // are read from it. An iterator would have to reconstruct the index anyway.
        #[allow(clippy::needless_range_loop)]
        for cut in 4..(n - 4) {
            let n0 = cut as f64;
            let n1 = (n - cut) as f64;
            let mean0 = prefix[cut] / n0;
            let mean1 = (total - prefix[cut]) / n1;

            // Harmonic mean of window sizes (Bifet-Gavaldà m parameter)
            let m = (n0 * n1) / (n0 + n1);
            // Bonferroni correction for trying many cut points
            let delta_prime = self.delta / n as f64;
            let epsilon_cut = ((1.0 / (2.0 * m)) * (4.0 / delta_prime).ln()).sqrt();

            if (mean0 - mean1).abs() > epsilon_cut {
                // Drift! Drop everything before `cut`.
                self.window.drain(..cut);
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_window_no_drift() {
        let mut a = Adwin::with_default();
        assert!(!a.add(0.5));
        assert_eq!(a.mean(), 0.5);
    }

    #[test]
    fn stable_distribution_no_drift() {
        let mut a = Adwin::with_default();
        let mut any_drift = false;
        for i in 0..200 {
            // Stable mean ~0.5 with small noise
            let noise = ((i * 7919) % 100) as f64 / 1000.0 - 0.05;
            any_drift |= a.add(0.5 + noise);
        }
        assert!(!any_drift, "stable distribution should not trigger drift");
    }

    #[test]
    fn shift_triggers_drift() {
        let mut a = Adwin::with_default();
        // 100 samples around 0.2, then 100 around 0.8 → clear drift
        for _ in 0..100 {
            a.add(0.2);
        }
        let mut detected = false;
        for _ in 0..100 {
            if a.add(0.8) {
                detected = true;
                break;
            }
        }
        assert!(detected, "100→100 mean shift 0.2→0.8 must be detected");
    }

    #[test]
    fn small_sample_does_not_panic() {
        let mut a = Adwin::with_default();
        for _ in 0..7 {
            assert!(!a.add(0.5));
        }
    }
}
