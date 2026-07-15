use std::collections::VecDeque;

const DEFAULT_DELTA: f64 = 0.002;

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

    pub fn add(&mut self, value: f64) -> bool {
        self.window.push_back(value);
        if self.window.len() > self.max_window {
            self.window.pop_front();
        }
        self.detect_and_shrink()
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().sum::<f64>() / self.window.len() as f64
    }

    fn detect_and_shrink(&mut self) -> bool {
        let n = self.window.len();
        if n < 8 {
            return false;
        }

        let mut prefix = Vec::with_capacity(n + 1);
        prefix.push(0.0);
        let mut running = 0.0;
        for &v in &self.window {
            running += v;
            prefix.push(running);
        }
        let total = prefix[n];

        #[allow(clippy::needless_range_loop)]
        for cut in 4..(n - 4) {
            let n0 = cut as f64;
            let n1 = (n - cut) as f64;
            let mean0 = prefix[cut] / n0;
            let mean1 = (total - prefix[cut]) / n1;

            let m = (n0 * n1) / (n0 + n1);
            let delta_prime = self.delta / n as f64;
            let epsilon_cut = ((1.0 / (2.0 * m)) * (4.0 / delta_prime).ln()).sqrt();

            if (mean0 - mean1).abs() > epsilon_cut {
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
            let noise = ((i * 7919) % 100) as f64 / 1000.0 - 0.05;
            any_drift |= a.add(0.5 + noise);
        }
        assert!(!any_drift, "stable distribution should not trigger drift");
    }

    #[test]
    fn shift_triggers_drift() {
        let mut a = Adwin::with_default();
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
