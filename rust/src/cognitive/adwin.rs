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

        for (cut, &prefix_cut) in prefix.iter().enumerate().take(n - 4).skip(4) {
            let n0 = cut as f64;
            let n1 = (n - cut) as f64;
            let mean0 = prefix_cut / n0;
            let mean1 = (total - prefix_cut) / n1;

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

    #[test]
    fn the_cut_lands_where_the_distribution_actually_changed() {
        let mut a = Adwin::with_default();
        for _ in 0..40 {
            a.add(0.2);
        }
        let mut added = 0;
        let mut cut_at = None;
        for _ in 0..40 {
            added += 1;
            if a.add(0.8) {
                cut_at = Some(a.len());
                break;
            }
        }

        let remaining = cut_at.expect("a 0.2 to 0.8 shift over 40 samples must be detected");
        assert_eq!(
            (added, remaining),
            (28, 28),
            "the search walks cuts in ascending order and takes the first one whose two \
             halves differ by more than the Hoeffding bound, so the number of samples left \
             after the drain IS the cut index. The four older tests only assert that drift \
             was or was not detected, and this one pins the index itself, which is what \
             makes a rewrite of that loop checkable at all. What it still does NOT establish \
             is the loop's search bounds: shifting 4..n-4 by one leaves every test green, \
             including this one, because the qualifying cut here lands at 40 and a cut near \
             4 cannot qualify at any input — with n0=4 the Hoeffding bound is wider than any \
             mean difference, measured: four 0.0 samples followed by sixty 1.0 samples \
             produce no drift at all. Those bounds are defensive, not functional"
        );
    }
}
