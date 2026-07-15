use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianCalibration {
    pub alpha: f32,
    pub beta: f32,
}

impl Default for BayesianCalibration {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl BayesianCalibration {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn expected_probability(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    pub fn update(&mut self, success: bool) {
        if success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }

    pub fn uncertainty(&self) -> f32 {
        let a = self.alpha;
        let b = self.beta;
        let sum = a + b;
        ((a * b) / (sum * sum * (sum + 1.0))).sqrt()
    }
}

pub fn mahalanobis_distance_diagonal(x: &[f32], mean: &[f32], variance: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..x.len().min(mean.len()).min(variance.len()) {
        let diff = x[i] - mean[i];
        sum += (diff * diff) / variance[i].max(1e-6);
    }
    sum.sqrt()
}

pub fn should_abstain(distance: f32, confidence: f32, uncertainty: f32) -> bool {
    distance > 3.0 || confidence < 0.5 || uncertainty > 0.2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_update() {
        let mut cal = BayesianCalibration::new();
        cal.update(true);
        cal.update(true);
        cal.update(false);
        assert!((cal.expected_probability() - 0.6).abs() < 0.02);
    }
}
