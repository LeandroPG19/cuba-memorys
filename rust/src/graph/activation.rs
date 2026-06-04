//! Simple activation propagation for seed nodes (Boltzmann-style decay).

pub struct ActivationPropagation {
    decay: f32,
}

impl Default for ActivationPropagation {
    fn default() -> Self {
        Self { decay: 0.85 }
    }
}

impl ActivationPropagation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn propagate(&self, seed_nodes: &[usize], graph_size: usize) -> Vec<f32> {
        let mut activations = vec![0.0_f32; graph_size];
        for &node in seed_nodes {
            if node < graph_size {
                activations[node] = 1.0;
            }
        }
        activations.iter().map(|&v| v * self.decay).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propagate_decay() {
        let ap = ActivationPropagation::new();
        let out = ap.propagate(&[0, 2], 4);
        assert!((out[0] - 0.85).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
    }
}
