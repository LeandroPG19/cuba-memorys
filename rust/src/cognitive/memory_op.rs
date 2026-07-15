#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOp {
    Add,
    Update,
    Delete,
    Noop,
}

impl MemoryOp {
    pub fn from_judgment(verdict: &str, confidence: f64, conf_floor: f64) -> Self {
        if confidence < conf_floor {
            return MemoryOp::Noop;
        }
        match verdict {
            "supersedes" | "contradicts" => MemoryOp::Update,
            "complementary" | "unrelated" => MemoryOp::Add,
            _ => MemoryOp::Noop,
        }
    }

    pub fn supersedes_old(self) -> bool {
        matches!(self, MemoryOp::Update | MemoryOp::Delete)
    }

    pub fn adds_new(self) -> bool {
        matches!(self, MemoryOp::Add | MemoryOp::Update)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MemoryOp::Add => "add",
            MemoryOp::Update => "update",
            MemoryOp::Delete => "delete",
            MemoryOp::Noop => "noop",
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OpBreakdown {
    pub add: u32,
    pub update: u32,
    pub delete: u32,
    pub noop: u32,
}

impl OpBreakdown {
    pub fn record(&mut self, op: MemoryOp) {
        match op {
            MemoryOp::Add => self.add += 1,
            MemoryOp::Update => self.update += 1,
            MemoryOp::Delete => self.delete += 1,
            MemoryOp::Noop => self.noop += 1,
        }
    }

    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "add": self.add,
            "update": self.update,
            "delete": self.delete,
            "noop": self.noop,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOOR: f64 = 0.5;

    #[test]
    fn supersedes_map_to_update() {
        assert_eq!(
            MemoryOp::from_judgment("supersedes", 0.9, FLOOR),
            MemoryOp::Update
        );
        assert_eq!(
            MemoryOp::from_judgment("contradicts", 0.7, FLOOR),
            MemoryOp::Update
        );
    }

    #[test]
    fn coexisting_maps_to_add() {
        assert_eq!(
            MemoryOp::from_judgment("complementary", 0.9, FLOOR),
            MemoryOp::Add
        );
        assert_eq!(
            MemoryOp::from_judgment("unrelated", 0.6, FLOOR),
            MemoryOp::Add
        );
    }

    #[test]
    fn low_confidence_is_noop_even_for_supersedes() {
        assert_eq!(
            MemoryOp::from_judgment("supersedes", 0.49, FLOOR),
            MemoryOp::Noop
        );
        assert_eq!(
            MemoryOp::from_judgment("contradicts", 0.1, FLOOR),
            MemoryOp::Noop
        );
    }

    #[test]
    fn unknown_verdict_is_noop() {
        assert_eq!(
            MemoryOp::from_judgment("unknown", 0.99, FLOOR),
            MemoryOp::Noop
        );
        assert_eq!(
            MemoryOp::from_judgment("garbage", 0.99, FLOOR),
            MemoryOp::Noop
        );
    }

    #[test]
    fn op_flags_are_consistent() {
        assert!(MemoryOp::Update.supersedes_old() && MemoryOp::Update.adds_new());
        assert!(MemoryOp::Add.adds_new() && !MemoryOp::Add.supersedes_old());
        assert!(MemoryOp::Delete.supersedes_old() && !MemoryOp::Delete.adds_new());
        assert!(!MemoryOp::Noop.supersedes_old() && !MemoryOp::Noop.adds_new());
    }

    #[test]
    fn breakdown_tallies() {
        let mut b = OpBreakdown::default();
        b.record(MemoryOp::Add);
        b.record(MemoryOp::Update);
        b.record(MemoryOp::Update);
        b.record(MemoryOp::Noop);
        assert_eq!((b.add, b.update, b.delete, b.noop), (1, 2, 0, 1));
    }
}
