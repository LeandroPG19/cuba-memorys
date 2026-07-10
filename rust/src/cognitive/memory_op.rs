//! Phase 1.2: explicit ADD / UPDATE / DELETE / NOOP memory operations.
//!
//! Closes gap #2 (see docs/PLAN-MEJORAS-v0.11.md): cuba used to decide
//! reinforce/update/create by a raw cosine threshold — the brittle approach
//! mem0 abandoned. Instead, the LLM judge classifies the relationship between a
//! candidate fact and the most similar existing one, and this module maps that
//! judgment to a concrete operation. The *decision* is the LLM's; the mapping is
//! deterministic and unit-tested.
//!
//! Semantics against cuba's bitemporal model (rows are never physically
//! deleted — "delete" means supersede/invalidate):
//!
//! | Op     | supersedes_old | adds_new | meaning                                   |
//! |--------|:--------------:|:--------:|-------------------------------------------|
//! | Add    |       no       |   yes    | new fact coexists with the old one        |
//! | Update |      yes       |   yes    | new fact replaces an older version        |
//! | Delete |      yes       |    no    | old fact retracted, nothing new to store  |
//! | Noop   |       no       |    no    | duplicate / not confident enough — ignore |
//!
//! auto_extract only ever emits Add/Update/Noop automatically (a coding turn
//! always carries a positive fact, so a bare retraction is not produced by
//! extraction). Delete is modeled for explicit callers (cuba_forget / a future
//! cuba_juez recommendation) and is never inferred from a low-signal turn.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOp {
    Add,
    Update,
    Delete,
    Noop,
}

impl MemoryOp {
    /// Map an LLM judgment to an operation. `verdict` is the judge's vocabulary
    /// (`contradicts | supersedes | complementary | unrelated | unknown`);
    /// `confidence` is 0..1; `conf_floor` is the minimum confidence below which
    /// we refuse to act (Noop) to avoid superseding on a weak call.
    pub fn from_judgment(verdict: &str, confidence: f64, conf_floor: f64) -> Self {
        if confidence < conf_floor {
            return MemoryOp::Noop;
        }
        match verdict {
            // Both mean "the new fact wins over the old" → supersede + keep new.
            "supersedes" | "contradicts" => MemoryOp::Update,
            // Both can hold at once, or they are unrelated → keep both.
            "complementary" | "unrelated" => MemoryOp::Add,
            // "unknown" or anything unexpected → do not touch existing memory.
            _ => MemoryOp::Noop,
        }
    }

    /// Whether the operation invalidates (supersedes) the matched old observation.
    pub fn supersedes_old(self) -> bool {
        matches!(self, MemoryOp::Update | MemoryOp::Delete)
    }

    /// Whether the candidate fact should be stored as a new active observation.
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

/// Tally of operations chosen across a batch — surfaced in the auto_extract
/// response so callers can see what the judge decided, not just a bare count.
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
