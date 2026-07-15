use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allen {
    Before,
    After,
    Meets,
    MetBy,
    Overlaps,
    OverlappedBy,
    Starts,
    StartedBy,
    During,
    Contains,
    Finishes,
    FinishedBy,
    Equals,
}

impl Allen {
    pub fn name(&self) -> &'static str {
        match self {
            Allen::Before => "before",
            Allen::After => "after",
            Allen::Meets => "meets",
            Allen::MetBy => "met_by",
            Allen::Overlaps => "overlaps",
            Allen::OverlappedBy => "overlapped_by",
            Allen::Starts => "starts",
            Allen::StartedBy => "started_by",
            Allen::During => "during",
            Allen::Contains => "contains",
            Allen::Finishes => "finishes",
            Allen::FinishedBy => "finished_by",
            Allen::Equals => "equals",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Interval {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl Interval {
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        debug_assert!(end >= start, "interval end must be ≥ start");
        Self { start, end }
    }
}

pub fn relation(x: Interval, y: Interval) -> Allen {
    use std::cmp::Ordering::*;
    let s_cmp = x.start.cmp(&y.start);
    let e_cmp = x.end.cmp(&y.end);
    let xe_vs_ys = x.end.cmp(&y.start);
    let xs_vs_ye = x.start.cmp(&y.end);

    match (s_cmp, e_cmp) {
        (Equal, Equal) => Allen::Equals,
        (Equal, Less) => Allen::Starts,
        (Equal, Greater) => Allen::StartedBy,
        (Greater, Equal) => Allen::Finishes,
        (Less, Equal) => Allen::FinishedBy,
        (Greater, Less) => Allen::During,
        (Less, Greater) => Allen::Contains,
        (Less, Less) => match xe_vs_ys {
            Less => Allen::Before,
            Equal => Allen::Meets,
            Greater => Allen::Overlaps,
        },
        (Greater, Greater) => match xs_vs_ye {
            Greater => Allen::After,
            Equal => Allen::MetBy,
            Less => Allen::OverlappedBy,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn iv(secs_start: i64, secs_end: i64) -> Interval {
        let base = DateTime::<Utc>::from_timestamp(0, 0).unwrap();
        Interval::new(
            base + Duration::seconds(secs_start),
            base + Duration::seconds(secs_end),
        )
    }

    #[test]
    fn before_after() {
        assert_eq!(relation(iv(0, 10), iv(20, 30)), Allen::Before);
        assert_eq!(relation(iv(20, 30), iv(0, 10)), Allen::After);
    }

    #[test]
    fn meets() {
        assert_eq!(relation(iv(0, 10), iv(10, 20)), Allen::Meets);
        assert_eq!(relation(iv(10, 20), iv(0, 10)), Allen::MetBy);
    }

    #[test]
    fn overlaps() {
        assert_eq!(relation(iv(0, 15), iv(10, 25)), Allen::Overlaps);
        assert_eq!(relation(iv(10, 25), iv(0, 15)), Allen::OverlappedBy);
    }

    #[test]
    fn during_contains() {
        assert_eq!(relation(iv(5, 10), iv(0, 20)), Allen::During);
        assert_eq!(relation(iv(0, 20), iv(5, 10)), Allen::Contains);
    }

    #[test]
    fn starts_started_by() {
        assert_eq!(relation(iv(0, 5), iv(0, 20)), Allen::Starts);
        assert_eq!(relation(iv(0, 20), iv(0, 5)), Allen::StartedBy);
    }

    #[test]
    fn finishes_finished_by() {
        assert_eq!(relation(iv(15, 20), iv(0, 20)), Allen::Finishes);
        assert_eq!(relation(iv(0, 20), iv(15, 20)), Allen::FinishedBy);
    }

    #[test]
    fn equals() {
        assert_eq!(relation(iv(0, 10), iv(0, 10)), Allen::Equals);
    }

    #[test]
    fn all_relations_have_unique_names() {
        let all = [
            Allen::Before,
            Allen::After,
            Allen::Meets,
            Allen::MetBy,
            Allen::Overlaps,
            Allen::OverlappedBy,
            Allen::Starts,
            Allen::StartedBy,
            Allen::During,
            Allen::Contains,
            Allen::Finishes,
            Allen::FinishedBy,
            Allen::Equals,
        ];
        let names: std::collections::HashSet<&str> = all.iter().map(|r| r.name()).collect();
        assert_eq!(names.len(), 13);
    }
}
