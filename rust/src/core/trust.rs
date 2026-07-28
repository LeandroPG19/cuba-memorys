pub const TRUSTED: &str = "trusted";
pub const QUARANTINED: &str = "quarantined";

pub fn quarantine_inference_enabled() -> bool {
    matches!(
        std::env::var("CUBA_QUARANTINE_INFERENCE").as_deref(),
        Ok("1") | Ok("on") | Ok("true")
    )
}

pub fn resolve(source: &str, explicit: Option<&str>, policy_quarantines_inference: bool) -> String {
    if let Some(t) = explicit
        && (t == TRUSTED || t == QUARANTINED)
    {
        return t.to_string();
    }
    if source == "inference" && policy_quarantines_inference {
        return QUARANTINED.to_string();
    }
    TRUSTED.to_string()
}

pub fn for_source(source: &str, explicit: Option<&str>) -> String {
    resolve(source, explicit, quarantine_inference_enabled())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_are_trusted_by_default() {
        assert_eq!(resolve("agent", None, false), TRUSTED);
        assert_eq!(resolve("user", None, false), TRUSTED);
        assert_eq!(resolve("inference", None, false), TRUSTED);
    }

    #[test]
    fn the_policy_only_quarantines_llm_inference() {
        assert_eq!(resolve("inference", None, true), QUARANTINED);
        assert_eq!(
            resolve("agent", None, true),
            TRUSTED,
            "an explicit agent write is not the poisoning vector the policy targets"
        );
        assert_eq!(resolve("user", None, true), TRUSTED);
    }

    #[test]
    fn an_explicit_value_overrides_the_policy_in_both_directions() {
        assert_eq!(resolve("inference", Some(QUARANTINED), false), QUARANTINED);
        assert_eq!(resolve("inference", Some(TRUSTED), true), TRUSTED);
    }

    #[test]
    fn an_unknown_explicit_value_falls_back_to_the_policy() {
        assert_eq!(resolve("inference", Some("nonsense"), true), QUARANTINED);
        assert_eq!(
            resolve("agent", Some("'; DROP TABLE --"), false),
            TRUSTED,
            "an unrecognized value must never reach the DB CHECK"
        );
    }
}
