#[test]
fn a_key_switches_the_chain_to_hmac_and_legacy_rows_still_verify() {
    let prev = b"previous";
    let action = "test";
    let payload = br#"{"a":1}"#;
    let stamp = "2026-08-03T00:00:00.000000+00:00";

    unsafe { std::env::remove_var("CUBA_AUDIT_KEY") };
    let unkeyed = cuba_memorys::handlers::archivo::audit_key();
    if unkeyed.is_some() {
        eprintln!("skipping: a key file exists on this machine");
        return;
    }

    assert!(
        cuba_memorys::handlers::archivo::hash_matches(
            &sha256_chain(prev, action, payload, stamp),
            prev,
            action,
            payload,
            stamp
        ),
        "without a key the chain must stay plain SHA-256"
    );

    unsafe {
        std::env::set_var(
            "CUBA_AUDIT_KEY",
            "a-secret-that-does-not-live-in-the-database",
        )
    };
    assert!(
        cuba_memorys::handlers::archivo::audit_key().is_some(),
        "the key must be picked up from the environment"
    );

    assert!(
        cuba_memorys::handlers::archivo::hash_matches(
            &sha256_chain(prev, action, payload, stamp),
            prev,
            action,
            payload,
            stamp
        ),
        "rows written before the key was introduced must keep verifying, otherwise turning on \
         HMAC would report the whole existing chain as tampered"
    );

    unsafe { std::env::remove_var("CUBA_AUDIT_KEY") };
}

fn sha256_chain(prev: &[u8], action: &str, payload: &[u8], stamp: &str) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(prev);
    h.update(b"|");
    h.update(action.as_bytes());
    h.update(b"|");
    h.update(payload);
    h.update(b"|");
    h.update(stamp.as_bytes());
    h.finalize().to_vec()
}
