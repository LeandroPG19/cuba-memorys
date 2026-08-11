use cuba_memorys::handlers::archivo::{audit_key, hash_matches};

const PREV: &[u8] = b"previous";
const ACTION: &str = "test";
const PAYLOAD: &[u8] = br#"{"a":1}"#;
const STAMP: &str = "2026-08-03T00:00:00.000000+00:00";

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

#[test]
fn the_key_is_resolved_from_env_then_disk_and_legacy_sha256_rows_keep_verifying() {
    let home = std::env::temp_dir().join(format!("cuba-audit-hmac-{}", std::process::id()));
    let key_file = home.join(".cache/cuba-memorys/audit_key");
    std::fs::create_dir_all(key_file.parent().expect("the key path has a parent"))
        .expect("a scratch HOME must be creatable, otherwise this test cannot be deterministic");
    let _ = std::fs::remove_file(&key_file);
    unsafe { std::env::set_var("HOME", &home) };
    unsafe { std::env::remove_var("CUBA_AUDIT_KEY") };

    assert!(
        audit_key().is_none(),
        "with no variable and no key file there is no key; this assertion is what replaces \
         the old silent early return, which handed a green result to every machine that \
         happened to have ~/.cache/cuba-memorys/audit_key"
    );

    assert!(
        hash_matches(
            &sha256_chain(PREV, ACTION, PAYLOAD, STAMP),
            PREV,
            ACTION,
            PAYLOAD,
            STAMP
        ),
        "without a key the chain is plain SHA-256 over prev|action|payload|timestamp; the \
         framing is spelled out here because archivo.rs exposes no public hash producer"
    );

    assert!(
        !hash_matches(
            &sha256_chain(PREV, ACTION, PAYLOAD, STAMP),
            PREV,
            ACTION,
            br#"{"a":2}"#,
            STAMP
        ),
        "editing the payload of a stored row must break its hash — a verifier that accepts \
         a rewritten payload turns the append-only log into decoration"
    );

    unsafe { std::env::set_var("CUBA_AUDIT_KEY", "   ") };
    assert!(
        audit_key().is_none(),
        "a blank variable is an operator who meant to set a key and did not; treating the \
         spaces as the key would sign the whole chain with a guessable secret"
    );

    unsafe {
        std::env::set_var(
            "CUBA_AUDIT_KEY",
            "a-secret-that-does-not-live-in-the-database",
        )
    };
    assert_eq!(
        audit_key().as_deref(),
        Some(b"a-secret-that-does-not-live-in-the-database".as_slice()),
        "the key must be picked up from the environment"
    );

    assert!(
        hash_matches(
            &sha256_chain(PREV, ACTION, PAYLOAD, STAMP),
            PREV,
            ACTION,
            PAYLOAD,
            STAMP
        ),
        "rows written before the key was introduced must keep verifying, otherwise turning on \
         HMAC would report the whole existing chain as tampered"
    );

    unsafe { std::env::remove_var("CUBA_AUDIT_KEY") };
    std::fs::write(&key_file, "key-from-the-file\n").expect("writing the scratch key file");
    assert_eq!(
        audit_key().as_deref(),
        Some(b"key-from-the-file".as_slice()),
        "the file under HOME is the fallback, and its trailing newline is not part of the \
         secret — keeping it would make the CLI-written key differ from the same key typed \
         into the environment"
    );

    unsafe { std::env::set_var("CUBA_AUDIT_KEY", "key-from-the-environment") };
    assert_eq!(
        audit_key().as_deref(),
        Some(b"key-from-the-environment".as_slice()),
        "the environment outranks the file: a service unit that overrides the key must not \
         be silently ignored because a stale file was left in the cache directory"
    );

    unsafe { std::env::remove_var("CUBA_AUDIT_KEY") };
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_key_actually_produces_a_different_hash_than_no_key() {
    use cuba_memorys::handlers::archivo::compute_hash;

    const PREV: &[u8] = b"\x01\x02\x03";
    const ACTION: &str = "append";
    const PAYLOAD: &[u8] = b"{\"a\":1}";
    const STAMP: &str = "2026-08-11T00:00:00.000000+00:00";

    unsafe { std::env::set_var("HOME", std::env::temp_dir().join("cuba-hmac-probe")) };
    unsafe { std::env::remove_var("CUBA_AUDIT_KEY") };
    let plain = compute_hash(PREV, ACTION, PAYLOAD, STAMP);

    unsafe { std::env::set_var("CUBA_AUDIT_KEY", "a-real-secret") };
    let keyed = compute_hash(PREV, ACTION, PAYLOAD, STAMP);

    assert_ne!(
        keyed, plain,
        "with a key set the chain must be HMAC, not SHA-256 with extra steps. Delete the \
         HMAC branch entirely and the old test still passed, because it only ever checked \
         that a legacy row keeps verifying"
    );

    unsafe { std::env::remove_var("CUBA_AUDIT_KEY") };
}
