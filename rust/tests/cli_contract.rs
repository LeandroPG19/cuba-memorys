//! The CLI's contract with the outside world. No database required — that is
//! precisely what half of these assert.
//!
//! Every one of these pins a bug that shipped:
//!
//!   * `--version` fell through to the server, so asking the binary what version
//!     it was connected to PostgreSQL and applied migrations. The one command a
//!     person runs because they do not yet trust what they have installed.
//!   * `--help` did the same, which is why nothing documented the 13 subcommands.
//!   * A typo (`doctro`) launched the MCP server on a stdio socket nobody was
//!     speaking to — indistinguishable, from the outside, from a hang.
//!   * Cargo's version and package.json's version build the GitHub Releases URL
//!     that npm's postinstall downloads from. Let them drift and every
//!     `npm install cuba-memorys` 404s — with no local test able to see it.

use std::process::Command;

/// A DATABASE_URL that cannot possibly connect. Any command that reaches the
/// database with this set will fail or hang; a command that passes with it set
/// has proven it never went near one.
const DEAD_DB: &str = "postgresql://nobody@127.0.0.1:1/nothing";

fn run(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_cuba-memorys"))
        .args(args)
        .env("DATABASE_URL", DEAD_DB)
        .output()
        .expect("binary runs");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn version_is_inert_and_matches_the_crate() {
    let (stdout, stderr, code) = run(&["--version"]);

    assert_eq!(
        code, 0,
        "--version must exit 0, got {code}\nstderr: {stderr}"
    );
    assert_eq!(
        stdout.trim(),
        format!("cuba-memorys {}", env!("CARGO_PKG_VERSION")),
        "--version must print exactly the crate version on stdout"
    );
    // The real assertion: it got here with DATABASE_URL pointing at a closed port.
    assert!(
        !stderr.contains("connected to PostgreSQL") && !stderr.contains("migrations"),
        "--version must not touch the database — it ran migrations before this test existed.\nstderr: {stderr}"
    );

    for alias in ["-V", "version"] {
        let (out, _, code) = run(&[alias]);
        assert_eq!(code, 0, "`{alias}` must work too");
        assert!(
            out.contains(env!("CARGO_PKG_VERSION")),
            "`{alias}` prints the version"
        );
    }
}

#[test]
fn help_documents_the_command_surface() {
    let (stdout, _, code) = run(&["--help"]);
    assert_eq!(code, 0, "--help must exit 0");

    // Every subcommand main.rs dispatches on. If you add one, add it to the help
    // — that is the whole point of this assertion.
    for cmd in [
        "search",
        "save",
        "delete",
        "export",
        "dashboard",
        "doctor",
        "recall",
        "reembed",
        "calibrate",
        "link",
        "skills",
        "eval",
        "setup",
    ] {
        assert!(stdout.contains(cmd), "--help must document `{cmd}`");
    }

    let (short, _, code) = run(&["-h"]);
    assert_eq!(code, 0);
    assert_eq!(short, stdout, "-h and --help must agree");
}

#[test]
fn an_unknown_argument_is_an_error_not_a_server_launch() {
    for typo in ["doctro", "--verison", "sarch"] {
        let (_, stderr, code) = run(&[typo]);
        assert_eq!(
            code, 2,
            "`{typo}` must exit 2 (usage error), not start the MCP server"
        );
        assert!(
            stderr.contains(typo),
            "the error must name the offending argument, so the typo is obvious"
        );
    }
}

/// npm's postinstall builds its download URL from package.json's version:
///   releases/download/v{version}/cuba-memorys-linux-x64
///
/// The release workflow builds that asset from Cargo.toml's version. Two files,
/// one number, and nothing but this test connecting them. Bump one without the
/// other and every `npm install` 404s on a tag that does not exist.
#[test]
fn cargo_and_package_json_agree_on_the_version() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root");
    let pkg =
        std::fs::read_to_string(root.join("package.json")).expect("package.json at repo root");
    let pkg: serde_json::Value = serde_json::from_str(&pkg).expect("package.json parses");
    let npm_version = pkg["version"].as_str().expect("package.json has a version");

    assert_eq!(
        npm_version,
        env!("CARGO_PKG_VERSION"),
        "package.json ({npm_version}) and Cargo.toml ({}) must match — npm's postinstall \
         downloads from releases/download/v{npm_version}/, which the release workflow only \
         creates for the Cargo version",
        env!("CARGO_PKG_VERSION")
    );
}
