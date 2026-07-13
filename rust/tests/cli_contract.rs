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

/// The `version` line under `[project]`, without pulling in a TOML parser.
fn pyproject_version(src: &str) -> Option<String> {
    let mut in_project = false;
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_project = t == "[project]";
            continue;
        }
        if !in_project {
            continue;
        }
        if let Some(v) = t
            .strip_prefix("version")
            .and_then(|rest| rest.trim_start().strip_prefix('='))
        {
            return Some(v.trim().trim_matches('"').to_string());
        }
    }
    None
}

/// One release, one number, four files that each hold their own copy of it.
///
/// * `Cargo.toml` — what the release workflow names the binaries after
/// * `package.json` — what npm's postinstall builds its download URL from:
///   `releases/download/v{version}/cuba-memorys-linux-x64`
/// * `pyproject.toml` — the PyPI version (a different series; see below)
/// * `server.json` — what gets published to the MCP Registry, telling the world
///   which npm and PyPI versions to install
///
/// Nothing connected them. `server.json` sat at 0.10.0 while the other three said
/// 0.11.0, and it took reading the publish workflow line by line to notice —
/// tagging would have advertised the previous release to every client discovering
/// this server through the registry.
///
/// PyPI runs its own series because it hit 1.0 first: `1.{minor+2}.{patch}`.
/// Cargo 0.9.3 → PyPI 1.11.3, Cargo 0.11.0 → PyPI 1.13.0. Asserted, not assumed —
/// if the convention ever changes, change it here, deliberately, once.
#[test]
fn every_file_that_holds_a_version_agrees() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root");
    let cargo = env!("CARGO_PKG_VERSION");

    let read = |name: &str| {
        std::fs::read_to_string(root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let json = |name: &str| -> serde_json::Value {
        serde_json::from_str(&read(name)).unwrap_or_else(|e| panic!("{name} must parse: {e}"))
    };

    // npm ── the postinstall download URL is built from this exact string.
    let pkg = json("package.json");
    let npm = pkg["version"].as_str().expect("package.json has a version");
    assert_eq!(
        npm, cargo,
        "package.json ({npm}) vs Cargo.toml ({cargo}): npm's postinstall downloads from \
         releases/download/v{npm}/, an asset the release workflow only builds for the Cargo version"
    );

    // PyPI ── its own series, by convention.
    let py =
        pyproject_version(&read("pyproject.toml")).expect("pyproject.toml has a [project] version");
    let (minor, patch) = {
        let p: Vec<&str> = cargo.split('.').collect();
        (p[1].parse::<u32>().expect("cargo minor"), p[2].to_string())
    };
    let expected_py = format!("1.{}.{}", minor + 2, patch);
    assert_eq!(
        py, expected_py,
        "pyproject.toml ({py}) must follow 1.{{minor+2}}.{{patch}} for Cargo {cargo}"
    );

    // MCP Registry ── tells the world which npm and PyPI versions to install.
    let srv = json("server.json");
    let srv_version = srv["version"].as_str().expect("server.json has a version");
    assert_eq!(
        srv_version, cargo,
        "server.json ({srv_version}) vs Cargo.toml ({cargo}) — this is what the MCP Registry \
         publishes; stale here means the registry advertises the previous release"
    );

    for p in srv["packages"]
        .as_array()
        .expect("server.json has packages")
    {
        let registry = p["registryType"]
            .as_str()
            .or_else(|| p["registry_name"].as_str())
            .expect("package has a registry");
        let got = p["version"].as_str().expect("package has a version");
        let want = match registry {
            "npm" => npm,
            "pypi" => &expected_py,
            other => panic!(
                "server.json declares an unknown registry `{other}` — teach this test about it"
            ),
        };
        assert_eq!(
            got, want,
            "server.json's {registry} entry says {got}, but {registry} will receive {want}"
        );
    }
}
