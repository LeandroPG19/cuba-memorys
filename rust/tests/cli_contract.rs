use std::process::Command;

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

    let pkg = json("package.json");
    let npm = pkg["version"].as_str().expect("package.json has a version");
    assert_eq!(
        npm, cargo,
        "package.json ({npm}) vs Cargo.toml ({cargo}): npm's postinstall downloads from \
         releases/download/v{npm}/, an asset the release workflow only builds for the Cargo version"
    );

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
