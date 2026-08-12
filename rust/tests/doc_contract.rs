use std::collections::BTreeSet;

fn read(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn counts_claimed(text: &str) -> BTreeSet<usize> {
    fn number(word: &str) -> Option<usize> {
        word.trim_matches(|c: char| !c.is_ascii_digit())
            .parse::<usize>()
            .ok()
    }

    let mut found = BTreeSet::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if !word.starts_with("tools") {
            continue;
        }
        if let Some(n) = i.checked_sub(1).and_then(|j| number(words[j])) {
            found.insert(n);
        }
        if words.get(i + 1).is_some_and(|w| *w == "of")
            && let Some(n) = words.get(i + 2).and_then(|w| number(w))
        {
            found.insert(n);
        }
    }
    found
}

#[test]
fn every_tool_count_in_the_docs_is_one_the_code_can_produce() {
    let full = cuba_memorys::constants::tools_for("full").len();
    let lean = cuba_memorys::constants::tools_for("lean").len();

    assert!(
        full > lean && lean > 0,
        "the profiles have to differ for this contract to mean anything: full={full} lean={lean}"
    );

    let readme = read("README.md");
    let crate_readme = read("rust/README.md");
    let schemas = read("rust/src/constants.rs");
    let described = schemas
        .split("fn meta_tool_defs")
        .nth(1)
        .expect("meta_tool_defs is where the cuba_tools description lives");

    for (source, text) in [
        ("README.md", readme.as_str()),
        ("rust/README.md", crate_readme.as_str()),
        ("cuba_tools", described),
    ] {
        for n in counts_claimed(text) {
            assert!(
                n == full || n == lean,
                "{source} claims «{n} tools», but the code produces {full} in the full profile \
                 and {lean} in lean. Three different numbers were in circulation at once — the \
                 README said 28 in one place and 30 in two others while the cuba_tools \
                 description shipped to every model said 29 — because each was written by hand \
                 at a different time. This assertion is the only thing that makes adding a tool \
                 and forgetting a document impossible."
            );
        }
    }
}

#[test]
fn the_docs_do_not_promise_commands_the_binary_does_not_have() {
    let readme = read("README.md");
    let known: BTreeSet<&str> = cuba_memorys::cli::COMMANDS.iter().copied().collect();

    let mut promised = BTreeSet::new();
    let mut inside_fence = false;
    for line in readme.lines() {
        if line.trim_start().starts_with("```") {
            inside_fence = !inside_fence;
            continue;
        }
        let spans: Vec<&str> = if inside_fence {
            vec![line]
        } else {
            line.split('`').skip(1).step_by(2).collect()
        };
        for span in spans {
            for rest in span.split("cuba-memorys ").skip(1) {
                let word = rest.split_whitespace().next().unwrap_or("");
                if word.starts_with('-') {
                    continue;
                }
                let word = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
                if !word.is_empty() && word.chars().all(|c| c.is_ascii_lowercase() || c == '-') {
                    promised.insert(word.to_string());
                }
            }
        }
    }
    assert!(
        promised.len() > 3,
        "the extractor found almost no commands in the README, so a green result would mean \
         nothing. Found: {promised:?}"
    );

    let unknown: Vec<&String> = promised
        .iter()
        .filter(|w| !known.contains(w.as_str()))
        .collect();

    assert!(
        unknown.is_empty(),
        "the README tells the reader to run `cuba-memorys <x>` for commands that do not exist. \
         A copied-and-pasted line that errors out is how a reader decides the whole document is \
         stale. Known commands: {known:?}. Not found: {unknown:?}"
    );
}

#[test]
fn a_credential_compiled_into_the_binary_is_disclosed_in_security_md() {
    let security = read("SECURITY.md");
    let mut compiled: BTreeSet<String> = BTreeSet::new();
    let mut stack = vec![std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")];

    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("src/ is readable").flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let body = std::fs::read_to_string(&path).expect("readable");
            for line in body.lines() {
                let upper = line.to_uppercase();
                if !upper.contains("PASSWORD") && !upper.contains("PASSWD") {
                    continue;
                }
                for literal in line.split('"').skip(1).step_by(2) {
                    if literal.len() >= 6
                        && literal.chars().any(|c| c.is_ascii_digit())
                        && !literal.contains(' ')
                        && !literal.contains("://")
                        && !literal.contains('{')
                    {
                        compiled.insert(literal.to_string());
                    }
                }
            }
        }
    }

    assert!(
        !compiled.is_empty(),
        "the scan found no compiled credential at all. That is either very good news or a \
         broken scan, and today it is the second: setup.rs carries the container defaults. A \
         check that cannot find what it knows is there proves nothing about what it cannot see"
    );

    let undisclosed: Vec<&String> = compiled
        .iter()
        .filter(|literal| !security.contains(literal.as_str()))
        .collect();

    assert!(
        undisclosed.is_empty(),
        "SECURITY.md is explicit that the fallback container password and the app-role password \
         are compiled into setup.rs, and naming them is what lets an operator decide to rotate \
         them. A credential that gets compiled in without appearing there is one nobody knows to \
         rotate — and an earlier version of this very test asserted the opposite claim, that no \
         credentials are compiled in, then passed three times in a row without ever running its \
         body because SECURITY.md never contained the phrase it was looking for. Undisclosed: \
         {undisclosed:?}"
    );
}
