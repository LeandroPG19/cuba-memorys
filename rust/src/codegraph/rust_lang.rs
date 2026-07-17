use super::{Symbol, SymbolKind, line_of_byte};
use anyhow::{Context, Result};
use tree_sitter::{Node, Parser};

pub fn extract(file: &str, source: &str) -> Result<(Vec<Symbol>, Vec<String>)> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .context("loading tree-sitter-rust grammar")?;
    let tree = parser
        .parse(source, None)
        .context("tree-sitter failed to parse (source too large or internal error)")?;

    let mut symbols = Vec::new();
    let mut imports = Vec::new();
    walk(
        tree.root_node(),
        source,
        file,
        None,
        &mut symbols,
        &mut imports,
    );
    Ok((symbols, imports))
}

fn walk(
    node: Node,
    source: &str,
    file: &str,
    impl_type: Option<&str>,
    symbols: &mut Vec<Symbol>,
    imports: &mut Vec<String>,
) {
    match node.kind() {
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let simple_name = text_of(name_node, source).to_string();
                let qualified_name = match impl_type {
                    Some(t) => format!("{file}::{t}::{simple_name}"),
                    None => format!("{file}::{simple_name}"),
                };
                let line_start = line_of_byte(source, node.start_byte());
                let line_end = line_of_byte(source, node.end_byte());
                let signature = first_line(node, source);
                let calls = collect_calls(node, source);
                symbols.push(Symbol {
                    qualified_name,
                    simple_name,
                    kind: SymbolKind::Function,
                    file: file.to_string(),
                    line_start,
                    line_end,
                    signature,
                    calls,
                });
            }
            // Fall through to the shared recursion below so nested fn/struct
            // items in the body are extracted as their own symbols too.
            // collect_calls_rec stops at nested function_item boundaries, so
            // their calls aren't double-attributed to this outer function.
        }
        "struct_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let simple_name = text_of(name_node, source).to_string();
                symbols.push(Symbol {
                    qualified_name: format!("{file}::{simple_name}"),
                    simple_name,
                    kind: SymbolKind::Struct,
                    file: file.to_string(),
                    line_start: line_of_byte(source, node.start_byte()),
                    line_end: line_of_byte(source, node.end_byte()),
                    signature: first_line(node, source),
                    calls: Vec::new(),
                });
            }
        }
        "use_declaration" => {
            if let Some(path) = use_path(node, source) {
                imports.push(path);
            }
            return;
        }
        "impl_item" => {
            let ty = node
                .child_by_field_name("type")
                .map(|n| text_of(n, source).to_string());
            for child in node.children(&mut node.walk()) {
                walk(child, source, file, ty.as_deref(), symbols, imports);
            }
            return;
        }
        _ => {}
    }

    for child in node.children(&mut node.walk()) {
        walk(child, source, file, impl_type, symbols, imports);
    }
}

fn collect_calls(fn_node: Node, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    collect_calls_rec(fn_node, source, &mut out);
    out
}

fn collect_calls_rec(node: Node, source: &str, out: &mut Vec<String>) {
    if node.kind() == "call_expression"
        && let Some(func) = node.child_by_field_name("function")
    {
        let callee = match func.kind() {
            "identifier" => Some(text_of(func, source).to_string()),
            "field_expression" => func
                .child_by_field_name("field")
                .map(|f| text_of(f, source).to_string()),
            "scoped_identifier" => func
                .child_by_field_name("name")
                .map(|f| text_of(f, source).to_string()),
            _ => None,
        };
        if let Some(name) = callee {
            out.push(name);
        }
    }
    for child in node.children(&mut node.walk()) {
        // Nested fn items are extracted as their own symbols with their own
        // calls; don't also attribute their internal calls to the enclosing fn.
        if child.kind() == "function_item" {
            continue;
        }
        collect_calls_rec(child, source, out);
    }
}

fn use_path(node: Node, source: &str) -> Option<String> {
    let arg = node.child_by_field_name("argument")?;
    Some(flatten_use_tree(arg, source))
}

fn flatten_use_tree(node: Node, source: &str) -> String {
    match node.kind() {
        "scoped_identifier" | "identifier" | "self" | "crate" | "super" => {
            text_of(node, source).to_string()
        }
        "use_as_clause" => node
            .child_by_field_name("path")
            .map(|p| flatten_use_tree(p, source))
            .unwrap_or_default(),
        "scoped_use_list" | "use_list" => node
            .child_by_field_name("path")
            .map(|p| flatten_use_tree(p, source))
            .unwrap_or_default(),
        _ => text_of(node, source).to_string(),
    }
}

fn text_of<'a>(node: Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

fn first_line(node: Node, source: &str) -> String {
    text_of(node, source)
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegraph::SymbolKind;

    #[test]
    fn top_level_function_is_extracted_with_a_bare_qualified_name() {
        let (symbols, _) = extract(
            "lib.rs",
            "fn greet(name: &str) { println!(\"hi {name}\"); }",
        )
        .unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].qualified_name, "lib.rs::greet");
        assert_eq!(symbols[0].simple_name, "greet");
        assert_eq!(symbols[0].kind, SymbolKind::Function);
    }

    #[test]
    fn method_inside_impl_is_qualified_by_its_type() {
        let src = "struct Foo; impl Foo { fn bar(&self) {} }";
        let (symbols, _) = extract("lib.rs", src).unwrap();
        let names: Vec<_> = symbols.iter().map(|s| s.qualified_name.as_str()).collect();
        assert!(names.contains(&"lib.rs::Foo"));
        assert!(names.contains(&"lib.rs::Foo::bar"));
    }

    #[test]
    fn calls_are_collected_including_method_calls() {
        let src = "fn a() { b(); self.c(); }";
        let (symbols, _) = extract("lib.rs", src).unwrap();
        let a = symbols.iter().find(|s| s.simple_name == "a").unwrap();
        assert!(a.calls.contains(&"b".to_string()));
        assert!(a.calls.contains(&"c".to_string()));
    }

    #[test]
    fn nested_fn_is_extracted_and_does_not_inherit_the_outer_functions_calls() {
        let src = "fn outer() { fn inner() { helper_call(); } inner(); }";
        let (symbols, _) = extract("lib.rs", src).unwrap();
        let names: Vec<_> = symbols.iter().map(|s| s.qualified_name.as_str()).collect();
        assert!(names.contains(&"lib.rs::outer"));
        assert!(names.contains(&"lib.rs::inner"));

        let outer = symbols.iter().find(|s| s.simple_name == "outer").unwrap();
        assert_eq!(outer.calls, vec!["inner".to_string()]);

        let inner = symbols.iter().find(|s| s.simple_name == "inner").unwrap();
        assert_eq!(inner.calls, vec!["helper_call".to_string()]);
    }

    #[test]
    fn use_declarations_are_flattened_to_their_full_path() {
        let (_, imports) = extract("lib.rs", "use crate::db::create_pool;").unwrap();
        assert_eq!(imports, vec!["crate::db::create_pool".to_string()]);
    }

    #[test]
    fn syntax_errors_do_not_panic_the_extractor() {
        let result = extract("lib.rs", "fn broken( {{{ not valid rust at all");
        assert!(
            result.is_ok(),
            "tree-sitter recovers from syntax errors instead of failing the whole file"
        );
    }
}
