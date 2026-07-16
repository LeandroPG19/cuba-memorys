use super::{Symbol, SymbolKind, line_of_byte};
use anyhow::{Context, Result};
use tree_sitter::{Node, Parser};

pub fn extract(file: &str, source: &str) -> Result<(Vec<Symbol>, Vec<String>)> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .context("loading tree-sitter-python grammar")?;
    let tree = parser
        .parse(source, None)
        .context("tree-sitter failed to parse (source too large or internal error)")?;

    let mut symbols = Vec::new();
    let mut imports = Vec::new();
    walk(tree.root_node(), source, file, None, &mut symbols, &mut imports);
    Ok((symbols, imports))
}

fn walk(
    node: Node,
    source: &str,
    file: &str,
    class_ctx: Option<&str>,
    symbols: &mut Vec<Symbol>,
    imports: &mut Vec<String>,
) {
    match node.kind() {
        "function_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let simple_name = text_of(name_node, source).to_string();
                let qualified_name = match class_ctx {
                    Some(c) => format!("{file}::{c}::{simple_name}"),
                    None => format!("{file}::{simple_name}"),
                };
                symbols.push(Symbol {
                    qualified_name,
                    simple_name,
                    kind: SymbolKind::Function,
                    file: file.to_string(),
                    line_start: line_of_byte(source, node.start_byte()),
                    line_end: line_of_byte(source, node.end_byte()),
                    signature: first_line(node, source),
                    calls: collect_calls(node, source),
                });
            }
            return;
        }
        "class_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let simple_name = text_of(name_node, source).to_string();
                symbols.push(Symbol {
                    qualified_name: format!("{file}::{simple_name}"),
                    simple_name: simple_name.clone(),
                    kind: SymbolKind::Class,
                    file: file.to_string(),
                    line_start: line_of_byte(source, node.start_byte()),
                    line_end: line_of_byte(source, node.end_byte()),
                    signature: first_line(node, source),
                    calls: Vec::new(),
                });
                if let Some(body) = node.child_by_field_name("body") {
                    for child in body.children(&mut body.walk()) {
                        walk(child, source, file, Some(&simple_name), symbols, imports);
                    }
                }
            }
            return;
        }
        "import_statement" => {
            for child in node.children(&mut node.walk()) {
                if matches!(child.kind(), "dotted_name" | "aliased_import") {
                    imports.push(dotted_name_text(child, source));
                }
            }
            return;
        }
        "import_from_statement" => {
            if let Some(module) = node.child_by_field_name("module_name") {
                imports.push(dotted_name_text(module, source));
            }
            return;
        }
        _ => {}
    }

    for child in node.children(&mut node.walk()) {
        walk(child, source, file, class_ctx, symbols, imports);
    }
}

fn dotted_name_text(node: Node, source: &str) -> String {
    match node.kind() {
        "aliased_import" => node
            .child_by_field_name("name")
            .map(|n| text_of(n, source).to_string())
            .unwrap_or_else(|| text_of(node, source).to_string()),
        _ => text_of(node, source).to_string(),
    }
}

fn collect_calls(fn_node: Node, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    collect_calls_rec(fn_node, source, &mut out);
    out
}

fn collect_calls_rec(node: Node, source: &str, out: &mut Vec<String>) {
    if node.kind() == "call"
        && let Some(func) = node.child_by_field_name("function")
    {
        let callee = match func.kind() {
            "identifier" => Some(text_of(func, source).to_string()),
            "attribute" => func
                .child_by_field_name("attribute")
                .map(|f| text_of(f, source).to_string()),
            _ => None,
        };
        if let Some(name) = callee {
            out.push(name);
        }
    }
    for child in node.children(&mut node.walk()) {
        collect_calls_rec(child, source, out);
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
    fn top_level_function_is_extracted() {
        let (symbols, _) = extract("app.py", "def greet(name):\n    print(f'hi {name}')\n").unwrap();
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].qualified_name, "app.py::greet");
        assert_eq!(symbols[0].kind, SymbolKind::Function);
    }

    #[test]
    fn method_inside_class_is_qualified_by_its_class() {
        let src = "class Foo:\n    def bar(self):\n        pass\n";
        let (symbols, _) = extract("app.py", src).unwrap();
        let names: Vec<_> = symbols.iter().map(|s| s.qualified_name.as_str()).collect();
        assert!(names.contains(&"app.py::Foo"));
        assert!(names.contains(&"app.py::Foo::bar"));
    }

    #[test]
    fn calls_are_collected_including_attribute_calls() {
        let src = "def a():\n    b()\n    self.c()\n";
        let (symbols, _) = extract("app.py", src).unwrap();
        let a = symbols.iter().find(|s| s.simple_name == "a").unwrap();
        assert!(a.calls.contains(&"b".to_string()));
        assert!(a.calls.contains(&"c".to_string()));
    }

    #[test]
    fn plain_and_from_imports_are_both_collected() {
        let (_, imports) = extract("app.py", "import os\nfrom pathlib import Path\n").unwrap();
        assert!(imports.contains(&"os".to_string()));
        assert!(imports.contains(&"pathlib".to_string()));
    }

    #[test]
    fn syntax_errors_do_not_panic_the_extractor() {
        let result = extract("app.py", "def broken(:\n    this is not python");
        assert!(result.is_ok());
    }
}
