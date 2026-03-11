//! Python code validator using tree-sitter AST analysis.
//!
//! Validates agent-generated Python code by parsing with tree-sitter-python
//! and scanning the CST for blocked imports, calls, and patterns.
//! Error-tolerant: produces partial trees on broken code.

use pyo3::prelude::*;
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};

/// Blocked module names — security-critical stdlib modules.
const BLOCKED_MODULES: &[&str] = &[
    "os", "sys", "subprocess", "shutil", "ctypes", "importlib",
    "socket", "http", "ftplib", "smtplib", "xmlrpc",
    "multiprocessing", "threading", "signal", "resource",
    "code", "codeop", "pathlib", "glob", "tempfile",
    "pickle", "shelve", "builtins",
];

/// Blocked function/method calls.
const BLOCKED_CALLS: &[&str] = &[
    "exec", "eval", "compile", "__import__", "breakpoint",
    "open", "getattr", "setattr", "delattr", "globals", "locals",
    // Indirect bypass vectors (Audit3 F-01):
    "vars", "dir", "chr", "type", "hasattr",
];

/// Blocked dunder attributes — kills __class__.__mro__.__subclasses__ chains.
const BLOCKED_DUNDERS: &[&str] = &[
    "__class__", "__bases__", "__mro__", "__subclasses__",
    "__globals__", "__builtins__", "__import__", "__init__",
    "__dict__", "__getattr__", "__setattr__", "__delattr__",
    "__code__", "__func__", "__self__", "__module__",
    "__qualname__", "__wrapped__", "__loader__", "__spec__",
];

/// S-expression queries for tree-sitter-python.
const IMPORT_QUERY: &str = r#"
(import_statement name: (dotted_name (identifier) @mod))
(import_from_statement module_name: (dotted_name (identifier) @mod))
"#;

const CALL_QUERY: &str = r#"
(call function: (identifier) @fn)
(call function: (attribute attribute: (identifier) @method))
"#;

/// Query to find all attribute access (obj.attr) patterns.
const ATTR_QUERY: &str = r#"
(attribute attribute: (identifier) @attr)
"#;

/// Validation result returned to Python.
#[pyclass]
#[derive(Clone)]
pub struct ValidationResult {
    #[pyo3(get)]
    pub valid: bool,
    #[pyo3(get)]
    pub errors: Vec<String>,
}

/// Validate Python code for security issues.
///
/// Returns ValidationResult with `valid=true` if no issues found,
/// or `valid=false` with list of error messages.
pub fn validate_python_code(code: &str) -> ValidationResult {
    let mut errors = Vec::new();

    // 1. Parse with tree-sitter
    let mut parser = Parser::new();
    let language = tree_sitter_python::LANGUAGE;
    parser.set_language(&language.into()).expect("tree-sitter-python language");

    let tree = match parser.parse(code, None) {
        Some(t) => t,
        None => {
            return ValidationResult {
                valid: false,
                errors: vec!["Failed to parse Python code".into()],
            };
        }
    };

    let root = tree.root_node();

    // Check for syntax errors in the tree
    if root.has_error() {
        errors.push("Code contains syntax errors (partial analysis)".into());
    }

    let source = code.as_bytes();

    // 2. Check imports
    if let Ok(query) = Query::new(&language.into(), IMPORT_QUERY) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, root, source);
        while let Some(m) = matches.next() {
            for capture in m.captures {
                let text = capture.node.utf8_text(source).unwrap_or("");
                let top_module = text.split('.').next().unwrap_or(text);
                if BLOCKED_MODULES.contains(&top_module) {
                    errors.push(format!(
                        "Blocked import: '{}' — module '{}' is not allowed (line {})",
                        text,
                        top_module,
                        capture.node.start_position().row + 1,
                    ));
                }
            }
        }
    }

    // 3. Check function calls
    if let Ok(query) = Query::new(&language.into(), CALL_QUERY) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, root, source);
        while let Some(m) = matches.next() {
            for capture in m.captures {
                let text = capture.node.utf8_text(source).unwrap_or("");
                if BLOCKED_CALLS.contains(&text) {
                    errors.push(format!(
                        "Blocked call: '{}()' is not allowed (line {})",
                        text,
                        capture.node.start_position().row + 1,
                    ));
                }
            }
        }
    }

    // 4. Check dunder attribute access (Audit3 F-01: __class__.__mro__.__subclasses__ chains)
    if let Ok(query) = Query::new(&language.into(), ATTR_QUERY) {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, root, source);
        while let Some(m) = matches.next() {
            for capture in m.captures {
                let text = capture.node.utf8_text(source).unwrap_or("");
                if BLOCKED_DUNDERS.contains(&text) {
                    errors.push(format!(
                        "Blocked dunder access: '{}' — not allowed (line {})",
                        text,
                        capture.node.start_position().row + 1,
                    ));
                }
            }
        }
    }

    ValidationResult {
        valid: errors.is_empty(),
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_code_passes() {
        let r = validate_python_code("import json\nresult = json.dumps({'a': 1})");
        assert!(r.valid, "errors: {:?}", r.errors);
    }

    #[test]
    fn test_blocks_os_import() {
        let r = validate_python_code("import os\nos.listdir('/')");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("os")));
    }

    #[test]
    fn test_blocks_from_import() {
        let r = validate_python_code("from subprocess import run");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("subprocess")));
    }

    #[test]
    fn test_blocks_exec() {
        let r = validate_python_code("exec('x = 1')");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("exec")));
    }

    #[test]
    fn test_blocks_eval() {
        let r = validate_python_code("eval('1+1')");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("eval")));
    }

    #[test]
    fn test_blocks_open() {
        let r = validate_python_code("f = open('/etc/passwd')");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("open")));
    }

    #[test]
    fn test_syntax_error_still_scans() {
        let r = validate_python_code("import os\ndef f(:\n  pass");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("os")));
    }

    #[test]
    fn test_multiple_violations() {
        let r = validate_python_code("import os\nimport subprocess\nexec('x')");
        assert!(!r.valid);
        assert!(r.errors.len() >= 3);
    }

    #[test]
    fn test_allowed_modules() {
        let r = validate_python_code(
            "import json\nimport math\nimport re\nimport collections\nimport itertools"
        );
        assert!(r.valid, "errors: {:?}", r.errors);
    }

    // ── Dunder bypass regression tests (Audit3 F-01) ────────

    #[test]
    fn test_blocks_dunder_class() {
        let r = validate_python_code("x = ().__class__");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("__class__")));
    }

    #[test]
    fn test_blocks_dunder_mro() {
        let r = validate_python_code("x = ().__class__.__mro__");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("__mro__")));
    }

    #[test]
    fn test_blocks_dunder_subclasses() {
        let r = validate_python_code("x = object.__subclasses__()");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("__subclasses__")));
    }

    #[test]
    fn test_blocks_dunder_globals() {
        let r = validate_python_code("x = func.__globals__");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("__globals__")));
    }

    #[test]
    fn test_blocks_dunder_builtins() {
        let r = validate_python_code("x = obj.__builtins__");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("__builtins__")));
    }

    #[test]
    fn test_blocks_dunder_dict() {
        let r = validate_python_code("x = cls.__dict__");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("__dict__")));
    }

    #[test]
    fn test_blocks_full_exploit_chain() {
        let code = "subs = ().__class__.__mro__[-1].__subclasses__()\nfor s in subs:\n    x = s.__init__.__globals__";
        let r = validate_python_code(code);
        assert!(!r.valid);
        assert!(r.errors.len() >= 3, "Expected 3+ errors, got {:?}", r.errors);
    }

    #[test]
    fn test_blocks_vars_dir_chr_type() {
        let r = validate_python_code("vars()\ndir()\nchr(65)\ntype(x)");
        assert!(!r.valid);
        assert!(r.errors.len() >= 4, "Expected 4+ blocked calls, got {:?}", r.errors);
    }

    #[test]
    fn test_blocks_hasattr() {
        let r = validate_python_code("hasattr(obj, 'x')");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("hasattr")));
    }

    #[test]
    fn test_safe_attribute_access_allowed() {
        // Normal attribute access like obj.name should be fine
        let r = validate_python_code("x = obj.name\ny = obj.value");
        assert!(r.valid, "errors: {:?}", r.errors);
    }
}
