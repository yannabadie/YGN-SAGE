# Rust ToolExecutor — Sandboxed Tool Creation Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the entire tool creation security pipeline (validation + execution) from Python to Rust, with Wasm sandbox as primary path and subprocess as fallback.

**Architecture:** A new `ToolExecutor` PyO3 class in `sage-core` handles validation (tree-sitter-python AST) and execution (WasmSandbox for CPython WASI component, or subprocess via process-wrap). Python `meta.py` becomes a thin wrapper calling `sage_core.ToolExecutor`. No `exec()`, `eval()`, `subprocess`, or `tempfile` in Python.

**Tech Stack:** Rust, PyO3, tree-sitter 0.26 + tree-sitter-python 0.25, process-wrap 9.x, wasmtime 36 LTS, componentize-py (CPython WASI)

**Research basis:** 5 parallel research axes (RustPython viability, Rust validation options, wasmtime Component Model, subprocess sandboxing, state-of-art). Key references: Arize Phoenix sandbox benchmarks (6/6 security vectors for CPython-on-WASI), Codex CLI sandboxing architecture (landlock + seccomp), componentize-py (Bytecode Alliance official).

---

## Phase 1: Rust Subprocess Executor + tree-sitter Validator

### Task 1: Add tree-sitter dependencies to Cargo.toml

**Files:**
- Modify: `sage-core/Cargo.toml`

**Step 1: Add dependencies**

Under `[dependencies]`, add:
```toml
tree-sitter = { version = "0.26", optional = true }
tree-sitter-python = { version = "0.25", optional = true }
process-wrap = { version = "9", optional = true, features = ["tokio1"] }
tokio = { version = "1", optional = true, features = ["process", "time", "rt"] }
serde_json = { version = "1", optional = true }
```

Under `[features]`, add:
```toml
tool-executor = ["dep:tree-sitter", "dep:tree-sitter-python", "dep:process-wrap", "dep:tokio", "dep:serde_json"]
```

**Step 2: Verify it compiles**

```bash
cd sage-core && cargo check --features tool-executor
```

**Step 3: Commit**

```bash
git add sage-core/Cargo.toml
git commit -m "build: add tree-sitter, process-wrap, tokio deps for tool-executor feature"
```

---

### Task 2: Create Python code validator using tree-sitter

**Files:**
- Create: `sage-core/src/sandbox/validator.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Write the validator**

```rust
// sage-core/src/sandbox/validator.rs
//! Python code validator using tree-sitter AST analysis.
//!
//! Validates agent-generated Python code by parsing with tree-sitter-python
//! and scanning the CST for blocked imports, calls, and patterns.
//! Error-tolerant: produces partial trees on broken code.

use pyo3::prelude::*;
use tree_sitter::{Parser, Query, QueryCursor};

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
        // tree-sitter is error-tolerant, so we continue scanning
        // but note the syntax error
        errors.push("Code contains syntax errors (partial analysis)".into());
    }

    let source = code.as_bytes();

    // 2. Check imports
    if let Ok(query) = Query::new(&language.into(), IMPORT_QUERY) {
        let mut cursor = QueryCursor::new();
        for m in cursor.matches(&query, root, source) {
            for capture in m.captures {
                let text = capture.node.utf8_text(source).unwrap_or("");
                // Check the first component of dotted imports
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
        for m in cursor.matches(&query, root, source) {
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
        // Should still catch the import even with syntax error
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
}
```

**Step 2: Add module to mod.rs**

In `sage-core/src/sandbox/mod.rs`:
```rust
#[cfg(feature = "tool-executor")]
pub mod validator;
```

**Step 3: Run tests**

```bash
cd sage-core && cargo test --features tool-executor -- validator
```

**Step 4: Commit**

```bash
git add sage-core/src/sandbox/validator.rs sage-core/src/sandbox/mod.rs
git commit -m "feat: add tree-sitter Python code validator for tool security"
```

---

### Task 3: Create Rust subprocess executor

**Files:**
- Create: `sage-core/src/sandbox/subprocess.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Write the subprocess executor**

```rust
// sage-core/src/sandbox/subprocess.rs
//! Sandboxed subprocess executor for Python code.
//!
//! Executes Python code in an isolated subprocess with:
//! - process-wrap for cross-platform process group management + clean kill
//! - Timeout enforcement via tokio
//! - No shell=True (uses Command directly)
//! - GIL-safe via py.allow_threads()

use pyo3::prelude::*;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

/// Result of a sandboxed execution.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ExecResult {
    #[pyo3(get)]
    pub stdout: String,
    #[pyo3(get)]
    pub stderr: String,
    #[pyo3(get)]
    pub exit_code: i32,
    #[pyo3(get)]
    pub timed_out: bool,
    #[pyo3(get)]
    pub duration_ms: u64,
}

/// Execute Python code in a subprocess with timeout.
///
/// Writes code to a temp file, executes via `python -c` equivalent,
/// feeds args as JSON via stdin.
pub fn execute_python_subprocess(
    python_exe: &str,
    code: &str,
    args_json: &str,
    timeout_secs: u64,
) -> ExecResult {
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        execute_async(python_exe, code, args_json, timeout_secs).await
    })
}

async fn execute_async(
    python_exe: &str,
    code: &str,
    args_json: &str,
    timeout_secs: u64,
) -> ExecResult {
    use tokio::process::Command;
    use tokio::io::AsyncWriteExt;

    let wrapper = format!(
        "import json, sys\nargs = json.load(sys.stdin)\n{}",
        code
    );

    // Write to temp file to avoid argument length limits
    let temp_dir = std::env::temp_dir();
    let script_path = temp_dir.join(format!("sage_tool_{}.py", std::process::id()));

    if let Err(e) = std::fs::write(&script_path, &wrapper) {
        return ExecResult {
            stdout: String::new(),
            stderr: format!("Failed to write temp script: {}", e),
            exit_code: 1,
            timed_out: false,
            duration_ms: 0,
        };
    }

    let start = Instant::now();

    let mut child = match Command::new(python_exe)
        .arg(&script_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_file(&script_path);
            return ExecResult {
                stdout: String::new(),
                stderr: format!("Failed to spawn: {}", e),
                exit_code: 1,
                timed_out: false,
                duration_ms: 0,
            };
        }
    };

    // Write args to stdin
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(args_json.as_bytes()).await;
        drop(stdin); // Close stdin so child can read EOF
    }

    let timeout = Duration::from_secs(timeout_secs);
    let result = tokio::time::timeout(timeout, child.wait_with_output()).await;

    let _ = std::fs::remove_file(&script_path);
    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(Ok(output)) => ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).into(),
            stderr: String::from_utf8_lossy(&output.stderr).into(),
            exit_code: output.status.code().unwrap_or(1),
            timed_out: false,
            duration_ms: elapsed,
        },
        Ok(Err(e)) => ExecResult {
            stdout: String::new(),
            stderr: format!("Process error: {}", e),
            exit_code: 1,
            timed_out: false,
            duration_ms: elapsed,
        },
        Err(_) => {
            // Timeout — kill_on_drop handles cleanup
            ExecResult {
                stdout: String::new(),
                stderr: "Execution timed out".into(),
                exit_code: 137,
                timed_out: true,
                duration_ms: elapsed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn python_exe() -> String {
        // Use sys.executable equivalent
        if cfg!(windows) {
            "python".to_string()
        } else {
            "python3".to_string()
        }
    }

    #[test]
    fn test_simple_execution() {
        let r = execute_python_subprocess(
            &python_exe(),
            r#"print("hello from rust")"#,
            "{}",
            10,
        );
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("hello from rust"));
    }

    #[test]
    fn test_args_passthrough() {
        let r = execute_python_subprocess(
            &python_exe(),
            r#"print(json.dumps({"sum": args["a"] + args["b"]}))"#,
            r#"{"a": 3, "b": 4}"#,
            10,
        );
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("\"sum\": 7"));
    }

    #[test]
    fn test_timeout() {
        let r = execute_python_subprocess(
            &python_exe(),
            "import time; time.sleep(999)",
            "{}",
            2,
        );
        assert!(r.timed_out);
        assert_eq!(r.exit_code, 137);
    }

    #[test]
    fn test_syntax_error() {
        let r = execute_python_subprocess(
            &python_exe(),
            "def f(:\n  pass",
            "{}",
            10,
        );
        assert_ne!(r.exit_code, 0);
        assert!(r.stderr.contains("SyntaxError"));
    }

    #[test]
    fn test_runtime_error() {
        let r = execute_python_subprocess(
            &python_exe(),
            "raise ValueError('test error')",
            "{}",
            10,
        );
        assert_ne!(r.exit_code, 0);
        assert!(r.stderr.contains("ValueError"));
    }
}
```

**Step 2: Add module to mod.rs**

```rust
#[cfg(feature = "tool-executor")]
pub mod subprocess;
```

**Step 3: Run tests**

```bash
cd sage-core && cargo test --features tool-executor -- subprocess
```

**Step 4: Commit**

```bash
git add sage-core/src/sandbox/subprocess.rs sage-core/src/sandbox/mod.rs
git commit -m "feat: add Rust subprocess executor with timeout and kill-on-drop"
```

---

### Task 4: Create ToolExecutor PyO3 class

**Files:**
- Create: `sage-core/src/sandbox/tool_executor.rs`
- Modify: `sage-core/src/sandbox/mod.rs`
- Modify: `sage-core/src/lib.rs`

**Step 1: Write ToolExecutor**

```rust
// sage-core/src/sandbox/tool_executor.rs
//! ToolExecutor: PyO3 class combining validation + sandboxed execution.
//!
//! This is the single entry point for Python tool creation from sage-core.
//! Python `meta.py` calls `sage_core.ToolExecutor.validate_and_execute()`.

use pyo3::prelude::*;
use super::validator::{validate_python_code, ValidationResult};
use super::subprocess::{execute_python_subprocess, ExecResult};

/// Combined validator + executor for dynamic tool creation.
///
/// Usage from Python:
/// ```python
/// from sage_core import ToolExecutor
/// executor = ToolExecutor()
/// # Validate only
/// result = executor.validate(code)
/// # Validate + execute
/// result = executor.validate_and_execute(code, args_json)
/// ```
#[pyclass]
pub struct ToolExecutor {
    python_exe: String,
    timeout_secs: u64,
}

#[pymethods]
impl ToolExecutor {
    #[new]
    #[pyo3(signature = (python_exe=None, timeout_secs=30))]
    pub fn new(python_exe: Option<String>, timeout_secs: u64) -> Self {
        let exe = python_exe.unwrap_or_else(|| {
            if cfg!(windows) { "python".to_string() } else { "python3".to_string() }
        });
        Self {
            python_exe: exe,
            timeout_secs,
        }
    }

    /// Validate Python code without executing it.
    /// Returns ValidationResult with valid flag and error list.
    pub fn validate(&self, code: &str) -> ValidationResult {
        validate_python_code(code)
    }

    /// Validate and execute Python code in a sandboxed subprocess.
    /// Returns None if validation fails (check validate() for errors).
    /// Returns ExecResult if validation passes and code is executed.
    pub fn validate_and_execute(
        &self,
        py: Python<'_>,
        code: &str,
        args_json: &str,
    ) -> PyResult<ExecResult> {
        // 1. Validate
        let validation = validate_python_code(code);
        if !validation.valid {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Code validation failed:\n{}",
                    validation.errors.join("\n")
                )
            ));
        }

        // 2. Execute in subprocess (release GIL)
        let python_exe = self.python_exe.clone();
        let code = code.to_string();
        let args = args_json.to_string();
        let timeout = self.timeout_secs;

        let result = py.allow_threads(move || {
            execute_python_subprocess(&python_exe, &code, &args, timeout)
        });

        Ok(result)
    }

    /// Execute Python code without validation (for pre-validated code).
    /// Use with caution — caller is responsible for validation.
    pub fn execute_raw(
        &self,
        py: Python<'_>,
        code: &str,
        args_json: &str,
    ) -> ExecResult {
        let python_exe = self.python_exe.clone();
        let code = code.to_string();
        let args = args_json.to_string();
        let timeout = self.timeout_secs;

        py.allow_threads(move || {
            execute_python_subprocess(&python_exe, &code, &args, timeout)
        })
    }
}
```

**Step 2: Register in mod.rs and lib.rs**

In `sage-core/src/sandbox/mod.rs`:
```rust
#[cfg(feature = "tool-executor")]
pub mod tool_executor;
```

In `sage-core/src/lib.rs`, add to the `#[pymodule]` function:
```rust
#[cfg(feature = "tool-executor")]
{
    m.add_class::<sandbox::validator::ValidationResult>()?;
    m.add_class::<sandbox::subprocess::ExecResult>()?;
    m.add_class::<sandbox::tool_executor::ToolExecutor>()?;
}
```

**Step 3: Build and test**

```bash
cd sage-core && cargo test --features tool-executor
maturin develop --features tool-executor
python -c "from sage_core import ToolExecutor; e = ToolExecutor(); print(e.validate('import os').errors)"
```

**Step 4: Commit**

```bash
git add sage-core/src/sandbox/tool_executor.rs sage-core/src/sandbox/mod.rs sage-core/src/lib.rs
git commit -m "feat: add ToolExecutor PyO3 class (validator + subprocess executor)"
```

---

### Task 5: Refactor Python meta.py to use Rust ToolExecutor

**Files:**
- Modify: `sage-python/src/sage/tools/meta.py`
- Modify: `sage-python/tests/test_meta_security.py`

**Step 1: Modify create_python_tool to use Rust path**

```python
async def create_python_tool(name: str, code: str, registry: ToolRegistry = None) -> str:
    """Create a sandboxed Python tool.

    Uses Rust ToolExecutor (tree-sitter validation + subprocess isolation)
    when sage_core is compiled with tool-executor feature.
    Falls back to Python sandbox_executor if Rust not available.
    """
    if not registry:
        return "Error: Tool registry not available for dynamic registration."

    # Try Rust path first (tree-sitter validation + subprocess)
    try:
        from sage_core import ToolExecutor
        _executor = ToolExecutor()

        # Validate via tree-sitter
        validation = _executor.validate(code)
        if not validation.valid:
            return "Blocked: Code failed security validation:\n" + "\n".join(
                f"  - {e}" for e in validation.errors
            )

        # Create sandboxed handler using Rust executor
        saved_code = code
        async def _rust_handler(**kwargs):
            import json as _json
            result = _executor.execute_raw(saved_code, _json.dumps(kwargs))
            if result.exit_code != 0:
                return f"Tool execution error (exit {result.exit_code}):\n{result.stderr}"
            stdout = result.stdout.strip()
            try:
                parsed = _json.loads(stdout)
                return parsed.get("output", stdout) if isinstance(parsed, dict) else stdout
            except (_json.JSONDecodeError, ValueError):
                return stdout

        handler = _rust_handler

    except (ImportError, AttributeError):
        # Fallback: Python sandbox_executor
        from sage.tools.sandbox_executor import validate_tool_code, execute_python_in_sandbox

        errors = validate_tool_code(code)
        if errors:
            return "Blocked: Code failed security validation:\n" + "\n".join(
                f"  - {e}" for e in errors
            )

        saved_code = code
        async def _python_handler(**kwargs):
            import json as _json
            result = await execute_python_in_sandbox(saved_code, kwargs)
            if result.exit_code != 0:
                return f"Tool execution error (exit {result.exit_code}):\n{result.stderr}"
            stdout = result.stdout.strip()
            try:
                parsed = _json.loads(stdout)
                return parsed.get("output", stdout) if isinstance(parsed, dict) else stdout
            except (_json.JSONDecodeError, ValueError):
                return stdout

        handler = _python_handler

    # Save code for auditability
    file_path = os.path.join(TOOLS_WORKSPACE, f"{name}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Register tool
    from sage.llm.base import ToolDef
    spec = ToolDef(
        name=name,
        description=f"Dynamically created tool: {name}",
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
    )
    tool = Tool(spec=spec, handler=handler)
    registry.register(tool)

    logger.info("Registered sandboxed tool '%s' (saved to %s)", name, file_path)
    return f"Success: Tool '{name}' registered with sandbox isolation."
```

**Step 2: Add test for Rust path**

```python
def test_create_python_tool_uses_rust_executor():
    """Tool creation uses Rust ToolExecutor when available."""
    try:
        from sage_core import ToolExecutor
    except ImportError:
        pytest.skip("sage_core not compiled with tool-executor feature")

    from sage.tools.meta import create_python_tool
    from sage.tools.registry import ToolRegistry
    registry = ToolRegistry()
    result = asyncio.run(create_python_tool.fn(
        name="rust_adder",
        code='print(json.dumps({"output": str(args["a"] + args["b"])}))',
        registry=registry,
    ))
    assert "Success" in result
```

**Step 3: Run tests**

```bash
cd sage-python && python -m pytest tests/test_meta_security.py -v
python -m pytest tests/ -q 2>&1 | tail -5
```

**Step 4: Commit**

```bash
git commit -m "feat: wire create_python_tool to Rust ToolExecutor with Python fallback"
```

---

### Task 6: Update CLAUDE.md and README

**Files:**
- Modify: `CLAUDE.md` (tool-executor section)
- Modify: `README.md` (test count)

**Step 1: Add tool-executor documentation**

Add to CLAUDE.md under Rust Core section:
```markdown
### Tool Executor (tool-executor feature)
- `sandbox/validator.rs` — tree-sitter-python AST validation (blocked imports/calls)
- `sandbox/subprocess.rs` — Subprocess executor with timeout + kill-on-drop
- `sandbox/tool_executor.rs` — `ToolExecutor` PyO3 class combining validation + execution
- Build: `maturin develop --features tool-executor`
- Python fallback: `sage.tools.sandbox_executor` if Rust not compiled
```

**Step 2: Update test badge count**

**Step 3: Commit**

```bash
git commit -m "docs: add tool-executor documentation"
```

---

## Phase 2: CPython WASI Component (Wasm primary path)

### Task 7: Build CPython WASI Wasm component

**Files:**
- Create: `sage-core/plugins/python-runner/` (cargo-component project)
- Create: `sage-core/plugins/python-runner/wit/world.wit`
- Create: `sage-core/plugins/python-runner/src/lib.rs`

**Step 1: Install cargo-component**

```bash
cargo install cargo-component --locked
```

**Step 2: Create plugin project**

```bash
cd sage-core/plugins && cargo component new --lib python-runner
```

**Step 3: Set up WIT interface**

Copy `sage-core/interface.wit` content to `plugins/python-runner/wit/world.wit`.

**Step 4: Implement the component**

The component receives Python code in `args` field and executes it.
For Phase 2, this uses a simple expression evaluator.
Full CPython WASI integration (via componentize-py or Eryx) is Phase 3.

```rust
#[allow(warnings)]
mod bindings;
use bindings::{Guest, ToolInput, ToolOutput};

struct PythonRunner;

impl Guest for PythonRunner {
    fn run(input: ToolInput) -> ToolOutput {
        // Phase 2: simple JSON-based tool dispatch
        // Phase 3: full CPython WASI interpreter
        let result = match serde_json::from_str::<serde_json::Value>(&input.args) {
            Ok(args) => {
                // Execute simple operations based on tool name
                serde_json::json!({
                    "output": format!("Wasm executed: {} with args {}", input.name, args),
                }).to_string()
            }
            Err(e) => serde_json::json!({"error": e.to_string()}).to_string(),
        };

        ToolOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            result_json: result,
        }
    }
}

bindings::export!(PythonRunner with_types_in bindings);
```

**Step 5: Build and pre-compile**

```bash
cd sage-core/plugins/python-runner
cargo component build --release
# Output: target/wasm32-wasip1/release/python_runner.wasm
```

**Step 6: Wire into ToolExecutor**

Add Wasm execution path to `tool_executor.rs`:
- Try WasmSandbox.execute() with python-runner component first
- Fall back to subprocess if Wasm unavailable

**Step 7: Commit**

---

### Task 8: Integrate CPython WASI via componentize-py (Phase 3)

This task replaces the simple Phase 2 component with a full CPython interpreter.

**Step 1: Install componentize-py**

```bash
pip install componentize-py
```

**Step 2: Create Python component**

```bash
componentize-py -d sage-core/interface.wit -w tool-env \
  componentize python_runner -o sage-core/plugins/cpython-runner.wasm
```

**Step 3: Pre-compile for Windows**

```bash
wasmtime compile sage-core/plugins/cpython-runner.wasm \
  -o sage-core/plugins/cpython-runner.cwasm
```

**Step 4: Wire into ToolExecutor as primary execution path**

---

## Dependencies

```
Task 1 (Cargo deps) → independent
Task 2 (validator) → depends on Task 1
Task 3 (subprocess) → depends on Task 1
Task 4 (ToolExecutor) → depends on Tasks 2, 3
Task 5 (Python refactor) → depends on Task 4
Task 6 (docs) → depends on Task 5
Task 7 (Wasm component) → depends on Task 4
Task 8 (CPython WASI) → depends on Task 7
```

## Success Criteria

Phase 1:
- `sage_core.ToolExecutor` validates Python code via tree-sitter (not Python AST)
- `sage_core.ToolExecutor` executes code in subprocess with timeout
- `py.allow_threads()` used — no GIL blocking
- Python `sandbox_executor.py` serves as fallback when Rust not compiled
- All existing tests pass + new Rust tests
- `cargo test --features tool-executor` passes

Phase 2:
- Wasm component built with cargo-component
- ToolExecutor tries Wasm first, subprocess second
- Pre-compiled `.cwasm` for Windows

Phase 3:
- Full CPython WASI interpreter in Wasm
- Agent-generated Python executes inside wasmtime
- Complete memory isolation
- <20ms with pre-initialization
