//! Sandboxed subprocess executor for Python code.
//!
//! Executes Python code in an isolated subprocess with:
//! - Timeout enforcement via tokio
//! - kill_on_drop for clean process cleanup
//! - No shell=True (uses Command directly)
//! - GIL-safe (designed for use via py.allow_threads())
//!
//! # Security Limitations (Audit3 F-02)
//!
//! This executor provides **timeout isolation only**.
//! There is NO OS-level sandboxing:
//! - No seccomp filters (Linux)
//! - No namespace isolation
//! - No cgroup resource limits
//! - No filesystem/network deny-by-default
//!
//! **Defense-in-depth**: AST validation (tree-sitter) runs BEFORE this executor
//! via `ToolExecutor::validate_and_execute()`. The subprocess is a fallback for
//! when Wasm WASI sandbox is unavailable.
//!
//! **Production recommendation**: Compile with `sandbox` feature for Wasm WASI
//! deny-by-default isolation. For Linux deployments, consider nsjail wrapper.

use pyo3::prelude::*;
use std::time::{Duration, Instant};
use tokio::runtime::Builder;

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
/// Writes code to a temp file, executes via `python <file>`,
/// feeds args as JSON via stdin.
pub fn execute_python_subprocess(
    python_exe: &str,
    code: &str,
    args_json: &str,
    timeout_secs: u64,
) -> ExecResult {
    let rt = Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async { execute_async(python_exe, code, args_json, timeout_secs).await })
}

async fn execute_async(
    python_exe: &str,
    code: &str,
    args_json: &str,
    timeout_secs: u64,
) -> ExecResult {
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let wrapper = format!("import json, sys\nargs = json.load(sys.stdin)\n{}", code);

    // Write to temp file to avoid argument length limits
    let temp_dir = std::env::temp_dir();
    let unique_id = format!(
        "{}_{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let script_path = temp_dir.join(format!("sage_tool_{}.py", unique_id));

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
        if cfg!(windows) {
            "python".to_string()
        } else {
            "python3".to_string()
        }
    }

    #[test]
    fn test_simple_execution() {
        let r = execute_python_subprocess(&python_exe(), r#"print("hello from rust")"#, "{}", 10);
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
        assert!(r.stdout.contains("7"));
    }

    #[test]
    fn test_timeout() {
        let r = execute_python_subprocess(&python_exe(), "import time; time.sleep(999)", "{}", 2);
        assert!(r.timed_out);
        assert_eq!(r.exit_code, 137);
    }

    #[test]
    fn test_syntax_error() {
        let r = execute_python_subprocess(&python_exe(), "def f(:\n  pass", "{}", 10);
        assert_ne!(r.exit_code, 0);
        assert!(r.stderr.contains("SyntaxError"));
    }

    #[test]
    fn test_runtime_error() {
        let r =
            execute_python_subprocess(&python_exe(), "raise ValueError('test error')", "{}", 10);
        assert_ne!(r.exit_code, 0);
        assert!(r.stderr.contains("ValueError"));
    }
}
