// Wasm component implementing the tool-env world (Phase 2: simple expression evaluator).
// Produces a Component Model .wasm that can be loaded by wasmtime v36 in sage-core.
// Phase 3 will replace this with CPython WASI via componentize-py.

// Generate bindings from WIT
wit_bindgen::generate!({
    world: "tool-env",
    path: "wit/world.wit",
});

struct PythonRunner;

impl Guest for PythonRunner {
    fn run(input: ToolInput) -> ToolOutput {
        // Parse args as JSON
        let args: serde_json::Value = match serde_json::from_str(&input.args) {
            Ok(v) => v,
            Err(e) => {
                return ToolOutput {
                    stdout: String::new(),
                    stderr: format!("JSON parse error: {}", e),
                    exit_code: 1,
                    result_json: String::new(),
                };
            }
        };

        // Simple expression evaluator for Phase 2
        // Supports: {"eval": "a + b", "a": 3, "b": 4}
        let result = if let Some(expr) = args.get("eval").and_then(|v| v.as_str()) {
            evaluate_simple_expr(expr, &args)
        } else {
            // Echo mode: just return the args as result
            Ok(serde_json::json!({
                "output": format!("Wasm executed tool '{}' with args: {}", input.name, args),
            }))
        };

        match result {
            Ok(value) => ToolOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
                result_json: value.to_string(),
            },
            Err(err) => ToolOutput {
                stdout: String::new(),
                stderr: err,
                exit_code: 1,
                result_json: String::new(),
            },
        }
    }
}

/// Evaluate simple arithmetic expressions from JSON args.
/// Supports: "a + b", "a - b", "a * b", "a / b"
fn evaluate_simple_expr(expr: &str, args: &serde_json::Value) -> Result<serde_json::Value, String> {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(format!("Expected 'var op var', got '{}'", expr));
    }

    let left = resolve_value(parts[0], args)?;
    let right = resolve_value(parts[2], args)?;

    let result = match parts[1] {
        "+" => left + right,
        "-" => left - right,
        "*" => left * right,
        "/" => {
            if right == 0.0 {
                return Err("Division by zero".to_string());
            }
            left / right
        }
        op => return Err(format!("Unknown operator: {}", op)),
    };

    Ok(serde_json::json!({"output": result}))
}

fn resolve_value(token: &str, args: &serde_json::Value) -> Result<f64, String> {
    // Try as number literal first
    if let Ok(n) = token.parse::<f64>() {
        return Ok(n);
    }
    // Try as variable name from args
    args.get(token)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("Variable '{}' not found in args", token))
}

export!(PythonRunner);
