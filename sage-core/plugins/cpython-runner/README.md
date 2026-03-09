# CPython WASI Tool Runner

CPython-based tool runner that executes Python code inside a Wasm sandbox
using WASI capabilities (deny-by-default).

## Architecture

- Built with [componentize-py](https://github.com/bytecodealliance/componentize-py)
- Implements the `tool-env` WIT world (`sage:plugin/tool-env`)
- Runs on wasmtime 36.0 LTS with WASI preview2
- Host (Rust) controls all WASI capabilities — guest has NO filesystem,
  network, env var, or subprocess access unless explicitly granted

## Build

Requires `componentize-py` (pip install componentize-py):

```bash
cd sage-core/plugins/cpython-runner

# Build the component (creates cpython-runner.wasm in plugins/)
componentize-py -d ../../interface.wit -w tool-env componentize app \
    -o ../cpython-runner.wasm
```

## Security Properties

The WASI host context (`WasiState::new_restrictive()`) blocks:
1. **Filesystem read** — no `preopened_dir()`
2. **Filesystem write** — no `preopened_dir()`
3. **Environment variables** — no `inherit_env()`
4. **Subprocess spawning** — not available in WASI preview2
5. **Network access** — not available in WASI preview2
6. **Dangerous imports** — tree-sitter validator blocks os/subprocess/etc.

Only stdout and stderr are inherited for output capture.

## Usage from Python

```python
from sage_core import ToolExecutor

executor = ToolExecutor()

# Load the pre-compiled component with WASI flag
with open("plugins/cpython-runner.wasm", "rb") as f:
    executor.load_precompiled_component(f.read(), wasi=True)

# Execute Python code in the sandbox
result = executor.validate_and_execute(
    'result = sum(args["numbers"])',
    '{"__code__": "result = sum(args[\\"numbers\\"])", "numbers": [1, 2, 3]}'
)
```
