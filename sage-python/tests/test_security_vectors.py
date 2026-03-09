"""End-to-end security validation: 6/6 Phoenix security vectors.

Tests verify that the YGN-SAGE tool sandbox blocks all 6 attack vectors
identified by the Arize Phoenix sandbox benchmarks:
1. Filesystem read
2. Filesystem write
3. Env var access
4. Network access
5. Subprocess spawn
6. Import dangerous module

Each vector is tested against:
- tree-sitter validator (Rust, via sage_core.ToolExecutor if available)
- Python AST validator (fallback, via sandbox_executor)
- Full pipeline (create_python_tool rejection)
"""
import pytest
import asyncio

from sage.tools.sandbox_executor import validate_tool_code


def _has_sage_core():
    """Check if sage_core with tool-executor feature is available."""
    try:
        from sage_core import ToolExecutor
        ToolExecutor()
        return True
    except (ImportError, AttributeError):
        return False


def _rust_validate(code: str):
    """Validate code with the Rust tree-sitter validator."""
    from sage_core import ToolExecutor
    return ToolExecutor().validate(code)


# ──────────────────────────────────────────────
# Vector 1: Filesystem Read
# ──────────────────────────────────────────────
class TestFilesystemRead:
    """Vector 1: Agent code must NOT be able to read host filesystem."""

    ATTACK_CODES = [
        'f = open("/etc/passwd", "r")\ndata = f.read()',
        'import pathlib\np = pathlib.Path("/etc/passwd")\ndata = p.read_text()',
        'from pathlib import Path\ndata = Path("/etc/hosts").read_bytes()',
        'import glob\nfiles = glob.glob("/etc/*")',
    ]

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_tree_sitter_blocks(self, code):
        """tree-sitter validator blocks filesystem read attempts."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(code)
        assert not result.valid, f"Should block: {code}"

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_python_validator_blocks(self, code):
        """Python AST validator blocks filesystem read attempts."""
        errors = validate_tool_code(code)
        assert len(errors) > 0, f"Should block: {code}"

    def test_full_pipeline_rejects(self):
        """create_python_tool rejects filesystem read code."""
        from sage.tools.meta import create_python_tool
        from sage.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = asyncio.run(create_python_tool._handler(
            name="fs_reader",
            code='f = open("/etc/passwd")\nprint(f.read())',
            registry=registry,
        ))
        assert "Blocked" in result
        assert registry.get("fs_reader") is None


# ──────────────────────────────────────────────
# Vector 2: Filesystem Write
# ──────────────────────────────────────────────
class TestFilesystemWrite:
    """Vector 2: Agent code must NOT be able to write to host filesystem."""

    ATTACK_CODES = [
        'f = open("/tmp/evil.txt", "w")\nf.write("pwned")',
        'import shutil\nshutil.copy("/etc/passwd", "/tmp/stolen")',
        'import tempfile\nf = tempfile.NamedTemporaryFile()',
    ]

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_tree_sitter_blocks(self, code):
        """tree-sitter validator blocks filesystem write attempts."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(code)
        assert not result.valid, f"Should block: {code}"

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_python_validator_blocks(self, code):
        """Python AST validator blocks filesystem write attempts."""
        errors = validate_tool_code(code)
        assert len(errors) > 0, f"Should block: {code}"


# ──────────────────────────────────────────────
# Vector 3: Env Var Access
# ──────────────────────────────────────────────
class TestEnvVarAccess:
    """Vector 3: Agent code must NOT access host environment variables."""

    ATTACK_CODES = [
        'import os\nsecret = os.environ["API_KEY"]',
        'import os\nsecret = os.getenv("SECRET")',
        'from os import environ\nkeys = dict(environ)',
    ]

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_tree_sitter_blocks(self, code):
        """tree-sitter validator blocks env var access attempts."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(code)
        assert not result.valid, f"Should block: {code}"

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_python_validator_blocks(self, code):
        """Python AST validator blocks env var access attempts."""
        errors = validate_tool_code(code)
        assert len(errors) > 0, f"Should block: {code}"


# ──────────────────────────────────────────────
# Vector 4: Network Access
# ──────────────────────────────────────────────
class TestNetworkAccess:
    """Vector 4: Agent code must NOT make network requests."""

    ATTACK_CODES = [
        'import socket\ns = socket.socket()',
        'import http.client\nc = http.client.HTTPConnection("evil.com")',
        'from http import server\nhttpd = server.HTTPServer(("", 8080), None)',
        'import ftplib\nftp = ftplib.FTP("evil.com")',
        'import smtplib\nsmtp = smtplib.SMTP("evil.com")',
        'import xmlrpc.client\nc = xmlrpc.client.ServerProxy("http://evil.com")',
    ]

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_tree_sitter_blocks(self, code):
        """tree-sitter validator blocks network access attempts."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(code)
        assert not result.valid, f"Should block: {code}"

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_python_validator_blocks(self, code):
        """Python AST validator blocks network access attempts."""
        errors = validate_tool_code(code)
        assert len(errors) > 0, f"Should block: {code}"


# ──────────────────────────────────────────────
# Vector 5: Subprocess Spawn
# ──────────────────────────────────────────────
class TestSubprocessSpawn:
    """Vector 5: Agent code must NOT spawn host subprocesses."""

    ATTACK_CODES = [
        'import subprocess\nsubprocess.run(["whoami"])',
        'from subprocess import Popen\np = Popen(["cat", "/etc/passwd"])',
        'import os\nos.system("rm -rf /")',
        'import os\nos.popen("id")',
        'import multiprocessing\np = multiprocessing.Process(target=print)',
    ]

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_tree_sitter_blocks(self, code):
        """tree-sitter validator blocks subprocess spawn attempts."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(code)
        assert not result.valid, f"Should block: {code}"

    @pytest.mark.parametrize("code", ATTACK_CODES)
    def test_python_validator_blocks(self, code):
        """Python AST validator blocks subprocess spawn attempts."""
        errors = validate_tool_code(code)
        assert len(errors) > 0, f"Should block: {code}"


# ──────────────────────────────────────────────
# Vector 6: Import Dangerous Module
# ──────────────────────────────────────────────
class TestDangerousImport:
    """Vector 6: Agent code must NOT import security-critical modules."""

    # All 23 blocked modules from the tree-sitter validator
    BLOCKED_MODULES = [
        "os", "sys", "subprocess", "shutil", "ctypes", "importlib",
        "socket", "http", "ftplib", "smtplib", "xmlrpc",
        "multiprocessing", "threading", "signal", "resource",
        "code", "codeop", "pathlib", "glob", "tempfile",
        "pickle", "shelve", "builtins",
    ]

    @pytest.mark.parametrize("module", BLOCKED_MODULES)
    def test_tree_sitter_blocks_import(self, module):
        """tree-sitter blocks `import <module>` for all 23 blocked modules."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(f"import {module}")
        assert not result.valid, f"Should block import {module}"
        assert any(module in e for e in result.errors)

    @pytest.mark.parametrize("module", BLOCKED_MODULES)
    def test_tree_sitter_blocks_from_import(self, module):
        """tree-sitter blocks `from <module> import *` for all 23 blocked modules."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(f"from {module} import *")
        assert not result.valid, f"Should block from {module} import"

    @pytest.mark.parametrize("module", BLOCKED_MODULES)
    def test_python_validator_blocks(self, module):
        """Python AST validator blocks all 23 blocked modules."""
        errors = validate_tool_code(f"import {module}")
        assert len(errors) > 0, f"Should block import {module}"

    # Blocked function calls — Rust tree-sitter has 11 blocked calls
    RUST_BLOCKED_CALLS = [
        "exec('x = 1')",
        "eval('1+1')",
        "compile('x=1', '<string>', 'exec')",
        "__import__('os')",
        "breakpoint()",
        "open('/etc/passwd')",
        "globals()",
        "locals()",
        "getattr(object, 'name')",
        "setattr(object, 'name', 'value')",
        "delattr(object, 'name')",
    ]

    @pytest.mark.parametrize("code", RUST_BLOCKED_CALLS)
    def test_tree_sitter_blocks_calls(self, code):
        """tree-sitter blocks all 11 dangerous function calls."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(code)
        assert not result.valid, f"Should block: {code}"

    # Python validator blocks a subset (6 of 11)
    PYTHON_BLOCKED_CALLS = [
        "exec('x = 1')",
        "eval('1+1')",
        "compile('x=1', '<string>', 'exec')",
        "__import__('os')",
        "breakpoint()",
        "open('/etc/passwd')",
    ]

    @pytest.mark.parametrize("code", PYTHON_BLOCKED_CALLS)
    def test_python_validator_blocks_calls(self, code):
        """Python AST validator blocks core dangerous calls."""
        errors = validate_tool_code(code)
        assert len(errors) > 0, f"Should block: {code}"


# ──────────────────────────────────────────────
# Combined: Safe Code Passes
# ──────────────────────────────────────────────
class TestSafeCodeAllowed:
    """Verify that legitimate safe code is NOT blocked."""

    SAFE_CODES = [
        'import json\nresult = json.dumps({"a": 1})\nprint(result)',
        'import math\nprint(math.sqrt(144))',
        'import re\nm = re.match(r"\\d+", "123")\nprint(m.group())',
        'import collections\nc = collections.Counter([1,1,2,3])\nprint(dict(c))',
        'import itertools\nprint(list(itertools.chain([1,2], [3,4])))',
        'import functools\nprint(functools.reduce(lambda a,b: a+b, [1,2,3]))',
        'import hashlib\nprint(hashlib.sha256(b"hello").hexdigest())',
        'import datetime\nprint(datetime.datetime.now().isoformat())',
        'import string\nprint(string.ascii_uppercase)',
        'x = sum(range(100))\nprint(x)',
    ]

    @pytest.mark.parametrize("code", SAFE_CODES)
    def test_tree_sitter_allows(self, code):
        """tree-sitter validator permits safe stdlib usage."""
        if not _has_sage_core():
            pytest.skip("sage_core not compiled with tool-executor")
        result = _rust_validate(code)
        assert result.valid, f"Should allow: {code}, errors: {result.errors}"

    @pytest.mark.parametrize("code", SAFE_CODES)
    def test_python_validator_allows(self, code):
        """Python AST validator permits safe stdlib usage."""
        errors = validate_tool_code(code)
        assert len(errors) == 0, f"Should allow: {code}, errors: {errors}"

    def test_safe_tool_creation_works(self):
        """create_python_tool accepts and executes safe code."""
        from sage.tools.meta import create_python_tool
        from sage.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = asyncio.run(create_python_tool._handler(
            name="safe_calculator",
            code='import json\nresult = args.get("a", 0) + args.get("b", 0)\nprint(json.dumps({"output": str(result)}))',
            registry=registry,
        ))
        assert "Success" in result
        assert registry.get("safe_calculator") is not None


# ──────────────────────────────────────────────
# Summary: count test coverage per vector
# ──────────────────────────────────────────────
# Vector 1 (FS Read):   4 Rust + 4 Python + 1 pipeline = 9
# Vector 2 (FS Write):  3 Rust + 3 Python              = 6
# Vector 3 (Env Var):   3 Rust + 3 Python              = 6
# Vector 4 (Network):   6 Rust + 6 Python              = 12
# Vector 5 (Subproc):   5 Rust + 5 Python              = 10
# Vector 6 (Imports):   23+23 Rust + 23 Python + 11+6  = 86
# Safe code:            10 Rust + 10 Python + 1 pipe   = 21
# ─────────────────────────────────────────────────────────
# Total:                                                = 150
