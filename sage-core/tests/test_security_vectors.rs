//! End-to-end security vector tests for the tool-executor.
//!
//! Tests all 6 Phoenix security vectors against the tree-sitter validator.
//! Requires: `cargo test --features tool-executor`

#[cfg(feature = "tool-executor")]
mod security {
    use sage_core::sandbox::validator::validate_python_code;

    // ────────────────────────────────────────────
    // Vector 1: Filesystem Read
    // ────────────────────────────────────────────

    #[test]
    fn v1_blocks_open_read() {
        let r = validate_python_code(r#"f = open("/etc/passwd", "r")"#);
        assert!(!r.valid, "Should block open(): {:?}", r.errors);
    }

    #[test]
    fn v1_blocks_pathlib_import() {
        let r = validate_python_code("import pathlib\np = pathlib.Path('/')");
        assert!(!r.valid, "Should block pathlib: {:?}", r.errors);
    }

    #[test]
    fn v1_blocks_from_pathlib() {
        let r = validate_python_code("from pathlib import Path\ndata = Path('/etc/hosts').read_bytes()");
        assert!(!r.valid, "Should block from pathlib: {:?}", r.errors);
    }

    #[test]
    fn v1_blocks_glob() {
        let r = validate_python_code(r#"import glob\nfiles = glob.glob("/etc/*")"#);
        assert!(!r.valid, "Should block glob: {:?}", r.errors);
    }

    // ────────────────────────────────────────────
    // Vector 2: Filesystem Write
    // ────────────────────────────────────────────

    #[test]
    fn v2_blocks_open_write() {
        let r = validate_python_code(r#"f = open("/tmp/evil", "w")"#);
        assert!(!r.valid, "Should block open(w): {:?}", r.errors);
    }

    #[test]
    fn v2_blocks_shutil() {
        let r = validate_python_code("import shutil\nshutil.rmtree('/tmp')");
        assert!(!r.valid, "Should block shutil: {:?}", r.errors);
    }

    #[test]
    fn v2_blocks_tempfile() {
        let r = validate_python_code("import tempfile\nf = tempfile.NamedTemporaryFile()");
        assert!(!r.valid, "Should block tempfile: {:?}", r.errors);
    }

    // ────────────────────────────────────────────
    // Vector 3: Env Var Access
    // ────────────────────────────────────────────

    #[test]
    fn v3_blocks_os_environ() {
        let r = validate_python_code("import os\nos.environ['KEY']");
        assert!(!r.valid, "Should block os: {:?}", r.errors);
    }

    #[test]
    fn v3_blocks_os_getenv() {
        let r = validate_python_code("import os\nos.getenv('SECRET')");
        assert!(!r.valid, "Should block os: {:?}", r.errors);
    }

    #[test]
    fn v3_blocks_from_os_environ() {
        let r = validate_python_code("from os import environ\nkeys = dict(environ)");
        assert!(!r.valid, "Should block from os: {:?}", r.errors);
    }

    // ────────────────────────────────────────────
    // Vector 4: Network Access
    // ────────────────────────────────────────────

    #[test]
    fn v4_blocks_socket() {
        let r = validate_python_code("import socket\ns = socket.socket()");
        assert!(!r.valid, "Should block socket: {:?}", r.errors);
    }

    #[test]
    fn v4_blocks_http_client() {
        let r = validate_python_code("import http.client\nc = http.client.HTTPConnection('evil.com')");
        assert!(!r.valid, "Should block http: {:?}", r.errors);
    }

    #[test]
    fn v4_blocks_from_http() {
        let r = validate_python_code("from http import server");
        assert!(!r.valid, "Should block from http: {:?}", r.errors);
    }

    #[test]
    fn v4_blocks_ftplib() {
        let r = validate_python_code("import ftplib\nftp = ftplib.FTP('evil.com')");
        assert!(!r.valid, "Should block ftplib: {:?}", r.errors);
    }

    #[test]
    fn v4_blocks_smtplib() {
        let r = validate_python_code("import smtplib\nsmtp = smtplib.SMTP('evil.com')");
        assert!(!r.valid, "Should block smtplib: {:?}", r.errors);
    }

    #[test]
    fn v4_blocks_xmlrpc() {
        let r = validate_python_code("import xmlrpc.client\nc = xmlrpc.client.ServerProxy('http://evil.com')");
        assert!(!r.valid, "Should block xmlrpc: {:?}", r.errors);
    }

    // ────────────────────────────────────────────
    // Vector 5: Subprocess Spawn
    // ────────────────────────────────────────────

    #[test]
    fn v5_blocks_subprocess_run() {
        let r = validate_python_code("import subprocess\nsubprocess.run(['whoami'])");
        assert!(!r.valid, "Should block subprocess: {:?}", r.errors);
    }

    #[test]
    fn v5_blocks_from_subprocess() {
        let r = validate_python_code("from subprocess import Popen\np = Popen(['cat', '/etc/passwd'])");
        assert!(!r.valid, "Should block from subprocess: {:?}", r.errors);
    }

    #[test]
    fn v5_blocks_os_system() {
        let r = validate_python_code("import os\nos.system('rm -rf /')");
        assert!(!r.valid, "Should block os.system: {:?}", r.errors);
    }

    #[test]
    fn v5_blocks_os_popen() {
        let r = validate_python_code("import os\nos.popen('id')");
        assert!(!r.valid, "Should block os.popen: {:?}", r.errors);
    }

    #[test]
    fn v5_blocks_multiprocessing() {
        let r = validate_python_code("import multiprocessing\np = multiprocessing.Process(target=print)");
        assert!(!r.valid, "Should block multiprocessing: {:?}", r.errors);
    }

    // ────────────────────────────────────────────
    // Vector 6: Dangerous Imports (all 23 modules)
    // ────────────────────────────────────────────

    #[test]
    fn v6_blocks_all_23_modules() {
        let blocked = [
            "os", "sys", "subprocess", "shutil", "ctypes", "importlib",
            "socket", "http", "ftplib", "smtplib", "xmlrpc",
            "multiprocessing", "threading", "signal", "resource",
            "code", "codeop", "pathlib", "glob", "tempfile",
            "pickle", "shelve", "builtins",
        ];
        for module in blocked {
            let r = validate_python_code(&format!("import {}", module));
            assert!(!r.valid, "Should block import {}: {:?}", module, r.errors);
            assert!(
                r.errors.iter().any(|e| e.contains(module)),
                "Error should mention '{}', got: {:?}", module, r.errors
            );
        }
    }

    #[test]
    fn v6_blocks_all_23_from_imports() {
        let blocked = [
            "os", "sys", "subprocess", "shutil", "ctypes", "importlib",
            "socket", "http", "ftplib", "smtplib", "xmlrpc",
            "multiprocessing", "threading", "signal", "resource",
            "code", "codeop", "pathlib", "glob", "tempfile",
            "pickle", "shelve", "builtins",
        ];
        for module in blocked {
            let r = validate_python_code(&format!("from {} import *", module));
            assert!(!r.valid, "Should block from {} import *: {:?}", module, r.errors);
        }
    }

    #[test]
    fn v6_blocks_all_11_calls() {
        let blocked_calls = [
            "exec('x')",
            "eval('x')",
            "compile('x','','exec')",
            "__import__('os')",
            "breakpoint()",
            "open('/etc/passwd')",
            "getattr(x, 'y')",
            "setattr(x, 'y', 1)",
            "delattr(x, 'y')",
            "globals()",
            "locals()",
        ];
        for code in blocked_calls {
            let r = validate_python_code(code);
            assert!(!r.valid, "Should block {}: {:?}", code, r.errors);
        }
    }

    // ────────────────────────────────────────────
    // Safe code must pass
    // ────────────────────────────────────────────

    #[test]
    fn allows_json() {
        let r = validate_python_code("import json\nresult = json.dumps({'a': 1})");
        assert!(r.valid, "Should allow json: {:?}", r.errors);
    }

    #[test]
    fn allows_math() {
        let r = validate_python_code("import math\nprint(math.sqrt(144))");
        assert!(r.valid, "Should allow math: {:?}", r.errors);
    }

    #[test]
    fn allows_re() {
        let r = validate_python_code("import re\nm = re.match(r'\\d+', '123')");
        assert!(r.valid, "Should allow re: {:?}", r.errors);
    }

    #[test]
    fn allows_collections() {
        let r = validate_python_code("import collections\nc = collections.Counter([1,1,2])");
        assert!(r.valid, "Should allow collections: {:?}", r.errors);
    }

    #[test]
    fn allows_itertools() {
        let r = validate_python_code("import itertools\nlist(itertools.chain([1], [2]))");
        assert!(r.valid, "Should allow itertools: {:?}", r.errors);
    }

    #[test]
    fn allows_pure_computation() {
        let r = validate_python_code("x = sum(range(100))\nprint(x)");
        assert!(r.valid, "Should allow pure computation: {:?}", r.errors);
    }

    #[test]
    fn allows_hashlib() {
        let r = validate_python_code("import hashlib\nprint(hashlib.sha256(b'hello').hexdigest())");
        assert!(r.valid, "Should allow hashlib: {:?}", r.errors);
    }

    #[test]
    fn allows_datetime() {
        let r = validate_python_code("import datetime\nprint(datetime.datetime.now().isoformat())");
        assert!(r.valid, "Should allow datetime: {:?}", r.errors);
    }

    #[test]
    fn allows_string() {
        let r = validate_python_code("import string\nprint(string.ascii_uppercase)");
        assert!(r.valid, "Should allow string: {:?}", r.errors);
    }

    #[test]
    fn allows_functools() {
        let r = validate_python_code("import functools\nfunctools.reduce(lambda a,b: a+b, [1,2,3])");
        assert!(r.valid, "Should allow functools: {:?}", r.errors);
    }
}
