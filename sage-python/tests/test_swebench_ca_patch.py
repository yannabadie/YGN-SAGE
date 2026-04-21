"""Tests for sage.bench.swebench_ca_patch.

These tests verify the monkey-patch behavior without requiring a real
Docker daemon or a real swebench base-image build. We check:
  - Idempotence
  - Disable flag
  - Missing-bundle graceful behavior
  - Dockerfile template mutation
  - build_image wrapper copies the bundle into build_dir
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

import pytest


# -- sage_core / swebench import guard --------------------------------------
# Stub sage_core only if absent (matches project convention).
if "sage_core" not in sys.modules:
    try:
        import sage_core  # noqa: F401
    except ImportError:
        import types as _types
        sys.modules["sage_core"] = _types.ModuleType("sage_core")

# Stub the platform.resource module swebench demands on Windows.
if sys.platform != "linux" and "resource" not in sys.modules:
    import types as _types
    _stub = _types.ModuleType("resource")
    _stub.RLIMIT_NOFILE = 7  # type: ignore[attr-defined]
    _stub.getrlimit = lambda _x: (1024, 1048576)  # type: ignore[attr-defined]
    _stub.setrlimit = lambda _x, _y: None  # type: ignore[attr-defined]
    sys.modules["resource"] = _stub


def _fresh_patch_module():
    """Import a fresh copy of the patch module, resetting the APPLIED guard.

    Each test gets isolated state so idempotence checks are meaningful.
    """
    if "sage.bench.swebench_ca_patch" in sys.modules:
        del sys.modules["sage.bench.swebench_ca_patch"]
    # Also reset the swebench modules we'll patch so the template resets.
    # Must include the dockerfiles package (__init__) because it holds the
    # aggregator dict _DOCKERFILE_BASE, which is rebuilt only on fresh import.
    for mod in (
        "swebench.harness.dockerfiles.python",
        "swebench.harness.dockerfiles",
        "swebench.harness.docker_build",
        "swebench.harness.run_evaluation",
    ):
        sys.modules.pop(mod, None)
    return importlib.import_module("sage.bench.swebench_ca_patch")


def _make_tmp_bundle(tmp_path: Path) -> Path:
    """Create a fake PEM bundle for tests."""
    bundle = tmp_path / "ca-bundle.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----\n")
    return bundle


def test_disable_flag_returns_false(monkeypatch, tmp_path):
    """SAGE_SWEBENCH_DISABLE_CA_PATCH=1 → patch is a no-op, returns False."""
    monkeypatch.setenv("SAGE_SWEBENCH_DISABLE_CA_PATCH", "1")
    monkeypatch.setenv("SAGE_CORPORATE_CA_BUNDLE", str(_make_tmp_bundle(tmp_path)))
    mod = _fresh_patch_module()
    assert mod.apply_corporate_ca_patch() is False


def test_missing_bundle_returns_false(monkeypatch, tmp_path, caplog):
    """No bundle found → returns False with a warning (no Dockerfile change)."""
    monkeypatch.delenv("SAGE_SWEBENCH_DISABLE_CA_PATCH", raising=False)
    monkeypatch.setenv("SAGE_CORPORATE_CA_BUNDLE", str(tmp_path / "does-not-exist.pem"))
    monkeypatch.setattr(
        "sage.bench.swebench_ca_patch._DEFAULT_CA_BUNDLE",
        tmp_path / "also-missing.pem",
        raising=False,
    )
    mod = _fresh_patch_module()
    # Re-patch after reimport
    monkeypatch.setattr(mod, "_DEFAULT_CA_BUNDLE", tmp_path / "also-missing.pem")
    with caplog.at_level(logging.WARNING):
        assert mod.apply_corporate_ca_patch() is False
    assert any("no bundle found" in rec.message.lower() for rec in caplog.records)


def test_applies_dockerfile_and_wrapper_secure_default(monkeypatch, tmp_path):
    """Default secure path: template gains COPY + append-to-ca-certificates
    and the build_image wrapper is installed, but SSL verification is NOT
    disabled (Directive #3 compliance)."""
    bundle = _make_tmp_bundle(tmp_path)
    monkeypatch.delenv("SAGE_SWEBENCH_DISABLE_CA_PATCH", raising=False)
    monkeypatch.delenv("SAGE_SWEBENCH_ALLOW_INSECURE", raising=False)
    monkeypatch.setenv("SAGE_CORPORATE_CA_BUNDLE", str(bundle))

    mod = _fresh_patch_module()
    assert mod.apply_corporate_ca_patch() is True

    from swebench.harness.dockerfiles import python as dockerfile_python
    from swebench.harness import docker_build

    tpl = dockerfile_python._DOCKERFILE_BASE_PY
    marker = "corporate-ca-bundle.crt"
    assert marker in tpl, "COPY line not injected"
    # Correctness of ordering: the COPY must come BEFORE the wget miniconda.
    assert tpl.index(marker) < tpl.index("miniconda.sh")
    # Append to system bundle is the secure path — always present.
    assert "/etc/ssl/certs/ca-certificates.crt" in tpl

    # Directive #3: SSL verification must NOT be disabled by default.
    assert "--no-check-certificate" not in tpl, (
        "wget --no-check-certificate injected without "
        "SAGE_SWEBENCH_ALLOW_INSECURE=1 — Directive #3 violation"
    )
    assert "ssl_verify false" not in tpl, (
        "conda ssl_verify=false injected without opt-in env — "
        "Directive #3 violation"
    )
    assert "trusted-host" not in tpl, (
        "pip trusted-host injected without opt-in env — "
        "Directive #3 violation"
    )

    # build_image wrapper flag present
    assert getattr(docker_build.build_image, "_sage_ca_wrapped", False) is True

    # Aggregator dict (dockerfiles.__init__._DOCKERFILE_BASE["py"]) must ALSO be
    # patched, otherwise get_dockerfile_base returns the stale original.
    from swebench.harness import dockerfiles as _dockerfiles_pkg
    assert marker in _dockerfiles_pkg._DOCKERFILE_BASE["py"], (
        "aggregator dict still holds pre-patch template — "
        "get_dockerfile_base would return unfixed Dockerfile"
    )


def test_insecure_bypass_adds_ssl_overrides(monkeypatch, tmp_path, caplog):
    """With SAGE_SWEBENCH_ALLOW_INSECURE=1, the Dockerfile gains SSL-bypass
    flags for Miniconda wget, conda, and pip AND a WARNING is logged."""
    bundle = _make_tmp_bundle(tmp_path)
    monkeypatch.delenv("SAGE_SWEBENCH_DISABLE_CA_PATCH", raising=False)
    monkeypatch.setenv("SAGE_SWEBENCH_ALLOW_INSECURE", "1")
    monkeypatch.setenv("SAGE_CORPORATE_CA_BUNDLE", str(bundle))

    mod = _fresh_patch_module()
    with caplog.at_level(logging.WARNING):
        assert mod.apply_corporate_ca_patch() is True

    from swebench.harness.dockerfiles import python as dockerfile_python
    tpl = dockerfile_python._DOCKERFILE_BASE_PY

    # All three bypasses must now be present (opt-in path).
    assert "wget --no-check-certificate 'https://repo.anaconda.com/miniconda/" in tpl
    assert "conda config --set ssl_verify false" in tpl
    assert "pip config --global set global.trusted-host" in tpl

    # A WARNING must have been logged (audit trail for opt-out of Directive #3).
    assert any(
        "ALLOW_INSECURE" in rec.message or "disabling SSL verification" in rec.message
        for rec in caplog.records
    ), "expected a WARNING when SAGE_SWEBENCH_ALLOW_INSECURE=1 is active"


def test_idempotent_second_call_no_double_patch(monkeypatch, tmp_path):
    """Calling apply twice doesn't duplicate the COPY line in the template."""
    bundle = _make_tmp_bundle(tmp_path)
    monkeypatch.delenv("SAGE_SWEBENCH_DISABLE_CA_PATCH", raising=False)
    monkeypatch.setenv("SAGE_CORPORATE_CA_BUNDLE", str(bundle))

    mod = _fresh_patch_module()
    assert mod.apply_corporate_ca_patch() is True
    assert mod.apply_corporate_ca_patch() is True

    from swebench.harness.dockerfiles import python as dockerfile_python
    tpl = dockerfile_python._DOCKERFILE_BASE_PY
    # Only one occurrence of the appended-to system bundle line.
    assert tpl.count("/etc/ssl/certs/ca-certificates.crt") == 1


def test_wrapper_copies_bundle_into_build_dir(monkeypatch, tmp_path):
    """The wrapped build_image copies ca-bundle.crt into build_dir before delegating."""
    bundle = _make_tmp_bundle(tmp_path)
    monkeypatch.delenv("SAGE_SWEBENCH_DISABLE_CA_PATCH", raising=False)
    monkeypatch.setenv("SAGE_CORPORATE_CA_BUNDLE", str(bundle))

    mod = _fresh_patch_module()
    assert mod.apply_corporate_ca_patch() is True

    # Replace the wrapped inner function with a capturing stub, then
    # recall via the wrapper.
    from swebench.harness import docker_build
    wrapped = docker_build.build_image

    build_dir = tmp_path / "build_dir"
    called_with: dict[str, object] = {}

    def _stub_inner(image_name, setup_scripts, dockerfile, platform, client, bd, nocache=False):
        called_with["build_dir"] = bd
        called_with["image_name"] = image_name
        return "OK"

    # Swap out the underlying function
    import sage.bench.swebench_ca_patch as patch_mod
    # The wrapper closed over `original_build_image` at apply time; we can't
    # easily rebind it after the fact. Instead, call the wrapper with a
    # fake inner via an attribute swap (the wrapper just re-calls whatever
    # docker_build.build_image was captured). For this test we verify the
    # COPY side effect, not the inner call: replace the *outer* ref.
    _prev = docker_build.build_image  # the wrapped one
    try:
        # Temporarily stub the underlying by replacing closure variables
        # via a new wrap — simpler: call our own copy logic directly.
        # Reverse path: just verify the wrapper is a callable and that
        # calling it side-effects the build_dir.
        build_dir.mkdir()
        # Call wrapper — the real inner build_image will fail because we
        # don't have Docker objects, but copy happens BEFORE that.
        try:
            wrapped("img", {}, "dockerfile", "linux/amd64", None, build_dir)
        except Exception:
            pass  # inner will fail, copy should still have happened
        assert (build_dir / "ca-bundle.crt").is_file(), (
            "wrapper did not copy ca-bundle.crt into build_dir"
        )
        # Content matches our stub PEM
        assert (build_dir / "ca-bundle.crt").read_text().startswith("-----BEGIN")
    finally:
        docker_build.build_image = _prev


def test_no_bundle_env_uses_default_when_it_exists(monkeypatch, tmp_path):
    """When env var is unset but the default path exists, the default is used."""
    default_bundle = _make_tmp_bundle(tmp_path)
    monkeypatch.delenv("SAGE_CORPORATE_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SAGE_SWEBENCH_DISABLE_CA_PATCH", raising=False)

    mod = _fresh_patch_module()
    monkeypatch.setattr(mod, "_DEFAULT_CA_BUNDLE", default_bundle)
    assert mod.apply_corporate_ca_patch() is True
