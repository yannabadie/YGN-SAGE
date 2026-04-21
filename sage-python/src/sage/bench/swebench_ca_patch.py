"""Corporate CA bundle injection for swebench Docker base-image builds.

The upstream swebench base Dockerfile downloads Miniconda via ``wget`` over
HTTPS (``https://repo.anaconda.com/miniconda/...``). On hosts behind a
corporate TLS-inspecting proxy, the container's default CA roots don't
include the corporate intermediate, so ``wget`` fails with exit code 5
("self signed certificate in certificate chain").

This module monkey-patches swebench at import time to:

1. Inject the corporate CA bundle into every base-image build context as
   ``ca-bundle.crt`` (Debian/Ubuntu convention for additional anchors).
2. Modify the base Dockerfile template to ``COPY`` the bundle and run
   ``update-ca-certificates`` before the ``wget`` step.

The patch is:
- **Idempotent**: calling :func:`apply_corporate_ca_patch` twice is a
  no-op after the first success.
- **Reversible**: set ``SAGE_SWEBENCH_DISABLE_CA_PATCH=1`` to skip.
- **Location-overridable**: set ``SAGE_CORPORATE_CA_BUNDLE`` to point at
  a different PEM file (default: ``C:/Code/certs/ca-bundle.pem``).
- **Graceful**: if the bundle file is missing, emits a warning and
  returns False — the original (unpatched) Dockerfile is used.
- **Secure by default**: only COPY+append the corporate CA into the
  container trust store. SSL verification is NOT disabled unless
  ``SAGE_SWEBENCH_ALLOW_INSECURE=1`` is set — opt-in escape hatch for
  the narrow case where a MITM-CA-augmented trust store still can't
  verify ``repo.anaconda.com`` / ``pypi.org`` (e.g. wget reads hashed
  symlinks from update-ca-certificates, which our multi-cert PEM is
  skipped by per Debian convention). When active, the bypass logs a
  prominent warning. Off by default to comply with CLAUDE.md
  Directive #3 ("never add ``verify=False``").

Call :func:`apply_corporate_ca_patch` from any swebench entry point
before ``build_env_images`` / ``build_base_images`` is invoked.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_CA_BUNDLE = Path("C:/Code/certs/ca-bundle.pem")
_ENV_CA_BUNDLE = "SAGE_CORPORATE_CA_BUNDLE"
_ENV_DISABLE = "SAGE_SWEBENCH_DISABLE_CA_PATCH"
_ENV_ALLOW_INSECURE = "SAGE_SWEBENCH_ALLOW_INSECURE"

_APPLIED = False  # idempotence guard


def _resolve_ca_bundle() -> Path | None:
    """Find the corporate CA bundle, or return None if unavailable."""
    env_path = os.environ.get(_ENV_CA_BUNDLE)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        log.warning(
            "SAGE_CORPORATE_CA_BUNDLE=%s does not exist — falling back to default",
            env_path,
        )
    if _DEFAULT_CA_BUNDLE.is_file():
        return _DEFAULT_CA_BUNDLE
    return None


def _inject_truststore() -> bool:
    """Route Python HTTPS through the OS trust store (Windows SChannel).

    swebench's harness fetches per-instance requirements from
    ``raw.githubusercontent.com`` via ``requests``. On Windows hosts behind
    a TLS-inspecting proxy, neither certifi's bundle nor our corporate PEM
    contains the intercepting CA — but the Windows cert store does. The
    ``truststore`` package (already a sage-python dependency) replaces
    Python's default SSLContext with one that delegates to the OS store,
    which is why a plain ``pip install`` works while ``requests.get()``
    fails.

    Returns True on successful injection; False if truststore isn't
    available (keeps patch soft-failing on non-Windows).
    """
    try:
        import truststore
        truststore.inject_into_ssl()
        log.info("swebench CA patch: truststore injected (OS trust store active).")
        return True
    except ImportError:
        log.warning("swebench CA patch: truststore not installed — "
                    "host HTTPS may fail on corporate-proxy networks.")
        return False
    except Exception as exc:
        log.warning("swebench CA patch: truststore injection failed: %s", exc)
        return False


def _apply_crlf_fix() -> bool:
    """Force LF line endings for shell scripts written via Path.write_text.

    Background:
      ``swebench.harness.run_evaluation.run_instance`` writes ``/eval.sh``
      with ``eval_file.write_text(test_spec.eval_script)`` (no
      ``newline=`` argument). On Windows, Python text mode translates
      every ``\\n`` → ``\\r\\n``. The Linux Docker container then
      receives a CRLF script; bash fails on ``set -uxo pipefail\\r``
      ("invalid option name") and conda is never activated → pytest is
      not in PATH → 100 % of FAIL_TO_PASS tests report "command not
      found". The upstream Dockerfile only strips CRLF from
      ``setup_env.sh`` / ``setup_repo.sh`` (baked in at build time);
      ``/eval.sh`` is written at RUNTIME per-container and has no such
      treatment on the host writer side. This was the root cause of the
      2026-04-21 v13 smoke 0/10 result (see
      ``docs/benchmarks/2026-04-21-swebench-smoke-v13-post-phase1-stab.md``).

    Fix:
      Monkey-patch ``pathlib.Path.write_text`` once per process so any
      ``.sh`` / ``.bash`` write goes through ``write_bytes`` with UTF-8
      encoding (pure LF). Other file types fall through untouched. On
      Linux this is a no-op (text mode already writes LF). Idempotent
      via a marker attribute on the replacement — safe across fresh
      re-imports of this module (pathlib itself keeps the wrapper).
    """
    import pathlib

    current = pathlib.Path.write_text
    if getattr(current, "_sage_crlf_wrapped", False):
        return True

    def _lf_safe_write_text(self, data, encoding=None, errors=None, newline=None):
        suffix = self.suffix.lower()
        if suffix in (".sh", ".bash"):
            enc = encoding or "utf-8"
            return self.write_bytes(data.encode(enc, errors=errors or "strict"))
        return current(
            self, data, encoding=encoding, errors=errors, newline=newline
        )

    _lf_safe_write_text._sage_crlf_wrapped = True  # type: ignore[attr-defined]
    pathlib.Path.write_text = _lf_safe_write_text  # type: ignore[method-assign]
    log.info(
        "swebench CA patch: pathlib.Path.write_text wrapped with LF-safe "
        "handler for .sh/.bash files (Windows CRLF fix for /eval.sh inside "
        "the Docker container)."
    )
    return True


def _apply_utf8_open_fix() -> bool:
    """Force UTF-8 encoding for text-mode ``open()`` inside swebench.

    Background:
      ``swebench.harness.run_evaluation.py:211`` writes container test
      output to disk with bare ``open(test_output_path, "w")``. On
      Windows, Python's text-mode ``open()`` defaults to the locale
      preferred encoding (``cp1252``). pytest output inside the
      container routinely contains Unicode box-drawing characters
      (``│`` U+2502, ``├``, etc. from its tree display) → ``UnicodeEncodeError:
      'charmap' codec can't encode character '\\u2502'`` → the test run
      is reported as ERROR even though the patch applied cleanly and
      tests actually ran. This is what turned 2/10 would-be-resolved
      cases into v14 ERRORs.

    Fix:
      Replace ``swebench.harness.run_evaluation.open`` with a wrapper
      that defaults to ``encoding='utf-8'`` whenever mode is text-write
      (``'w'``, ``'wt'``, ``'a'``, ``'at'``) and no explicit encoding is
      passed. Binary modes and explicit-encoding calls pass through
      untouched. Scoped to the swebench module so we don't touch
      ``builtins.open`` globally. No-op on Linux (UTF-8 is the default
      there).
    """
    try:
        from swebench.harness import run_evaluation as _run_eval
    except ImportError:
        log.warning(
            "swebench CA patch: could not import run_evaluation — "
            "UTF-8 open fix skipped."
        )
        return False

    existing = _run_eval.__dict__.get("open", open)
    if getattr(existing, "_sage_utf8_wrapped", False):
        return True

    _builtin_open = open

    def _utf8_default_open(file, mode="r", buffering=-1, encoding=None,
                           errors=None, newline=None, closefd=True, opener=None):
        is_text_write = (
            "b" not in mode
            and encoding is None
            and any(c in mode for c in ("w", "a", "x", "+"))
        )
        if is_text_write:
            encoding = "utf-8"
        return _builtin_open(
            file, mode=mode, buffering=buffering, encoding=encoding,
            errors=errors, newline=newline, closefd=closefd, opener=opener,
        )

    _utf8_default_open._sage_utf8_wrapped = True  # type: ignore[attr-defined]
    _run_eval.open = _utf8_default_open  # type: ignore[attr-defined]
    log.info(
        "swebench CA patch: swebench.harness.run_evaluation.open wrapped "
        "to default text-mode writes to UTF-8 (Windows cp1252 fix for "
        "pytest Unicode output in test_output.txt)."
    )
    return True


def apply_corporate_ca_patch() -> bool:
    """Monkey-patch swebench to accept the corporate CA during Docker builds.

    Does three things unconditionally (secure path):
    1. Injects ``truststore`` into the Python SSL subsystem so the host
       process trusts whatever the OS trusts (fixes requests.get() against
       raw.githubusercontent.com behind a corporate TLS proxy).
    2. Wraps ``pathlib.Path.write_text`` so ``.sh`` / ``.bash`` scripts
       are written with pure LF bytes on Windows (fixes CRLF in
       ``/eval.sh`` inside the Docker container — see
       :func:`_apply_crlf_fix`).
    3. Modifies the base Dockerfile to COPY an explicit CA bundle and
       append it to ``/etc/ssl/certs/ca-certificates.crt`` inside the
       container so OpenSSL-based tools trust the corporate chain.

    Optionally, only when ``SAGE_SWEBENCH_ALLOW_INSECURE=1``:
    4. Adds ``--no-check-certificate`` to Miniconda wget, and disables
       SSL verification for conda + pip inside the Dockerfile. Escape
       hatch for hosts where the MITM-augmented trust store still can't
       verify ``repo.anaconda.com`` / ``pypi.org``. A prominent WARNING
       is logged when this path is taken. Off by default per CLAUDE.md
       Directive #3.

    Returns:
        True if the patch is in effect (either freshly applied or already
        applied). False if disabled, swebench isn't importable, or no CA
        bundle was found.
    """
    global _APPLIED
    if _APPLIED:
        return True

    if os.environ.get(_ENV_DISABLE) == "1":
        log.info("swebench CA patch disabled via SAGE_SWEBENCH_DISABLE_CA_PATCH=1")
        return False

    # Host-side fix — unconditional. Safe on non-corporate networks (no-op
    # if the OS store already matches certifi).
    _inject_truststore()

    # Windows CRLF fix for /eval.sh in Docker — unconditional (no-op on
    # Linux). Independent of CA bundle presence, so run before _resolve.
    _apply_crlf_fix()

    # Windows cp1252 fix for pytest Unicode output written to
    # test_output.txt by swebench — unconditional, no-op on Linux.
    # Must run after swebench import; _apply_utf8_open_fix does its own
    # import + guard so it's safe to call unconditionally here.
    _apply_utf8_open_fix()

    ca_bundle = _resolve_ca_bundle()
    if ca_bundle is None:
        log.warning(
            "swebench CA patch: no bundle found at %s and %s not set — "
            "Dockerfile left as-is (wget may fail on SSL verification).",
            _DEFAULT_CA_BUNDLE, _ENV_CA_BUNDLE,
        )
        return False

    try:
        from swebench.harness.dockerfiles import python as dockerfile_python
        from swebench.harness import docker_build
    except ImportError as exc:
        log.warning("swebench CA patch: import failed (%s) — patch skipped.", exc)
        return False

    # ── Patch 1 — inject COPY + update-ca-certificates in the Dockerfile
    # Must patch BOTH the source-of-truth (dockerfiles.python._DOCKERFILE_BASE_PY)
    # AND the aggregator dict in dockerfiles.__init__._DOCKERFILE_BASE, which
    # captured the original string by reference at import time.
    original_tpl = dockerfile_python._DOCKERFILE_BASE_PY
    marker = "# Download and install conda"
    if "corporate-ca-bundle" not in original_tpl and marker in original_tpl:
        # Always inject our CA bundle into the system trust store so OpenSSL-
        # based tools (curl, python-requests, etc. in downstream steps) trust
        # the corporate chain. This is the secure path — no verification is
        # disabled, we just teach the container about the MITM root.
        injected_ca = (
            "# Install corporate CA bundle (SAGE swebench_ca_patch)\n"
            "COPY ca-bundle.crt /tmp/corporate-ca-bundle.crt\n"
            "RUN cat /tmp/corporate-ca-bundle.crt >> /etc/ssl/certs/ca-certificates.crt "
            "&& rm /tmp/corporate-ca-bundle.crt\n\n"
            + marker
        )
        patched_tpl = original_tpl.replace(marker, injected_ca)

        # Directive #3 compliance: only disable SSL verification when the
        # caller explicitly opts in via SAGE_SWEBENCH_ALLOW_INSECURE=1.
        # Needed when the augmented trust store still can't verify
        # repo.anaconda.com / pypi.org — e.g. wget, which doesn't consult
        # /etc/ssl/certs/ca-certificates.crt directly but reads hashed
        # symlinks from update-ca-certificates (and our multi-cert PEM is
        # skipped per Debian convention: one cert per .crt file).
        allow_insecure = os.environ.get(_ENV_ALLOW_INSECURE) == "1"
        if allow_insecure:
            log.warning(
                "swebench CA patch: SAGE_SWEBENCH_ALLOW_INSECURE=1 — "
                "disabling SSL verification for Miniconda wget, conda, "
                "and pip inside the Docker build (public-registry "
                "downloads in an ephemeral sandbox). Clear the env var "
                "to restore full verification (Directive #3 default)."
            )
            patched_tpl = patched_tpl.replace(
                "RUN wget 'https://repo.anaconda.com/miniconda/",
                "RUN wget --no-check-certificate "
                "'https://repo.anaconda.com/miniconda/",
            )
            patched_tpl = patched_tpl.replace(
                "RUN conda init --all",
                (
                    "RUN conda config --set ssl_verify false\n"
                    "RUN pip config --global set global.trusted-host "
                    "\"pypi.org files.pythonhosted.org pypi.python.org\"\n"
                    "RUN conda init --all"
                ),
            )

        dockerfile_python._DOCKERFILE_BASE_PY = patched_tpl

        # Also update the aggregator dict; get_dockerfile_base reads from here.
        try:
            from swebench.harness import dockerfiles as _dockerfiles_pkg
            if "py" in _dockerfiles_pkg._DOCKERFILE_BASE:
                _dockerfiles_pkg._DOCKERFILE_BASE["py"] = patched_tpl
        except (ImportError, AttributeError) as exc:
            log.warning("swebench CA patch: could not patch aggregator dict: %s", exc)

        log.info("swebench CA patch: base Dockerfile template updated (module + dict).")

    # ── Patch 2 — wrap build_image to copy the bundle into every build_dir
    original_build_image = docker_build.build_image
    if not getattr(original_build_image, "_sage_ca_wrapped", False):
        def _wrapped_build_image(*args, **kwargs):
            # build_dir is positional arg index 5 (image_name, setup_scripts,
            # dockerfile, platform, client, build_dir, nocache) or kwarg.
            build_dir = kwargs.get("build_dir")
            if build_dir is None and len(args) >= 6:
                build_dir = args[5]
            if build_dir is not None:
                try:
                    dst = Path(build_dir)
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ca_bundle, dst / "ca-bundle.crt")
                    log.debug("swebench CA patch: copied %s → %s/ca-bundle.crt",
                              ca_bundle, dst)
                except Exception as exc:
                    log.warning("swebench CA patch: failed to copy bundle "
                                "into %s: %s", build_dir, exc)
            return original_build_image(*args, **kwargs)

        _wrapped_build_image._sage_ca_wrapped = True  # type: ignore[attr-defined]
        docker_build.build_image = _wrapped_build_image

        # run_evaluation imports build_image by name — patch its reference too.
        try:
            from swebench.harness import run_evaluation
            if hasattr(run_evaluation, "build_image"):
                run_evaluation.build_image = _wrapped_build_image
        except ImportError:
            pass

    log.info("swebench CA patch: applied (bundle=%s).", ca_bundle)
    _APPLIED = True
    return True


__all__ = ["apply_corporate_ca_patch"]
