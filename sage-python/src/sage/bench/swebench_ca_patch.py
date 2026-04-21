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


def apply_corporate_ca_patch() -> bool:
    """Monkey-patch swebench to accept the corporate CA during Docker builds.

    Does two things:
    1. Injects ``truststore`` into the Python SSL subsystem so the host
       process trusts whatever the OS trusts (fixes requests.get() against
       raw.githubusercontent.com behind a corporate TLS proxy).
    2. Modifies the base Dockerfile to COPY an explicit CA bundle into
       ``/usr/local/share/ca-certificates/`` + run update-ca-certificates
       so ``wget`` inside the container trusts the same chain.

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
    if "corporate.crt" not in original_tpl and marker in original_tpl:
        injected = (
            "# Install corporate CA bundle (SAGE swebench_ca_patch)\n"
            "COPY ca-bundle.crt /usr/local/share/ca-certificates/corporate.crt\n"
            "RUN apt-get update && apt-get install -y ca-certificates "
            "&& update-ca-certificates\n\n"
            + marker
        )
        patched_tpl = original_tpl.replace(marker, injected)
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
