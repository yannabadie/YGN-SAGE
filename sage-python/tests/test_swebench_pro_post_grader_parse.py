"""RESOLUTION_UNBLOCKERS criteria 5-6 — post-grader verdict taxonomy.

cgpro post-run 2026-06-10 (conv cgpro_b2_unblockers_verify) Q3 findings on
the first fully-graded N=5:

- teleport's ``output.json`` collapsed to ``NO_TESTS_FOUND_OR_PARSING_ERROR``
  while its stdout carried the real causal signal
  (``[build failed]``) — the machine artifact lost what the human read;
- tutanota-219's first causal diagnostic was ``error TS2551 ... Did you
  mean '_message'?`` BEFORE the make/native noise;
- NodeBB's local result write crashed on cp1252 (emoji log) leaving NO
  machine-readable output at all;
- every negative case must still produce a normalized verdict.

The parser normalizes each graded instance dir into a verdict with an
explicit taxonomy + the FIRST compiler/test error (advisory fields for
repair-loop feedback and triage — NOT gate-bearing).

Fixture shapes replicate the committed bundle
``docs/benchmarks/2026-06-10-phase2a-n5-graded/grading/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sage-python" / "scripts"))

import swebench_pro_post_grader_parse as parser_mod  # noqa: E402

PREFIX = "ygn-sage-arm-d-smoke"


def _make_instance_dir(
    grading_dir: Path,
    instance_id: str,
    *,
    stdout: str | None = None,
    output_json: dict | None = None,
    patch: str = "x",
) -> Path:
    inst = grading_dir / instance_id
    inst.mkdir(parents=True, exist_ok=True)
    (inst / f"{PREFIX}_patch.diff").write_text(patch, encoding="utf-8")
    if stdout is not None:
        (inst / f"{PREFIX}_stdout.log").write_text(stdout, encoding="utf-8")
    if output_json is not None:
        (inst / f"{PREFIX}_output.json").write_text(
            json.dumps(output_json), encoding="utf-8"
        )
    return inst


def _run(grading_dir: Path, predictions: list[dict], eval_results: dict) -> dict:
    pred_path = grading_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")
    eval_path = grading_dir / "eval_results.json"
    eval_path.write_text(json.dumps(eval_results), encoding="utf-8")
    out_path = grading_dir / "graded_verdicts.json"
    rc = parser_mod.main(
        [
            "--grading-dir", str(grading_dir),
            "--predictions", str(pred_path),
            "--eval-results", str(eval_path),
            "--output", str(out_path),
        ]
    )
    assert rc == 0
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_build_failed_beats_opaque_no_tests_bucket(tmp_path) -> None:
    """teleport shape: output.json says NO_TESTS_FOUND_OR_PARSING_ERROR but
    stdout carries the causal '[build failed]' — the verdict must surface
    BUILD_FAILED, and first_compiler_error must quote the Go build line."""
    iid = "instance_teleport"
    _make_instance_dir(
        tmp_path, iid,
        stdout=(
            "ok github.com/gravitational/teleport/lib/auth 1.2s\n"
            "FAIL\tgithub.com/gravitational/teleport/lib/benchmark [build failed]\n"
            "lib/benchmark/linear.go:120:5: undefined: someSymbol\n"
        ),
        output_json={"tests": [{"name": "test/unknown | error", "status": "ERROR"}]},
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={iid: False},
    )
    v = verdicts[iid]
    assert v["resolved"] is False
    assert v["verdict"] == "BUILD_FAILED"
    assert "[build failed]" in v["first_compiler_error"]


def test_typescript_first_compiler_error_extracted(tmp_path) -> None:
    """tutanota-219 shape: the first TS diagnostic is the causal signal,
    extracted BEFORE the native make noise."""
    iid = "instance_tutanota_219"
    _make_instance_dir(
        tmp_path, iid,
        stdout=(
            "compiling...\n"
            "src/api/EventBusClient.ts(88,14): error TS2551: Property "
            "'_onMessage' does not exist on type 'EventBusClient'. "
            "Did you mean '_message'?\n"
            "Server: Builder: stack Error: `make` failed with exit code: 2\n"
        ),
        output_json={"tests": [{"name": "test/unknown | error", "status": "ERROR"}]},
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={iid: False},
    )
    v = verdicts[iid]
    assert v["verdict"] == "BUILD_FAILED"
    assert "TS2551" in v["first_compiler_error"]
    assert "Did you mean" in v["first_compiler_error"]


def test_patch_apply_failure_classified(tmp_path) -> None:
    iid = "instance_apply_fail"
    _make_instance_dir(
        tmp_path, iid,
        stdout=(
            "Applying patch...\n"
            "error: patch failed: src/mod.py:10\n"
            "error: src/mod.py: patch does not apply\n"
        ),
        output_json=None,
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={iid: False},
    )
    assert verdicts[iid]["verdict"] == "PATCH_APPLY_FAILED"


def test_empty_patch_takes_precedence(tmp_path) -> None:
    """protonmail shape: empty patch is its own verdict even when the
    sandbox produced test noise."""
    iid = "instance_protonmail"
    _make_instance_dir(
        tmp_path, iid,
        stdout="Test failed: packages/components/whatever\n",
        output_json=None,
        patch="",
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": ""}],
        eval_results={iid: False},
    )
    assert verdicts[iid]["verdict"] == "EMPTY_PATCH"


def test_grader_output_write_failed_when_machine_artifact_missing(tmp_path) -> None:
    """NodeBB cp1252 shape: non-empty patch, NO output.json — the wrapper
    must still emit a machine verdict (GRADER_OUTPUT_WRITE_FAILED) rather
    than nothing."""
    iid = "instance_nodebb"
    _make_instance_dir(
        tmp_path, iid,
        stdout="info: NodeBB Ready\n4 passing\n2 failing\n",
        output_json=None,
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={iid: False},
    )
    v = verdicts[iid]
    assert v["verdict"] == "GRADER_OUTPUT_WRITE_FAILED"
    assert v["first_test_error"] is not None


def test_test_failed_with_first_test_error(tmp_path) -> None:
    """tutanota-db90 shape: applied cleanly, tests ran, f2p unresolved —
    TEST_FAILED with the first failing-test diagnostic captured."""
    iid = "instance_db90"
    _make_instance_dir(
        tmp_path, iid,
        stdout=(
            "> tutanota-3 test\n"
            "test/tests/login/LoginViewModelTest.ts(40,8): error TS2554: "
            "Expected 2 arguments, but got 1.\n"
        ),
        output_json={
            "tests": [
                {"name": "api/main", "status": "PASSED"},
                {"name": "test/tests/login", "status": "ERROR"},
            ]
        },
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={iid: False},
    )
    v = verdicts[iid]
    # TS error in a TEST file during the test phase: compiler-class signal
    # still captured; verdict stays BUILD_FAILED only when the build/compile
    # phase itself broke — here tests RAN (output.json has results), so the
    # verdict is TEST_FAILED with the diagnostic in first_compiler_error.
    assert v["verdict"] == "TEST_FAILED"
    assert "TS2554" in (v["first_compiler_error"] or "")


def test_resolved_instance_verdict(tmp_path) -> None:
    iid = "instance_win"
    _make_instance_dir(
        tmp_path, iid,
        stdout="all good\n",
        output_json={"tests": [{"name": "t1", "status": "PASSED"}]},
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={iid: True},
    )
    assert verdicts[iid]["verdict"] == "RESOLVED"
    assert verdicts[iid]["resolved"] is True


def test_every_instance_gets_a_verdict_even_without_artifacts(tmp_path) -> None:
    """Criterion 6: ALL negative grading cases write machine-readable
    output — an instance with no grading dir at all still gets a verdict."""
    iid = "instance_ghost"
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={},
    )
    v = verdicts[iid]
    assert v["verdict"] == "GRADER_OUTPUT_WRITE_FAILED"
    assert v["resolved"] is False


def test_legit_test_name_containing_error_is_not_opaque(tmp_path) -> None:
    """Review MAJOR (2026-06-10): a real test literally named
    'error_handling_test' must count as a REAL test result — the verdict
    stays TEST_FAILED even when stdout carries compiler-class noise."""
    iid = "instance_legit_error_name"
    _make_instance_dir(
        tmp_path, iid,
        stdout=(
            "running tests...\n"
            "src/x.ts(1,1): error TS2554: Expected 2 arguments, but got 1.\n"
        ),
        output_json={
            "tests": [
                {"name": "error_handling_test", "status": "ERROR"},
                {"name": "test_error_recovery", "status": "ERROR"},
            ]
        },
    )
    verdicts = _run(
        tmp_path,
        predictions=[{"instance_id": iid, "patch": "--- a/x\n+++ b/x\n"}],
        eval_results={iid: False},
    )
    assert verdicts[iid]["verdict"] == "TEST_FAILED"
