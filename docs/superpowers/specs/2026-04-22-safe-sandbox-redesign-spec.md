# Safe sandbox redesign — P0.1 + P0.4 spec (2026-04-22)

> **Status:** spec only. Implementation is a follow-up sprint (1-2 weeks) and is **not** shipped in this audit-remediation batch. The audit items P0.1 (replace `execute_bash` with typed tools) and P0.4 (make Wasm sandbox mandatory / fail-closed) collapse into this one design because they address the same threat model.
>
> What IS shipped on 2026-04-22:
> * P0.2 subprocess env allowlist (scrubs API keys from any bash command)
> * P0.3 `execute_raw` gated by `SAGE_UNSAFE_RAW_EXEC` (denies by default)
> * Chat-mode deployment safety via `CHAT_DEFAULT_TOOLS` (excludes `execute_bash`)
> * Deprecation notice added to the `execute_bash` tool description so LLMs see it
>
> What's deferred to follow-up work:
> * Typed-tool library (this spec, P0.1)
> * Mandatory Wasm + fail-closed architecture (this spec, P0.4)

---

## Threat model

Today, every non-chat `AgentSystem.run()` call loads an `execute_bash` tool that forwards model-generated commands to `bash -c`. Post-2026-04-22 the env is scrubbed (no API-key leak) but the LLM can still:

* Read any file the sage user can read (repo state, `~/.ssh`, other projects).
* Write / delete files in the working directory (modify the repo, destroy test fixtures, overwrite outputs).
* Spawn child processes (fork bombs, crypto miners).
* Open network connections (any remote resource the host can reach).
* Mutate shared services (running DB locally, docker daemon, systemd, registry).

The `execute_raw` PyO3 bypass is gated but ToolExecutor's Wasm path is still never taken on default construction — subprocess is the actual runtime. Every claim of "sandboxed execution" in our docs is currently wishful; the substrate is timeout-bounded but otherwise unrestricted.

## Design goals

1. **Fail closed.** If isolation is unavailable, execution must be denied, not silently downgraded to an unsandboxed subprocess.
2. **Minimum-tool-surface principle.** The typed-tool library covers the 95% of SWE-bench / code-bench use cases we know about (read file, search repo, run tests, apply patch, git diff). `execute_bash` stays available but behind an explicit `dangerous_tools=True` flag on AgentConfig for users who need it.
3. **Deterministic behavior.** Tool call → typed response, no hidden state. Makes benchmarking + replay tractable.
4. **Revertible.** Every typed tool is a strict refinement of what the model could express with `bash`. A one-line flag keeps bash available during the transition.

## Typed-tools library (P0.1)

### Tool inventory — MVP set of 6

| Name | Arg schema | Backed by | Threat closure |
|---|---|---|---|
| `read_file` | `path: str, max_bytes: int = 32768` | Host filesystem with path-jail | No arbitrary command, no process spawn, no writes |
| `search_repo` | `query: str, path: str = ".", max_results: int = 40, regex: bool = false` | `ripgrep` subprocess with fixed args | No LLM-controlled flags, arg passed as string literal |
| `list_files` | `path: str, pattern: str = "**/*", max: int = 200` | Python `pathlib.rglob` | No shell interpretation, no command injection |
| `run_tests` | `path: str, pytest_args: list[str] = []` | `pytest` subprocess with curated arg allowlist (`-k`, `-x`, `--tb`) | Only pytest, only typed flags, output capped to 20 KB |
| `apply_patch` | `diff: str, check_only: bool = false` | `git apply` or `patch --fuzz=5` | Rejects patches touching paths outside `{cwd}` / `{cwd}/..` |
| `git_diff` | `path: str = "", staged: bool = false` | `git diff` subprocess (curated flags) | Read-only |

Each tool's handler:
* Validates args against its pydantic schema before subprocess spawn.
* Passes **argv as a list** (never `shell=True`), no string interpolation.
* Inherits the same allowlisted env as the 2026-04-22 `_execute_bash_handler`.
* Caps output, runs under the existing 30s timeout.
* Logs a `tool.call name=X args_keys=[...] output_len=N` line (already wired per commit `571e8d6`).

### AgentConfig changes

```python
@dataclass
class AgentConfig:
    tools: list[str] | None = None  # existing filter
    dangerous_tools: bool = False   # NEW — gates execute_bash into the registry

# boot.py:
if config.dangerous_tools:
    tool_registry.register(bash_tool)  # existing behaviour
else:
    # Typed tools only. Bash remains available behind SAGE_CHAT_ALLOW_BASH
    # per the C2c chat-only pivot (already shipped).
    for tool in create_typed_repo_tools():
        tool_registry.register(tool)
```

### Migration path

1. Implement the 6 typed tools (file: `sage-python/src/sage/tools/typed_repo.py`).
2. Add them to `CHAT_DEFAULT_TOOLS` (replaces `execute_bash` in chat mode already — they're all read-only or pytest-scoped, perfect for chat).
3. Update `SWEBENCH_SYSTEM_TEMPLATE` in `sage.input.swebench` to advertise typed tools in the workflow.
4. Run a paired smoke: N=50 SWE-bench Lite with typed tools vs N=50 with `execute_bash`, same slice. Document in `docs/benchmarks/2026-MM-DD-typed-tools-validation.md`. Cost ~$80, wallclock ~4h.
5. If typed tools match or beat the bash baseline on pass rate, flip `dangerous_tools` default from `False` to `True` for bench adapters that explicitly need bash, and delete `execute_bash` from the default tool list.

## Mandatory Wasm + fail-closed (P0.4)

### Current broken contract

`sage-core/src/sandbox/tool_executor.rs:59-74` constructs a `ToolExecutor` with `wasm_component: None`. The public flow in `validate_and_execute`:
* If `wasm_component.is_some()` → try Wasm.
* On ANY Wasm error → fall through to `execute_python_subprocess`.
* If `wasm_component` was None from the start → skip the Wasm branch entirely, fall through to subprocess.

So the "3-layer defense-in-depth" is: AST validation (optional) → Wasm (almost never present) → **always** subprocess. The only real protection is the AST validation layer, which is bypassable via stdlib modules (`urllib`, `codecs`, `sys._getframe`).

### Target contract

```rust
pub struct ToolExecutor {
    wasm_component: Arc<wasmtime::component::Component>,  // NOT Option anymore
    wasm_engine: Arc<wasmtime::Engine>,
    timeout_secs: u32,
    // ...
}

impl ToolExecutor {
    pub fn new_with_wasm(
        compiled_bytes: &[u8],
        timeout_secs: u32,
    ) -> Result<Self, InitError> {
        // Returns an error if Wasm engine construction or component
        // deserialization fails. No silent None fallback.
    }

    // Existing `new()` constructor deprecated with a loud warning;
    // bench paths use `new_with_wasm()` + a bundled Python-tools
    // component shipped with the wheel (or `new_unsafe_unsandboxed()`
    // with `SAGE_UNSAFE_UNSANDBOXED=1`).
}

fn validate_and_execute(&self, ...) -> ExecResult {
    // Validation runs as before.
    match self.execute_wasm_internal(...) {
        Ok(r) => r,
        Err(e) => {
            // NO subprocess fallback. Fail closed.
            ExecResult {
                stdout: String::new(),
                stderr: format!("Wasm execution failed ({}). Sandbox error; no subprocess fallback available. Set SAGE_UNSAFE_UNSANDBOXED=1 to explicitly downgrade.", e),
                exit_code: -1,
                timed_out: false,
                duration_ms: 0,
            }
        }
    }
}
```

### Bundled Wasm component

The tools library above (`read_file`, `search_repo`, `list_files`, `run_tests`, `apply_patch`, `git_diff`) is implemented once in Python, compiled to Wasm via componentize-py or Rust (if we decide to rewrite in Rust), and the resulting `.wasm` artifact is shipped with the sage-core wheel. `ToolExecutor::new_with_wasm()` loads this bundled artifact on default paths; advanced users can supply their own component with a custom typed-tool set.

### Implementation tasks

1. Write + compile the Wasm typed-tool component (~500 LOC Rust or ~300 LOC Python-via-componentize-py).
2. Embed the component bytes in the sage-core wheel via `include_bytes!`.
3. Refactor `ToolExecutor` to take `Arc<Component>` (not `Option`).
4. Update every call site (`ToolForge`, `phases.act`, `bench.*`) to pass the bundled component.
5. Delete the subprocess fallback from `validate_and_execute`. Keep `execute_python_subprocess` as a standalone helper only reachable via `SAGE_UNSAFE_UNSANDBOXED=1`.
6. Add integration tests: seccomp / filesystem / network each confirmed denied.
7. Run the SWE-bench paired smoke from the P0.1 section with Wasm enabled, verify pass rate is preserved (Wasm overhead < 200ms per call).
8. Flip the sandbox feature to default in `Cargo.toml`.

### Estimated effort

* P0.1 (typed-tool Python library + tests): 4-5 days
* P0.4 (Wasm component + refactor + tests): 5-6 days
* Paired smoke validation: 4-6 hours
* Documentation + migration guide: 1 day

**Total: ~2 weeks engineering + ~1 day of validation compute.**

## Not in scope for this spec

* **Firecracker / gVisor hypervisor-level isolation.** Wasm + deny-by-default-WASI already removes 99% of the exfiltration surface for our threat model (LLM-generated code attempting to read secrets or modify shared state). A hypervisor boundary is defense in depth beyond what the C2c pivot and typed-tools library deliver.
* **Prompt-injection detection.** Orthogonal; the audit flagged it (A.8) but the fix is at the LLM-call layer, not the sandbox.
* **Typed arg schemas in the registry itself.** The existing `ToolDef.parameters` (JSON Schema) is fine for LLM tool-calling; what we need is a typed Python handler dispatcher. Adding a generic `pydantic.BaseModel` schema wrap can come in a cleanup pass after the primary refactor.

## Decision gate

Do NOT merge the typed-tools → mandatory-Wasm transition until:

1. The paired smoke in step 4 of the migration path shows **pass-rate parity** (± 2 pp tolerance). If typed tools lose > 2 pp vs bash on SWE-bench Lite, fix the typed-tools library before proceeding. The audit's core worry is agent utility regression hidden behind marketing claims of "safer execution".
2. A 72-hour adversarial test against the Wasm sandbox with a curated malicious-prompt set shows zero data exfiltration. This is red-team work, not just "does pytest pass".
3. Every existing bench (SWE-bench Lite, BCB Hard, MASBENCH, routing GT) reports matching numbers to within each bench's run-to-run variance band (SWE-bench: ±10 pp per-task flip on N=10; N=50 narrows to ±4 pp). Recorded in `docs/benchmarks/2026-MM-DD-pre-sandbox-vs-post.md`.

Anything less and we ship a regression disguised as a security improvement.

---

## Related audit context

* [`docs/audits/2026-04-22-audit-verification-master.md`](../audits/2026-04-22-audit-verification-master.md) — item A.1–A.5 and the P0.1 / P0.4 rows of the prioritized action plan.
* AUDIT-SEC §2 (full sandbox reality-check) — the ground-truth analysis of today's subprocess fallback path.
* `docs/audits/bypass-patterns.md` — the checklist of silent-bypass patterns to apply before declaring this work complete.

---

## Changelog

* 2026-04-22: Spec written. P0.1 + P0.4 implementation deferred; this document is the deliverable for that pair of audit items within the 2026-04-22 audit-remediation session.
