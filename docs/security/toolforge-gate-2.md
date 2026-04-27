# ToolForge Gate 2 Execution Isolation

ToolForge Gate 2 executes generated tests supplied with a generated tool body.
Those tests are user-authored/untrusted code and must run through
`sage.sandbox.isolated_executor`.

If `sage.sandbox.isolated_executor` is unavailable, ToolForge fails closed by
default and refuses to run the tests in a plain Python subprocess.

For local development only, the legacy fallback can be enabled explicitly:

```powershell
$env:SAGE_UNSAFE_TOOLFORGE_SUBPROCESS = "1"
```

This fallback has no filesystem jail, namespace isolation, seccomp boundary, or
bubblewrap/wasm runner. Do not enable it in production or shared execution
contexts.

The opt-in only applies when `sage.sandbox.isolated_executor` cannot be
imported. If the isolated executor imports but fails at runtime, ToolForge still
fails closed and does not downgrade to the unsafe subprocess path.
