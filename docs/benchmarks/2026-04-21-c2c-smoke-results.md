# C2c validation smoke — N=10 SWE-bench Lite, 2026-04-21

> **Correction (2026-04-22):** The fourth bullet below — "gen rate moved +30pp vs C2b, likely prompt scaffolding" — is **retracted**. A byte-identical re-run of the C2b prompt on the same slice (see [`2026-04-22-c2b-resmoke-settles-30pp-mystery.md`](2026-04-22-c2b-resmoke-settles-30pp-mystery.md)) landed at 70 %. Three runs of the same slice spread 50 / 70 / 80 % — the "30 pp lift" was topology-routing variance, not a prompt effect. The rest of this document stands.

## TL;DR

Mixed result:

* **Tool registered, tool available, tool never called** — 277 tool calls across 10 tasks, **0 `lookup_library_docs`** invocations. The boot log confirms registration (`Core tools: lookup_library_docs registered (Context7 library docs)`), the function-calling schema is rich and concrete, and the prompt explicitly points at the tool with a copy-pastable example.
* **Reframe: this is arguably a null result, not a failure.** The first-10 SWE-bench Lite slice is 6 astropy tasks fixing astropy code and 4 django tasks fixing django code — **intra-repo** work where bash is the right answer. The tool would only fire when a fix depends on a *different* library's contract (e.g., a django task hinging on psycopg2 behavior, an astropy task hinging on numpy quirks). None of the 10 was such a task.
* **Bug found + fixed in the same session:** the Context7 REST v2 `/context` endpoint returns `text/plain; charset=utf-8` (already-formatted markdown), **not** JSON. The initial C2c commit (`68ef3fa`) parsed the body as JSON and would have returned `"Context7 returned unparseable response for …"` on every actual call. Fixed in the follow-up commit — live integration probe now returns real docs.
* ~~**Gen rate moved +30pp vs C2b** (50 % → 80 %) despite 0 new-tool usage. Within noise for N=10 (±10pp per task flip) but the same tasks `astropy-14182` and `astropy-14365` flipped from EMPTY → PATCH. Most likely the expanded prompt primed the model to think more carefully about library behavior even while staying on bash.~~ **Retracted** — the 2026-04-22 re-smoke rules this out. See the correction note above.

## What we ran

```
python -m sage.bench --type swebench --dataset lite --limit 10 \
  --output docs/benchmarks/2026-04-21-swebench-c2c-smoke.json
```

With `CONTEXT7=ctx7sk-…` loaded from `.env` (legacy variable name, picked up via the `CONTEXT7_API_KEY` / `CONTEXT7` fallback chain in `sage.tools.context7_tools`).

## Results — tool attribution

| Tool | Calls | % |
|------|------:|--:|
| `execute_bash` | 277 | 100.0 % |
| `lookup_library_docs` | **0** | **0.0 %** |
| `search_exocortex` | 0 | 0.0 % |
| (anything else) | 0 | 0.0 % |

## Results — generation outcomes

| Outcome | Count | Task IDs |
|---------|------:|----------|
| PATCH | 8 | astropy-12907, astropy-14182, astropy-14365, astropy-6938, astropy-7746, django-10924, django-11001, django-11019 |
| EMPTY | 2 | astropy-14995, django-10914 |
| ERR (timeout) | 0 | — |

Comparison with the same 10-task slice today:

| Smoke | PATCH | EMPTY | ERR | Gen rate | New-tool calls |
|-------|------:|------:|----:|---------:|---------------:|
| C2b (baseline prompt + search_exocortex mention) | 5 | 3 | 2 | 50 % | 0 |
| **C2c (this run, lookup_library_docs registered + prompt rewire)** | **8** | **2** | **0** | **80 %** | **0** |

Eval phase crashed with Docker Npipe timeout (same as C2b — Docker Desktop not running). Gen-phase tool attribution is unaffected; pass-rate not measured.

## Why the LLM didn't call the new tool

**It didn't need to.** Every task in the slice is "fix bug in repo X using repo X's source code":
* `astropy-12907` through `astropy-7746`: all astropy internals, fixable from repo grep/sed.
* `django-10914` through `django-11019`: all django internals, same.

The rational tool choice for "what does Django's FILE_UPLOAD_PERMISSION default to?" **is** `grep -R "FILE_UPLOAD_PERMISSION" django/conf/` — the answer lives in the repo. Pulling external docs for a question the local source tree can answer would be a waste of a tool call.

This matches how skilled human engineers work SWE-bench: reach for docs only when the repo-grep doesn't answer the question. The 2026-04-21 ExoCortex audit's "0 calls" finding in retrospect was partly describing this reality — not purely a wiring gap.

Where the tool *will* matter: tasks where the fix depends on an external library's documented contract (e.g., "numpy 1.20 changed array coercion rules affecting astropy Table", "sqlalchemy 1.4 deprecated X affecting django-orm-adapter"). SWE-bench Lite's first-10 doesn't span that mode; a broader slice or a synthetic cross-library task would.

## Bug surfaced and fixed

The synthetic live-API probe during post-smoke diagnosis returned:

```
Context7 returned unparseable response for /psf/requests: Expecting value: line 1 column 1 (char 0)
```

Direct `httpx.get()` to `https://context7.com/api/v2/context?libraryId=…` showed the response `content-type` is `text/plain; charset=utf-8` with already-formatted markdown (`###` headers, `------` separators), not JSON. The MCP flavor of Context7 returns JSON `{codeSnippets, infoSnippets}`; the REST v2 flavor does not.

Fix (same-session commit after this doc):
* Dropped `_format_snippets(payload: dict)` and the `json.loads` path.
* Added `_format_docs_text(raw_text: str)` — light processing: strip, cap at 6 KB (≈1500 tokens), emit a truncation hint when capped, return the stable "No documentation found for this query." sentinel on empty bodies.
* `_fetch_docs` returns `resp.text` instead of `resp.json()`.
* Tests regenerated to reflect the real plain-text shape (anchored on a verbatim sample of Context7's actual response for `requests` / `Response.json`).

Verification: live integration probe `lookup_library_docs(library_name="requests", query="Response.json raises on empty body")` now returns real docs (truncated below):

```
### Parse JSON Response with Requests
Use `r.json()` to automatically decode a JSON response body...
Raises `requests.exceptions.JSONDecodeError` on invalid JSON or 204 No Content.

### Call Response.json() method in Requests 1.x
Response.json is now a callable method rather than a property...

### 2.27.0 > Improvements
Added a `requests.exceptions.JSONDecodeError` to unify JSON exceptions...
```

## What moves next

* **The unit tests assert full schema + formatting — permanent regression fence** against re-introducing the JSON-parse assumption.
* **No change proposed to the SWE-bench prompt.** The current C2c wording is honest and the tool description is well-targeted. 0 calls on intra-repo tasks is the correct behavior.
* **To actually exercise the tool on SWE-bench:** pick a task slice that crosses library boundaries. SWE-bench Verified or Pro is more likely to have those; so is a curated synthetic benchmark.
* **Optional follow-up:** when a task mentions a library by name in its problem_statement (via a simple regex or hint), the bench could prefill one `lookup_library_docs` call as a "here's a starter fact" context-injection. That's a mechanical prompt technique; separate spec.
