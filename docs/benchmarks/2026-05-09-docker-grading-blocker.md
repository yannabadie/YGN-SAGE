# SWE-bench Pro Grader Preflight - NO_GO_LOCAL_DOCKER

**Date**: 2026-05-09  
**Commit under test**: `80681595214cc22b75ceb4d1105cb478cbfd3d50`  
**Artifact**: `docs/benchmarks/2026-05-09-swebench-pro-grader-preflight.json`  
**Decision**: `NO_GO_LOCAL_DOCKER`

## Command

```powershell
cd sage-python
python scripts\swebench_pro_grader_preflight.py --output ..\docs\benchmarks\2026-05-09-swebench-pro-grader-preflight.json --json
```

Observed exit code: `2` (expected for a blocked preflight).

## Blockers

1. Docker Desktop Linux daemon is not reachable.

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: Le fichier specifie est introuvable.
```

2. The local SWE-bench Pro grader checkout is dirty:

```text
external/SWE-bench_Pro-os @ 0c64e26f00b9c190432de7fc520c8ceed5c25518
M swe_bench_pro_eval.py
```

3. Modal is installed but not authenticated:

```text
modal client version: 1.3.5
modal token info -> Token missing. Could not authenticate client.
```

## Consequence

Do not launch or label a local SWE-bench Pro N=5/N=50 result as official graded evidence from this host until the preflight returns `READY_LOCAL_DOCKER`, or until a clean remote Linux Docker / verified Modal grading path is documented with its own preflight artifact.

Generate-only traces may continue if they are explicitly marked ungraded. They must not update `delivered`, `default-on`, or performance claims.

## Next Grading Path

Preferred next path for official evidence:

1. Use a clean Linux Docker runner, ideally GitHub Actions or another reproducible Linux host.
2. Clone `scaleapi/SWE-bench_Pro-os` at a pinned commit with no dirty patch.
3. Run `swebench_pro_grader_preflight.py` equivalent on that host.
4. Only then run the N=5 canary grading step.

Modal remains a candidate only after `modal token info` is green and a minimal one-task Pro grader job is verified.
