# SWE-bench Pro Grader Preflight - NO_GO_LOCAL_DOCKER

**Date**: 2026-05-09  
**Commit under test**: `60f8438028f048179d9c9d91b31a5f42b3f55bb5`
**Artifact**: `docs/benchmarks/2026-05-09-swebench-pro-grader-preflight.json`  
**Decision**: `NO_GO_LOCAL_DOCKER`

## Command

```powershell
cd sage-python
python scripts\swebench_pro_grader_preflight.py --output ..\docs\benchmarks\2026-05-09-swebench-pro-grader-preflight.json --json
```

Observed exit code: `2` (expected for a blocked preflight).

## Blockers

1. Local free disk is below the SWE-bench Docker minimum used by this gate:

```text
free_disk_gb=19.21
min_free_disk_gb=120.0
```

2. Docker Desktop Linux daemon is not reachable. The active Docker context is
   local `desktop-linux`, endpoint `npipe:////./pipe/dockerDesktopLinuxEngine`,
   so this is a daemon availability blocker, not a remote-context ambiguity.

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: Le fichier specifie est introuvable.
```

3. The local SWE-bench Pro grader checkout is dirty. Its remote and commit are
   now explicitly checked and match the pinned upstream provenance, but the
   dirty local patch still blocks official grading:

```text
external/SWE-bench_Pro-os @ 0c64e26f00b9c190432de7fc520c8ceed5c25518
remote=https://github.com/scaleapi/SWE-bench_Pro-os
M swe_bench_pro_eval.py
```

4. Modal is installed but not authenticated:

```text
modal client version: 1.3.5
modal token info -> Token missing. Could not authenticate client.
```

## Consequence

Do not launch or label a local SWE-bench Pro N=5/N=50 result as official graded evidence from this host until the preflight returns `READY_LOCAL_DOCKER`, or until a clean remote Linux Docker / verified Modal grading path is documented with its own preflight artifact.

Generate-only traces may continue if they are explicitly marked ungraded. They must not update `delivered`, `default-on`, or performance claims.

## cgpro VERIFY follow-up

cgpro accepted the original negative artifact only for the narrow
`NO_GO_LOCAL_DOCKER` conclusion, not as benchmark readiness. Follow-up
hardening added before this artifact was regenerated:

- expected SWE-bench Pro grader remote + commit checks;
- active Docker context / endpoint / `DOCKER_HOST` recording and remote-context blocker;
- `docker run --rm hello-world` smoke requirement before future `READY_LOCAL_DOCKER`;
- local disk-space gate;
- tighter wording that generic GitHub Actions is not enough unless the runner
  has sufficient disk/RAM.

## Next Grading Path

Preferred next path for official evidence:

1. Use a clean Linux Docker runner with proven disk/RAM capacity: self-hosted,
   GitHub larger runner, or another reproducible Linux host with a recorded
   preflight artifact. Do not assume a standard GitHub-hosted runner has enough
   disk for SWE-bench Pro Docker images.
2. Clone `scaleapi/SWE-bench_Pro-os` at a pinned commit with no dirty patch.
3. Run `swebench_pro_grader_preflight.py` equivalent on that host.
4. Only then run the N=5 canary grading step.

Modal remains a candidate only after `modal token info` is green and a minimal one-task Pro grader job is verified.
