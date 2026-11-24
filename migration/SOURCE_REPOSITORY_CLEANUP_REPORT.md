# Source repository cleanup

After migration gates A–F passed, the 14 V17C refs listed in `BRANCH_HISTORY_MAP.md` were individually rechecked and removed from `snejokzboxik/three-phase-12-electrode-trap` with ordinary Git pushes. No wildcard or force push was used. The per-ref log records the exact deleted source SHA and the preserved target SHA.

A fresh remote listing after cleanup contains only:

- `master` at `e5e6b2473bf519c5670ee1c48882183fbcfc8f2d`;
- `research/v17b` at `9e49ae1d921841a62fbe5d30880722dc8660ed6d`.

Neither current branch contains `research/v17c/`; therefore no master cleanup commit was needed. The source master SHA did not change. The new repository retains `main` and all 14 V17C historical refs, each at its original commit SHA. The temporary `migration/bootstrap` target ref was removed after main publication.

Historical objects may remain reachable through GitHub storage until garbage collection; no history rewriting or force push was attempted.
