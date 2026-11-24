# Session handoff

## Project and publication

- Repository: https://github.com/snejokzboxik/counterrotating-motion
- Branch: main
- Latest published commit before this handoff: e90e82d970f7ea69263ddac0a90e4d30e62d2e6b
- Complete V17C migration, cleanup, and archive integrity audit are documented in migration/MIGRATION_MANIFEST.json.
- All 14 historical V17C refs are retained in the new repository at their original commits; the source V17C refs were removed only after the clone SHA-256 audit and bundle verification.

## Scientific progress

The analytical stage preserves refine12_sobol17 data, proves that the Reynolds coordinate change relocates but does not remove independent dimensionless groups, and defines a conditional (A,B,beta) section. A leading-order two-mode torque model distinguishes the full RF torque zero from the secular torque zero. A leading radial balance predicts nominal radii within 0.32% and reproduces four previously saved holdouts within 0.35%.

A fresh M3 Fourier solve decomposes the eight harmonic torques at the published root. It confirms opposite RF-family torque contributions and nonlinear-drag compensation. A fixed-radius torque scan over orbital rate confirms that the torque varies nonlinearly; its discrete points are not a continuation of stationary roots.

At 96x8x8 with 2x oversampling, both leading harmonics produce a verified counterrotating M3 root. The same root persists with Coulomb set to zero. Single-family calculations did not converge from the joint-root seed; this is a solver status, not evidence that no one-family branch exists.

## Validation

- python -m pytest -q tests: 5 passed.
- New JSON data parses with the strict Python JSON parser.
- Published V20B DOP853 RHS independently checks the nondimensionalization.
- The migration's selected historical checks and full clean-clone SHA-256 audit are documented in the migration report.
- Current environment has no pdflatex; Tectonic cannot provide the missing Russian T2A fonts. The Overleaf ZIP is a draft source package, not a compiled manuscript.

## Remaining scientific work

1. Improve single-family controls by branch continuation or independent seeds; report only roots whose inner HB solves and outer balance both converge.
2. Resolve the full RF-mode removal controls and quantify when higher modes alter the torque sign.
3. Continue the coupled M3 stationary branch through its local torque zeros; verify whether signs change before or after radial roots shift.
4. Compare those boundary predictions against a small set of full-RF integrations, preserving RF phases and reporting slow-orbit counts.
5. Add figures only from computed data. No parameter-space phase diagram has yet been calculated, so the requested broad three-parameter map is outstanding.
6. Install/use an environment with pdfLaTeX and Russian T2A fonts, compile twice, check the output, then rebuild the Overleaf archive.

## Current evidence limits

The result supports a long-lived counterrotating orbit and a local M3 root. It does not establish transverse attraction, a torus of the full forced system, a global basin, collective phase order, or a finite-electrode experimental trap. One-family solver failures do not imply nonexistence.
