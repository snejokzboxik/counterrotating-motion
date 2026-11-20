# SESSION_HANDOFF — V17C.14

## Current repository state

- Working branch: `research/v17c-theory` (isolated worktree; base is `research/v17c` SHA `fc1d29ecc1067e0191a4cf597189b4973546864f`).
- Destination: `research/v17c/v17c_14/` only.
- Source N=6 artifacts are pinned to `research/v17c-crystals` commit `5438b1cdd79c47abab45c6cbaf99c6daec9fbb60`.
- `master` and source branches are not modified.
- No V17C.14 trajectory process is active or was launched.
- No TeX engine was available, so `theory_ru.tex` and `theory_en.tex` remain uncompiled source drafts.

## Scientific status

The generalized RHS, symbolic identities, frozen-position averaging and figure data are complete for this theory stage. Leading-only local15 gives stable opposite-sign single-particle circular roots. V17C.13 provides one selected long 3+3 run and ten discrete Coulomb sweep records; these are reused without claiming probability or independent reproduction. H4 shows only a fixed-radius axial-extrema window; the full coupled 3D state remains unresolved.

See `REPORT_RU.md`, `THEORY_STATUS_RU.md`, `NUMERICAL_VALIDATION_RU.md` and `FIGURE_AUDIT_RU.md` for exact results and limitations.

## Reproduction

From repository root:

```powershell
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' -m unittest discover -s research/v17c/v17c_14/tests -v
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/symbolic/derive_generalized_identities.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/validate_averaged_harmonic.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/compute_leading_branches.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/make_figures.py
```

All numerical integration counts for V17C.14 are zero. The plotted quadratures and parameter meshes are deterministic analysis, not simulated particle trajectories. PDF/SVG/PNG assets are in `figures/`; raw tables are in `results/figure_data/`.

## Deferred work (not started)

Do not interpret this handoff as permission to begin the next physical-feasibility or trajectory research. The following items require their own scoped instruction and budget:

- independently rerun the six-particle reference, convergence grid and DOP853;
- full Floquet/QR-Lyapunov analysis and nonlinear perturbation tests for time-varying N-body states;
- forward and reverse Coulomb/parameter continuation with multiple initial states;
- full 3D H4/DC-field electrode realization, FEM and field-strength/breakdown analysis;
- physical charge, drag, particle-size and material constraints;
- shared z-separated multi-ring and higher-N stability campaigns.

Parallel work at home on collective 3+3 dynamics must remain separate to avoid duplicate trajectory campaigns.

## Publication state

This directory is prepared for commit and push to `research/v17c-theory`. After publishing, record the final commit SHA and verify it equals the remote branch tip here. Update `MANIFEST.json` after all edits and before commit; regenerate it if any file changes.
