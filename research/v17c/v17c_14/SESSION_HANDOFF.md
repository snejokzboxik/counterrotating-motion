# SESSION_HANDOFF — V17C.14

## Current repository state

- Working branch: `research/v17c-theory` (isolated worktree; base is `research/v17c` SHA `fc1d29ecc1067e0191a4cf597189b4973546864f`).
- Destination: `research/v17c/v17c_14/` only.
- Source N=6 artifacts are pinned to `research/v17c-crystals` commit `5438b1cdd79c47abab45c6cbaf99c6daec9fbb60`.
- `master` and source branches are not modified.
- No V17C.14 trajectory process is active or was launched.
- Scientific-results package commit: 6bdeacd12f12ca33a43506dd09ba3fe7e75ac129; published as a fast-forward on research/v17c-theory. The handoff and manifest are being updated in a follow-up commit on this same branch.
- No TeX engine was available, so `theory_ru.tex` and `theory_en.tex` remain uncompiled source drafts.

## Scientific status

The generalized RHS, symbolic identities, frozen-position averaging and figure data are complete for this theory stage. Leading-only local15 gives stable opposite-sign single-particle circular roots. V17C.13 provides one selected long 3+3 run and ten discrete Coulomb sweep records, plus a 5×16 formation screen and five outcome-selected long continuations; these source summaries are reused without claiming a universal probability or independent reproduction. H4 shows only a fixed-radius axial-extrema window; the full coupled 3D state remains unresolved. The V17C.12B executable Cunningham multiplier has been found to use (C_c) instead of (1/C_c); finalist drag bias and limitations are recorded in `SOURCE_MODEL_AUDIT_RU.md`.

See `REPORT_RU.md`, `THEORY_STATUS_RU.md`, `NUMERICAL_VALIDATION_RU.md` and `FIGURE_AUDIT_RU.md` for exact results and limitations.
The requested seven ((p,s)) cases have been screened for candidate52, local15, refine12 and refine12_sobol17 in `GENERALIZED_MULTIPOLE_SWEEP_RU.md`; no trajectory integrations were added. Finite-window mixed-frequency averaging and explicit RF-power/torque identities are independently checked in the validation results JSON files.

## Reproduction

From repository root:

```powershell
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' -m unittest discover -s research/v17c/v17c_14/tests -v
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/symbolic/derive_generalized_identities.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/validate_averaged_harmonic.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/compute_leading_branches.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/generalized_multipole_sweep.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/validate_mixed_frequency_average.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/validate_power_torque_identity.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/audit_cunningham_convention.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/reduced_collective_locking.py
& '.\research\v17c\v17c_14\.venv\Scripts\python.exe' research/v17c/v17c_14/numerical/make_figures.py
```

All numerical integration counts for V17C.14 are zero. The plotted quadratures and parameter meshes are deterministic analysis, not simulated particle trajectories. PDF/SVG/PNG assets are in `figures/`; raw tables are in `results/figure_data/`.

Deterministic analysis counts: 28 candidate-by-multipole configurations, 56 species-root analyses, 301 beta-grid nodes per species, and seven fixed-radius phase-lock coupling cases. V17C.14 integrated 0 new particle trajectories; the 10-point Coulomb continuation and 80-start formation cohort are prior V17C.13 source data.

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

The V17C.14 scientific package was pushed to research/v17c-theory as commit 6bdeacd12f12ca33a43506dd09ba3fe7e75ac129, a fast-forward from cce93c2e7c8a7ca989d4921997653b399567f21c. After that push, local HEAD and the remote branch tip matched. This handoff/manifest update is published as a follow-up on the same research branch; the final ref was checked again after the follow-up and matched local HEAD. master was not modified.

## Added reduced phase-lock result

The exact nine-pair fixed-radius 3+3 phase model predicts a reduced lock fold at \(C/C_0=3.078733591\), inside the one-way V17C.13 source sign-change bracket \((2,5]\), with common \(\bar\omega=-7.84368305\times10^{-4}\). At the fold the relative phase eigenvalue is neutral. This is a calibrated reduced-model prediction, not a full N-body bifurcation, measured basin boundary, or evidence of hysteresis. The script writes results/reduced_collective_locking.json and must be run before regenerating FIG04.

No new trajectory integration, reverse continuation, or physical-field computation was started for this result.
