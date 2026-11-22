# V17C.16 — handoff

## Repository state

- Research branch: `research/v17c-two-particle-oseen`.
- Parent branch: `research/v17c-theory-validation` at
  `9716e9a701b4ebd8fc3ae4b72a8aaeda9437289b`.
- All new study material is confined to `research/v17c/v17c_16/`.
- `master` and all older study directories are unchanged by this branch.
- Publication is a normal fast-forward push. Record the final commit SHA and
  verify the remote branch points to that exact SHA before closing the study.

## Scientific status

This work tests one dimensionally consistent 35/45 μm polystyrene pair in air,
not a parameter optimization. The point-charge model counterrotates over the
tested finite horizon in the leading-harmonic field and the full 2D infinite-rod
MFS RF field; the signs persist when Coulomb coupling is set to zero. This does
not prove a global attractor or experimental feasibility.

The real-sphere polarization sensitivity is material: a full-MFS DOP853 run of
1000 primary RF periods with an illustrative real, frequency-independent
`εr=2.6` retains both particles geometrically but shifts particle 2's mean
rotation to approximately zero and moves its mean radius from about `0.766R` to
`0.939R`. This parameter is not a measured complex dielectric response. Thus
the nominal point-charge counterrotation is not yet a qualified polystyrene
result. Run outcomes, limitations, and mechanism are detailed in
`REPORT_RU.md` and `PHYSICAL_SCALING_RU.md`.

Quasi-steady Oseen drag is not a validated 1–2% physical closure for this pair:
exact full-velocity trajectories have finite-Reynolds and unsteady-flow
parameters outside a comfortably asymptotic regime. The frequency error of the
averaged branch is 1.77%/5.16% against full MFS; radius error is 0.44%/0.084%.
The 2D MFS field excludes finite rod ends and a realizable DC electrode system.
Breakdown, charge retention, measured complex permittivity, wall-corrected
hydrodynamics, and independent FEM remain open gates.

## Reproduction commands

Run from the repository root on this branch:

```powershell
python research/v17c/v17c_16/numerical/physical_screen.py
python research/v17c/v17c_16/numerical/averaged_theory.py
python research/v17c/v17c_16/numerical/run_mfs_audit.py
python research/v17c/v17c_16/numerical/averaged_coulomb_screen.py
python research/v17c/v17c_16/numerical/run_trajectory_validation.py --periods 1000 --steps-per-fast-period 16 --workers 4 --tag 1000p_s16
python research/v17c/v17c_16/numerical/run_trajectory_validation.py --periods 1000 --steps-per-fast-period 32 --workers 1 --field ideal --tag ideal1000_s32 --selected-cases oseen_coulomb_nominal_start01
python research/v17c/v17c_16/numerical/run_trajectory_validation.py --periods 1000 --steps-per-fast-period 16 --workers 2 --field mfs --mfs-order-per-rod 24 --tag mfs1000_s16 --selected-cases oseen_coulomb_nominal_start01 oseen_kappa0_independent_control
python research/v17c/v17c_16/numerical/run_trajectory_validation.py --periods 250 --steps-per-fast-period 16 --workers 2 --field mfs --mfs-order-per-rod 24 --tag mfs250_dep_eps26 --induced-dipole-epsilon-r 2.6 --selected-cases oseen_coulomb_nominal_start01 oseen_kappa0_independent_control
python research/v17c/v17c_16/numerical/run_trajectory_validation.py --periods 250 --steps-per-fast-period 32 --workers 1 --field mfs --mfs-order-per-rod 24 --tag mfs250_dep_eps26_s32 --induced-dipole-epsilon-r 2.6 --selected-cases oseen_coulomb_nominal_start01
python research/v17c/v17c_16/numerical/run_trajectory_validation.py --periods 1000 --steps-per-fast-period 16 --workers 1 --field mfs --mfs-order-per-rod 24 --tag mfs1000_dep_eps26 --induced-dipole-epsilon-r 2.6 --selected-cases oseen_coulomb_nominal_start01
python research/v17c/v17c_16/numerical/run_rk4_convergence.py
python research/v17c/v17c_16/numerical/run_lyapunov.py
python research/v17c/v17c_16/numerical/physical_force_screen.py
python research/v17c/v17c_16/numerical/audit_results.py
python research/v17c/v17c_16/numerical/audit_polarization_sensitivity.py
python research/v17c/v17c_16/numerical/make_figures.py
python -m unittest discover -s research/v17c/v17c_16/tests -v
python research/v17c/v17c_16/numerical/build_manifest.py
```

The reproduced trajectories use deterministic initial states and fixed RF
phases as encoded in `run_trajectory_validation.py`; they are not a probability
sample. The DOP853 summary JSON files contain exact solver tolerances, steps,
retention, contact, and mean-frequency classifications. RK4, QR/Lyapunov, and
MFS surface-screen settings are preserved beside their outputs.

## Main artifacts

- `REPORT_RU.md`: results, errors, physical limitations, and conclusion.
- `PHYSICAL_SCALING_RU.md`: dimensional parameter reconstruction and material
  and discharge screens.
- `theory/ANALYTICAL_THEORY_RU.md`, `theory/ANALYTICAL_THEORY_EN.md`: two-body
  nonlinear-Oseen averaging and stability theory.
- `numerical/`: independent RHS, MFS field and force, analytic branch, finite-
  time variation, audit, and figure-generation code.
- `results/`: JSON summaries and compressed representative/full trajectories.
- `figures/`: four composite figures in PDF, SVG, and 600-dpi PNG, figure data,
  and Russian scientific-informativeness audit.
- `sources/REFERENCES_RU.md`: DOI and technical-source register with transfer
  limitations.
- `tests/`: 13 unit and consistency tests.
- `MANIFEST.json`: per-file SHA-256 inventory; regenerate after any edits.

There are 25 saved DOP853 trajectory cases in the study folders, plus three
RK4 refinement integrations and their DOP853 reference, and three separate
finite-time QR/Lyapunov integrations. The primary dimensional set is one pair;
the other runs are explicit drag, Coulomb, field, gravity-orientation, or
polarization controls, not separately optimized physical configurations.

## Known caveats and continuation

1. The transverse-gravity run that crosses an electrode wall is retained as an
   early-contact result. Older transverse trajectories that continued beyond
   a wall are marked legacy and must not be interpreted physically.
2. The `εr=2.6` induced-dipole sensitivity uses a constant real polarizability;
   repeat only after measuring or sourcing suitable `ε*(ω)` and charge-loss
   data. Also assess mutual dipoles, charge-polarization coupling, and
   near-wall resistance before making a material claim.
3. The longest trajectories cover about one particle orbit, and the slow
   relative phase beat is not fully sampled. Finite-time negative QR exponents
   drift toward zero; no asymptotic attractor is claimed.
4. No mass optimization, broad parameter search, FEM, 3D electrode-end model,
   or experimental protocol qualification is included.
5. Before any continuation, inspect local and remote branch SHAs. Push only
   this research branch; do not merge to or edit `master` as part of V17C.16.
