# V17C.16 — two charged microparticles with low-Reynolds-number drag

This directory contains the new, isolated V17C.16 study. Its starting commit is
`9716e9a701b4ebd8fc3ae4b72a8aaeda9437289b` from
`research/v17c-theory-validation`; the work is carried on
`research/v17c-two-particle-oseen`. Earlier research directories and `master`
are not modified.

The first calculation is a dimensional consistency and validity screen for the
specified 35/45 micrometre polystyrene pair in atmospheric air. No parameter
optimization is launched until this screen and the analytic averaging have
been audited. The 24-rod geometry has two independent RF families with leading
spatial harmonics `k=+2` and `k=-4`; it is not the earlier 12-electrode trap.

## Reproducibility map

- `PHYSICAL_SCALING_RU.md` — dimensional definitions and first physical gates.
- `theory/ANALYTICAL_THEORY_RU.md` and `theory/ANALYTICAL_THEORY_EN.md` —
  averaged equations and their stated limits.
- `numerical/physical_screen.py` — reconstructs dimensional and dimensionless
  trial-pair values from one source parameter set.
- `numerical/two_particle_oseen.py` — exact two-body RHS for the ideal two-mode
  RF field, DC quadrupole, Coulomb force, gravity/buoyancy, and nonlinear Oseen
  drag. The Stokes-only limit is selectable with `oseen=False`.
- `numerical/mfs_two_particle_dynamics.py` — independently reconstructed 24-rod
  MFS gradients/Hessians and full-field two-particle RHS, including the optional
  induced-dipole sensitivity term.
- `numerical/audit_polarization_sensitivity.py` and
  `results/polarization_sensitivity_audit.json` — step-refinement comparison
  for the MFS dielectric-polarization control.
- `numerical/physical_force_screen.py` and
  `results/secondary_force_screen.json` — order-of-magnitude DEP, Brownian,
  and near-wall drag screens.
- `results/dimensional_screen.json` — generated screening output.
- `sources/REFERENCES_RU.md` — source register, evidence class, and limitations.
- `tests/` — executable unit and consistency checks.

## Main results

- `REPORT_RU.md` summarizes the physical screening, analytical comparison,
  MFS and trajectory results, numerical convergence, and remaining gates.
- `SESSION_HANDOFF.md` records the branch state, exact commands, run counts,
  output locations, and continuation notes.
- `results/trajectory_validation_1000p_s16/` contains the fixed three-start
  ideal-field checks, `trajectory_validation_mfs1000_s16/` the 2D full-MFS
  field run, and `trajectory_validation_ideal1000_s32/` the long-horizon
  DOP853 step-refinement run.
- `results/lyapunov_qrf*.json` and `results/rk4_convergence.json` contain the
  finite-time tangent-space and RK4 checks. The QR estimates are not converged
  asymptotic exponents.
- `figures/` contains four audited composite figures in PDF, SVG and PNG at
  600 dpi, plus plot source data and generation code.
- `MANIFEST.json` provides SHA-256 hashes for the published files.

The fixed physical trial pair counterrotates over the finite tested horizon in
both ideal-leading-harmonic and full 2D infinite-rod MFS RF fields. The same
signs persist at `kappa=0`, so Coulomb is not the cause of the rotation in this
example. This is not a discharge-safety certification, a charge-retention
measurement, or experimental validation. Oseen is only a quasi-steady
approximation here, the finite 3D electrode field is not modeled, and the
second particle's vertical gravity offset is outside the central `R` scale.

An additional induced-dipole sensitivity run using a real, frequency-independent
`epsilon_r=2.6` is illustrative only. Over 1000 primary periods it keeps both
particles geometrically clear of rods but shifts the second particle's mean
angular velocity close to zero. This demonstrates that the point-charge
counterrotation result does not yet establish a realizable polystyrene pair;
the dielectric response and surface-charge physics need measured inputs.
