# FEM comparison and reusable FEM evidence — V12 audit

## Scope and provenance

This note audits the FEM inputs and FEM-backed results already present in the repository for V12. It does not launch integrations or modify V3–V11. Paths below are relative to the repository root.

The most useful authoritative reports are:

- `outputs/octupole_comparison_home/physical_verification_v10/FEM_VALIDATION.md`
- `outputs/octupole_comparison_home/physical_verification_v10/NUMERICAL_VALIDATION.md`
- `outputs/octupole_comparison_home/physical_verification_v10/config_v10.json`
- `outputs/octupole_comparison_home/confinement_maps_v11/REPORT_RU.md`
- `outputs/octupole_comparison_home/confinement_maps_v11/NUMERICAL_CHECKS_RU.md`
- `outputs/octupole_comparison_home/confinement_maps_v11/PHASE45_RU.md`

## Field availability by drive

| Drive | Existing independent field evidence | What is safe to use in V12 | Main limitation |
|---|---|---|---|
| A: rotating ABCABC | Native P4 sector solutions and h35/h50 mesh inputs used by V10; saved FEM coefficients and direct-P4 trajectories. Relevant saved P4 sector data are under `outputs/abc_validation/direct_fem_validation/v2/h35_P4/` and `h50_P4/`. | Native P4 gradients for selected validation trajectories; P4-derived harmonic parameters (including leading signed (m=4) and higher (m=8,16,20,…)); N48 reconstruction for exploratory ensemble runs when its provenance is retained. | The prior native-P4 sample is small and mostly fixed at (E_0=2) eV, (r_0=1) mm, tangent launches, phase (0°), horizon 1000 RF. It is not a four-dimensional acceptance estimate. |
| B: standing ABC | Same independent A/B electrostatic basis and mesh family as A; V10 directly integrates rotating and standing phase superpositions of the saved P4 spatial basis. | Same spatial FEM field/harmonics as A with the correct standing temporal law and voltage normalization. Reuse the V10 matched-start controls only for exact matching states and drive. | The V10 outcomes are a sparse controlled sample. The standing field contains angular pseudopotential modulation from (m=4)–(m=8) interference; it cannot be represented by only the leading (m=4) mode. |
| C: anti-phase (+ - + -…) | New, independent anti-periodic P4 FEM solve in V11. Saved h35/h50 data are in `outputs/octupole_comparison_home/confinement_maps_v11/fields/h35_P4_C/` and `h50_P4_C/`, including `sector_solution.npz`, `validation.json`, and `reconstruction_N48_C.json`. | Use the P4 sector solutions directly for validation; use the N48 reconstructions for mass trajectories only with the corresponding field JSON and FEM provenance. The field is not an analytic ideal (m=6) substitute. | Only V11's zero-speed XY finite-time maps and a small h35 edge audit exist. There is no V10-style broad direct-P4 trajectory set for C and no V12 acceptance scan yet. |

The V10 mesh/solution hashes for the ABC field are:

- h35 mesh: `4919f8c75b2a60f516d9bc54af7e29cb7261cdf885c2f8557dbfecb6bc975abf`
- h35 P4 solution: `da90df98f9db31b5b7d84397bd9605bc6a4b940d8b30a8f9acb7d19cd8815a3d`
- h50 mesh: `8ea670e6bf5bada2c86756bc98718063fe38cbfd1692c872c4a4e77f59b1a284`
- h50 P4 solution: `336882933c476bc7ecd2fd97bdc060991ddda5b6d685a3c0889e5ee36b493872`

V10 reports 647,484 h35 and 319,956 h50 elements. Its FEM solver count was zero: V10 reused these saved solutions and built no new mesh. The direct P4 implementation evaluates the quartic element potential and its analytic in-element gradient; trajectories are not classified using extrapolation outside the mesh.

For C, V11 solved the alternating-electrode problem independently with electrode 0 at (+1) V, outer boundary at 0 V, and the anti-periodic sector boundary condition (Phi(R_30° x)=-Phi(x)). The saved h50 solution hash is `0bc795eb2e2b93c2c299e3df5c9f11650ca150c067017cf3c8cdb48019309d57`; h35 is `abbd20a32ff2923d053f892c0c49a68196021f8f4fce1a4ec0670cdda3535254`. The corresponding mesh hashes are recorded in each `validation.json`. The N48 C reconstruction holdout relative RMS errors are approximately (1.155 × 10^{-8}) for potential and (1.038 × 10^{-7}) for field on both meshes.

## Harmonic information for the reduced V12 model

### A/B

V10 extracted harmonics from the native P4 potential on the (r=8) mm circle with 4096 angular samples per ring. The file `physical_verification_v10/data/p4_harmonics/native_p4_harmonic_audit.json` stores hashes and dominant signed orders; `native_p4_harmonic_coefficients.csv` stores the radius-resolved projections; `data/phase_structure_control/h50_p4_mode_coefficients.json` is a compact h50 (m=4,8) extraction.

At h50, the dominant signed orders are (l=-4) and (l=+8), with potential-series coefficient magnitudes (1.3216298094 × 10^8) and (1.0630451538 × 10^{16}) in the file's (V/(V·m^m)) convention. h35 gives the same pattern and close leading coefficient. The CSV also includes orders (16,20,28,32,40,44). Do not directly fit the high-order coefficients at small radius: the power-series coefficients become extremely ill-conditioned there. For reduced-model fitting, prefer the aperture-circle projections and/or evaluate the stored P4 field directly, then report the actual dimensionless amplitude ratios at a declared reference radius.

The V10 native-P4 pseudopotential audit finds:

- Rotating (m=4) self-term agrees with the standing (m=4) self-term to relative RMS (1.76 × 10^{-16}) at h50.
- The (m=4)–(m=8) cross term is numerically zero for the rotating quadratures (maximum absolute value (5.56 × 10^{-16}) eV in the stored audit), whereas the standing cross term spans about (-7.37) to (+7.37) eV.
- At the aperture, the direct-P4 pseudopotential ranges are 7.53–8.47 eV (rotating) and 0.795–16.47 eV (standing) for the V10 matched-leading-(m=4) protocol.

These values support retaining at least the (m=8) correction in the A/B reduced model. They are tied to the V10 voltage normalization and the aperture evaluation; they are not a general acceptance result.

### C

The V11 (r=8) mm projections for 1 V alternating-electrode excitation show:

| Mesh | (|m=6|) | (|m=18|) | (|m=30|) | (|m=42|) |
|---|---:|---:|---:|---:|
| h35 P4 | 0.676021107 | 0.009794009 | 0.000170597 | 0.000048348 |
| h50 P4 | 0.675984098 | 0.009794827 | 0.000170419 | 0.000048331 |

These are mode amplitudes in V/V on the stated radius, not universal constants independent of radius. The h35/h50 relative difference in leading (m=6) is (5.47 × 10^{-5}). The leading mode is (m=6), but finite-electrode deviations include (m=18,30,42,…); retain those terms or quantify a truncation error for any reduced model.

V11 additionally reports the scaled FEM residual (8.16 × 10^{-16}) for h35 and the sector seam matching audit. This validates the anti-periodic FEM setup; it does not prove the full dynamical acceptance converges.

## Voltage and mass/frequency protocols available to reproduce

V10's physical constants and drive are explicit in `physical_verification_v10/config_v10.json`:

- (q=+e), (M=1.6605390666 × 10^{-22}) kg (100000 Th).
- (f=100) kHz; (Omega=628318.5307179586) rad/s.
- Aperture (R_a=8) mm; outer geometry radius 25 mm.
- A rotating: 250 V; B standing: 353.5533905932738 V, matching the leading (m=4) pseudopotential only.
- (1000) RF-period maximum follow-up; survivors are right-censored at this finite horizon.

V11's equal-peak XY maps use 250 V for A/B/C. Its (v_0=0), (21 × 21) launch grid contains 305 points within (r<7.5) mm, with phase (0°) or (45°), and horizon 1000 RF. These are N48 trajectories reconstructed from h50 P4 fields (C from the independent anti-periodic field). V11 gives C's equal-pseudopotential calibration as radius-dependent: on h50, 3020.08 V at 2 mm, 755.66 V at 4 mm, and 342.78 V at 6 mm when matching the ⟨|E_c|² + |E_s|²⟩ to A. This is evidence that a cross-order comparison must specify its calibration radius; it is not a unique voltage prescription.

## Previously computed initial conditions: FEM-aware reuse matrix

| Existing set | Exact archived material | FEM model and setup | Reuse in V12 |
|---|---|---|---|
| V3 small maps / long individual trajectories | Broad legacy artifacts under `outputs/octupole_comparison_home/rf_confinement_v1/`; V11's inventory describes small RF launch samples and some individual runs through 10000 RF | Must read each row's field/drive, launch-state, normalization, and horizon metadata before reuse. | Reuse only exact state + model + phase + normalization + field provenance and sufficient horizon. Do not use a V3 label alone to infer compatibility. |
| V4 522 paired starts | `confinement_maps_v4/trajectory_results_v4.csv` | A/B at 250/250 V, 100000 Th, initial radii 1 or 2 mm, energy mainly 0.5–3.5 eV, four position azimuths, four RF phases, both tangent signs, 1000 RF. | Potential validation slices if the initial state and phase coincide exactly; not interchangeable with zero-speed XY or arbitrary 4D sample points. |
| V5 192 matched starts | `matched_pseudopotential_v5/processed/matched_strength_balanced_pairs.csv` | A/B at 250/353.553390593 V, radii 1 or 2 mm, energies 1/2/3 eV, four position azimuths/phases and tangent signs, 1000 RF. | Reuse only for exact state/drive comparisons; use as calibration/protocol controls, not as an unbiased 4D volume sample. |
| V6 boundary scan | `chaos_escape_v6/processed/trajectory_results_v6.csv` | (r_0=1) mm, (E_0=2) eV, 65 position azimuths from 0 to 22.5 degrees, phase 0, both tangent signs, 1000 RF. | Strong validation candidates for the V12 FEM/reduced models. Exact state match required. Includes previously identified angular structure, but not proof of fractality. |
| V7 harmonic mechanism / ablations | `harmonic_ablation_v7/processed/ablation_trajectory_results.csv` plus V7 report/data | Controlled (m=4), (m=8), and combined-model comparisons around the angle-sensitive starts. | Useful for validating harmonic-term implementation and mechanism; not a trajectory set for the full native FEM field or an acceptance estimator. |
| V8 phase/initial-state sensitivity | `chaos_pilot_v8/` | Local scans near the V6 angular interval, perturbations of start and RF phase; some controls extended to 3000 RF. | Reuse only records with matching drive, state, phase, field representation and duration. Earlier mesh-sensitive apparent islands must remain flagged as unresolved. |
| V9 friction controls | `dissipative_dynamics_v9/` | Selected starts with phenomenological drag; includes a few (gamma=0) controls. | Only the (gamma=0) controls can match the V12 collisionless model, subject to exact parameter/state match. Frictional trajectories cannot be counted as V12 outcomes. |
| V10 native-P4 controls | `physical_verification_v10/data/native_p4/index.csv`, paired JSON/NPZ in `data/native_p4/`; matched outcome table `processed/native_P4_vs_N48_matched_outcomes.csv` | Direct h35/h50 P4 DOP853 trajectories, mostly A/B, tangent velocity, 100000 Th, 100 kHz, phase 0, 1000 RF. | Best existing independent FEM validation set. Includes center/deep (1 mm, 2 eV), early escape and boundary brackets near 9.61°/10.27°, and selected robustness starts. Not a probability sample. |
| V10 phase controls | `physical_verification_v10/data/phase_structure_control/` | Same selected trajectories replayed with phase adjustments; direct P4 h50 files include phases (0,90,180°); phase probes include 45 and 315 degrees. | Useful for exact checks of initial-phase conventions and symmetry; retain the actual state and RF phase from each JSON. |
| V11 zero-speed maps | `confinement_maps_v11/xy/4ceec3a67c8fdfcd/` phase 0; `xy/7c854b0d1f8f059f/` phase 45. Each contains `initial_states.csv`, `results.csv`, `config.json`, checkpoints and figures. | 305 matched positions per phase, (v_0=0); A/B/C, all 250 V; h50 P4-derived N48 fields, 1000 RF. | Reusable as exact comparisons for the same field representation and coordinates. Phase-0 h35 edge repeats are in `numerical_checks/h35_edge_equal_phase0/results.csv`; only 48 (16 positions × 3 drives) were checked. Phase-45 outcomes have not had the separate h35/P4 audit. |
| V11 phase-90° zero-speed map | `confinement_maps_v11/v11_completion/xy/phase90_21/` | 305 matched positions per phase, (v_0=0); A/B/C, all 250 V; h50 P4-derived N48 fields, 1000 RF. | Reusable as a new exact phase-90° comparison; it is still a coarse, finite-time map. |

The remote V11 completion commit adds the phase-90° zero-speed map, 21×21/41×41/61×61 maps at phases 0°/45°/90°, and edge-transition tables. The phase-90° 21×21 map has 871 retained and 44 escaped starts out of 915 pooled model-start cases. The 61×61 cell-area estimates (A/B/C) are 168.062/95.562/160.812 mm² at phase 0°, 167.062/105.562/169.562 mm² at 45°, and 166.062/160.312/175.312 mm² at 90°. Thus the zero-speed equal-peak ranking changes with RF phase; these are coarse cell-area estimates, not 4D acceptance or a phase average. The completion report counts 332 measured opposite-outcome edge intervals at 0.25 mm resolution; those intervals do not rule out additional crossings between measured points. Nonzero-speed maps, long continuations, matched-normalization maps, and broad independent P4 checks remain unfinished in the remote V11 report.

### High-value exact V10 starts for independent model validation

All coordinates and velocities are SI in `physical_verification_v10/config_v10.json` and trajectory JSONs:

- Deep tangent start: (r_0=1) mm, (E_0=2) eV, azimuth 0°, positive tangent, phase 0°; ((x,y,v_x,v_y)=(0.001,0,0,62.12417634275831)) in SI.
- Early case: (r_0=1) mm, (E_0=2) eV, azimuth 22.5°, positive tangent, phase 0°; ((0.0009238795325,0.0003826834324,-23.7738930357,57.3952549972)).
- Boundary cases at 9.60205078125°, 9.6240234375°, 10.26123046875°, 10.283203125°, with both tangent signs represented on their respective sides of the brackets.
- V8 phase-sensitive case: azimuth (10.272215234375°), positive tangent, phase (45°), (E_0=2) eV, (r_0=1) mm.

The V10 matched N48/P4 comparison has 20 matched records, but the report documents a standing h35 classification disagreement for a near-boundary case. Treat that point's N48 class as spatially unresolved; direct P4 and h35/h50 details are in the tables above.

## What existing FEM results do and do not establish

Established and reusable:

1. The V10 ABC P4 fields reproduce the saved native in-element gradients to (about 4.1 × 10^{-11}) relative error in finite-difference checks; temporal superposition checks are also (below 6.4 × 10^{-11}).
2. V10's strict-tolerance h50 early standing escape is stable under integration refinement: (185.49320025428142) RF versus base (185.4932002543185) RF. This is local time-integration convergence, not a general boundary convergence result.
3. The same early standing escape survives the h35/h50 spatial check and agrees closely with N48.
4. The h35/h50 direct-P4 outcomes of the late V6 boundary samples agree in class, but V10 reports exit-time shifts as large as about 165 RF periods. One h35 N48 classification disagrees with direct P4 for the near-boundary positive case. Fine boundary topology and escape-time rankings therefore require fresh P4 checks.
5. V11's C field is independently solved, has an (m=6)-leading spectrum, and has close h35/h50 (m=6) coefficients. One C start ((3.75,-6)) mm, (v=0), phase 0 exits at 25.5692927 RF (h50 P4), 25.5693877 RF (h35 P4), and 25.5693120 RF (h50 N48). This is one direct-P4 check, not a C convergence study over the XY domain.

Not established:

- No previous output is a full, statistically controlled (a_4(Lambda,…)) or (A_4(N)) scan.
- V11 retained-to-1000-RF labels are not permanent-stability evidence.
- A/B results at 250/250 V cannot be combined with V10/V5 results at 250/353.553 V as if they represented one normalization.
- The V11 C calibration at 2, 4, and 6 mm is explicitly different at each reference radius; no unique order-independent matched voltage exists.
- A small N48-vs-P4 field error does not bound the long-time trajectory-classification error near an escape boundary.
- The FEM modes at aperture cannot alone validate the ideal model across the whole aperture or replace the field away from the fitting circle. Compare full trajectories against native P4 for representative matched starts.

For V12, the safest division is: use the analytical (m)-pole as the geometry-independent baseline; derive dimensionless FEM correction amplitudes and phases at explicit radii from the saved P4 solutions; validate reduced-field trajectories against the native P4 controls; and keep the V11 XY/phase maps as exact finite-time FEM-informed validation slices rather than acceptance statistics.

## V12 field-strength normalization audit

For equal 250 V electrode peaks, the h50 angular RMS RF field per applied volt is strongly radius dependent. At `r=4 mm`, A is `47.8887 V m⁻¹ V⁻¹` while C is `15.8434 V m⁻¹ V⁻¹` (A/C=`3.02`). At `r=7.5 mm`, A is `353.962` and C is `367.233 V m⁻¹ V⁻¹` (C/A=`1.037`). Thus the equal-peak scan is a valid voltage-protocol comparison, but it is not a global equal-effective-field comparison and cannot establish a radius-independent strength ranking.

The V12 `matched` protocol states its rule and reference radius explicitly. B uses `250√2 = 353.553390593 V` to match the leading m=4 pseudopotential of rotating A. Real C matches the full angular RMS of both RF field quadratures of real A at `r_ref`; the ideal m=6 control instead matches the leading-mode RMS field of ideal A at the same radius. h50 voltages for `A=250 V` are:

| `r_ref` | B and ideal B | Real FEM C, full field match | Ideal m=6 C, leading-mode match |
|---:|---:|---:|---:|
| 2 mm | 353.553 V | 3020.080 V | 3020.070 V |
| 4 mm | 353.553 V | 755.658 V | 755.018 V |
| 6 mm | 353.553 V | 342.778 V | 335.563 V |

The several-fold shift with `r_ref` follows the different radial growth of m=4 and m=6 fields, with smaller corrections from the FEM higher harmonics. Matched results must be reported at 2, 4, and 6 mm as separate protocols; one radius cannot establish that one order is universally superior. Full-field and leading-mode voltages are recorded separately in each run's `model_coefficients.json`.

## New V12 paired h35/h50 acceptance-method pilot

A new, deduplicated Sobol subset was run on the N48 h35 and h50 fields using the same eight physical starts for each model and mesh: equal peak 250 V, RF phase 0°, horizon 1000 RF periods, DOP853 `rtol=2e-10`, `max_step=T_RF/64`, four worker processes. All 48 trajectories completed. The retention classifications agreed for all 24 model/start mesh pairs: A retained 7/8, B 2/8, and C 8/8 on both meshes. This small subset is only a mesh-sensitivity pilot; it is not an acceptance estimate or evidence of indefinite retention.

Among escaping starts, A had one exit-time comparison with `|Δt|=8.46e-5 RF`; B had six, with maximum `|Δt|=0.01709 RF` and mean signed `t_h50−t_h35=+0.00290 RF`. C had no exits in this subset, so it provides no exit-time comparison. The different status groups and short finite sample prevent conclusions about long-time convergence.

The start set uses seed 121114 and was checked against the 32-start h50 cost pilot and the prior 30-trajectory short SI check; there were no exact duplicate physical starts. Full job data and the comparison are in `runs/c0b99b2d4dd6413aeddd_multimesh_equal_phase0_8sobol/`, especially `mesh_classification_comparison.csv`, `mesh_convergence_summary.json`, and `summarize_mesh_convergence.py`.
