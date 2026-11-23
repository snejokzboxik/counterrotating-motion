# SESSION HANDOFF — V17C.18B–19

## Latest continuation: continuous rods / DC endcaps (2026-11-23)

Architecture is now 24 continuous cylindrical RF rods. Both RF phase patterns (k=+2,-4) remain unchanged along each full active rod; no RF axial segmentation is used. Separate DC end electrodes at the trap ends are the main axial-confinement concept. Axially segmented DC/two-well electrodes remain a fallback. Do not continue from the earlier segmented-RF proposal.

New code: `code/endcap_3d_fem.py`. New P1 FEM checkpoint: `results/continuous_rods_endcap_fem.json`. It models 24 finite continuous rods, two separate DC disks at (z=\pm23) mm, a grounded 50 mm cylindrical shield, and grounded outer end plates. The solved levels are (h_{max}=0.60,0.45,0.34,0.27,0.24) mm (352,144 nodes and 1,992,168 tetrahedra on the finest solved mesh). At (r=0.9R,z=0), the finest transfers are 0.72136 ((k=+2)) and 0.54018 ((k=-4)). The last two refinement pairs, 0.34→0.27 and 0.27→0.24 mm, change RF magnitudes by (0.885%,0.930%) and (0.327%,0.604%); the second force-balance root moves by 0.0059 and 0.0400 mm. The scalar two-consecutive-refinement gate passes. This does not establish convergence of the full field or sharp-edge surface maxima.

A 0.22 mm mesh was generated (509,343 vertices, 3,003,109 tetrahedra) but its FEM solve did not finish and was stopped without field results. It is archived in `results/continuous_rods_endcaps_h0.22mm_incomplete_mesh.zip` with SHA-256 values in the result JSON; do not treat it as a solved level. The cap voltage needed to balance particle 1 at (z=-2.5) mm is approximately 567.3 kV at the finest solved mesh; the screening force balance puts particle 2 near (z=-4.563) mm, giving a formal separation of 2.063 mm. The shortest cap-to-RF-rod gap is approximately 0.75 mm, so the crude (V/d) screen is about (7.56\times10^8) V/m. This rejects the tested endcap placement in air but is not a converged peak-field or breakdown calculation. No final particle trajectories were run because the physical voltage screen fails.

The FEM checkpoint records values and limitations; `AXIAL_ELECTRODE_DESIGN_RU.md` has the detailed table. The design note, feasibility report, numerical validation, figure audit and updated FIG06 are saved. Keep earlier V18 FEM and trajectory data as historical results; do not overwrite them.

## Branch, source pins and scope

- Branch: `research/v17c18b19-axial-stability-diagrams`; source parent V17C.17 SHA `718d9eadf5873cdf1a5b602de31ec20647092f0f`.
- Immutable audit sources: V17C.16 `6708ec62acec19f11ade7b8bdc3d24c0b6985a33`, V17C.15 `9716e9a701b4ebd8fc3ae4b72a8aaeda9437289b`, V17C.12B reference `fc1d29ecc1067e0191a4cf597189b4973546864f`.
- GitHub API audit before publication confirmed `master` at `e5e6b2473bf519c5670ee1c48882183fbcfc8f2d`; the requested new research ref was absent then.
- No master checkout or modification. All new study files are under `research/v17c/v17c_18b19/`.
- This package separates inherited evidence, new reduced-model maps, and finite-geometry field work. Do not interpret the initial screening as a prototype design.

## Completed in this session

1. Audited PS V17C.16 and the separate `phys0570` V17C.17 benchmark. Their size/frequency scales remain separate. Dimensional values and source SHA pins are in `data/source_benchmarks.json`.
2. Added dimensionless equations and code for leading RF harmonics (k=+2,-4), ideal DC quadrupole, linked particle masses/charges/Coulomb, low-Re Oseen option, and induced-dipole sensitivity coefficient. `tests/test_model.py`: **4 passed**.
3. Completed four single-particle 13×13 mathematical sections and a pair 9×9 grid with four deterministic starts/cell: **757 grid configurations, 1000 trajectories**, 10 workers. Results: `data/*.csv`, `results/screening_summary.json`. Pair outcomes: 40 CR(+−), 35 CR(−+), 108 CO(++), 66 CO(−−), 35 bound/low-turn, 22 unresolved, 18 aperture-contact; zero collisions and zero both-lost cases. The sampled CR fraction 75/324=23.15% is not a universal basin probability.
4. Added frozen 1000-RF1-period RK4 64/128/256 and independent DOP853 validation of the P0 PS pair. Last-window pair radius changes for RK4 64→256 were 0.0040% and 0.0037%; angular increments changed 0.0157% and 0.0250%. DOP853 vs RK4/256 differences were <0.024% in radius and <0.00002% in angular increment for this case. Full values: `results/selected_numerical_validation.json`.
5. Built a preliminary finite-length 3D electrostatic FEM solver for two RF basis patterns on 24 finite cylindrical rods inside a grounded shield with grounded end plates. It uses a linear tetrahedral mesh, with no axial RF segmentation and no DC-segment solve. Corrected midplane probes at r=0.9R give finest-mesh transfers 0.71955 (+2), 0.53777 (−4); difference from infinite-2D reference at the same radius is 2.1%/2.7%, while the 0.42→0.32 mm mesh changes remain 4.7%/5.6%, so convergence is not demonstrated. The earlier r=R probe is tangent to rods; retain it only as an explicitly invalid diagnostic and do not cite it.
6. Added exact infinite-cylinder DC Fourier attenuation, an earlier segmented-rod concept schematic (now historical/unindexed), inherited trajectories, and physical-parameter linkage figures. Current FIG06 shows continuous RF rods with separate DC endcaps and finite-FEM observations. Figure source code and tabular data are included. Figures are PNG 600 dpi, PDF and SVG; current publication stems are indexed by `data/figure_index.json`.
7. Literature/technical sources with DOI and transfer limits are documented in `sources/REFERENCES_RU.md` and the physical-feasibility report.

## Key preliminary findings / gates

- In V17C.16 PS ideal 2D-MFS + DC quadrupole, the P0 orbit was (+−); the conditional instantaneous-real PS polarization case shifted particle 2 almost to zero mean angular speed. New P1 map is coefficient sensitivity only, not frequency-dependent dielectric physics.
- The V16 RF electrode screening had a conservative rod-surface peak sum about 8.11 MV/m and a neighboring-gap peak difference near a simple planar Paschen comparison. Corona/AC/end/roughness safety margin was not established.
- Exact cylindrical Fourier attenuation is about 0.00182 at (d_z=2) mm, 0.159 at 5 mm, 0.553 at 10 mm, and 0.753 at 15 mm for (R=5) mm and shield-wall radius 5.263 mm. Short axial DC features are strongly smoothed toward axis in this screening geometry.
- The finite-rod FEM is a first RF-only field solve. It does not yet choose DC segment voltages, demonstrate distinct stable (z) wells, compute field Hessians/polarization, or feed trajectories through the solved field. Its transfer factors and mesh sensitivity must be read from the corrected JSON only.
- No P2 complex dielectric, charge-retention experiment, nonlinear breakdown/corona prediction, or experimental confirmation was produced.

## Remaining before claiming a physical operating window

1. Redesign or reposition the tested DC end electrodes: the estimated required voltage is about 567.3 kV and fails the simple (V/d) screen. Candidate changes include placing the separate DC end electrodes closer to the central working volume or shortening continuous RF rods; recompute every changed finite-field basis.
2. The scalar mesh-refinement criterion now passes for the last two solved intervals. The existing 0.22 mm mesh has no solved field data; do not treat it as a refinement result. Full-field, derivative and surface-maximum convergence still need separate checks for any redesigned geometry.
3. Refine rod/cap edges with realistic radii and include supports/feedthroughs before reporting local surface fields, corona, or breakdown margins.
4. After a redesigned geometry passes the voltage and FEM convergence gates, reconstruct (E_x,E_y,E_z), gradients and Hessians and verify radial/axial force balance.
5. Only then integrate the full two-particle dynamics (Coulomb, validated drag, gravity, finite particle size, and justified polarization), check initial phases/velocities, and establish whether the inherited (+−) regime survives.
6. A physical operating window and a prototype remain unestablished. Segmented DC/two-well electrodes are a fallback; axially segmented RF rods are not the primary design.

## Final package audit

- Endcap FEM bookkeeping audit: `results/endcap_geometry_audit.json`; five solved levels, boundary classification, six finest-level residuals, two consecutive scalar-refinement passes, incomplete-mesh archive checksum and the DC voltage screen recorded. Surface-field convergence and electrical breakdown are not established.
- Output audit: `results/output_audit.json`; all stored CSV row counts, pair-outcome totals, selected trajectory validation, FEM records, and indexed PNG/PDF/SVG figure files passed integrity assertions.
- Reproduction tests: `4 passed` using `$env:PYTHONPATH = (Resolve-Path code).Path; python -m pytest -q tests` from this directory.
- The SHA-256 file manifest is `MANIFEST.json` and covers every regular study artifact except itself and Python cache files.
- Publication audit: package commit `4cd85093a1d5d3f663d61fac939e72541e7320eb` was pushed with a normal fast-forward. `git ls-remote` matched that local and remote branch SHA; remote `master` remained `e5e6b2473bf519c5670ee1c48882183fbcfc8f2d`. A small follow-up commit adds this publication record to the handoff and manifest.

## Reproduction

```powershell
$env:PYTHONPATH = "research/v17c/v17c_18b19/code"
python code/run_screening_maps.py
python code/validate_selected.py
python code/endcap_3d_fem.py
python code/audit_endcap_fem.py
python code/make_aux_figures.py
python code/publish_figures.py
python code/finite_3d_rf_fem.py
python code/endcap_3d_fem.py
python code/audit_endcap_fem.py
python -m pytest -q research/v17c/v17c_18b19/tests
```

The earlier screening driver used 10 workers out of 12 logical processors, with about 15.4 GiB available RAM reported by that session. FEM runs serially and should not overlap another heavy field job. Install the pinned endcap FEM dependencies with `pip install -r code/requirements_endcap_fem.txt`. The latest AMG-CG FEM script checkpoints each completed mesh level in `results/continuous_rods_endcap_fem.json`; `audit_endcap_fem.py` checks stored residuals, refinement deltas and the incomplete-mesh archive hash.
