# SESSION HANDOFF — V17C.18B–19

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
6. Added exact infinite-cylinder DC Fourier attenuation, a finite segmented-rod design schematic, inherited trajectories, and physical-parameter linkage figures. Figure source code and tabular data are included. Figures are PNG 600 dpi, PDF and SVG; publication stems are indexed by `data/figure_index.json`.
7. Literature/technical sources with DOI and transfer limits are documented in `sources/REFERENCES_RU.md` and the physical-feasibility report.

## Key preliminary findings / gates

- In V17C.16 PS ideal 2D-MFS + DC quadrupole, the P0 orbit was (+−); the conditional instantaneous-real PS polarization case shifted particle 2 almost to zero mean angular speed. New P1 map is coefficient sensitivity only, not frequency-dependent dielectric physics.
- The V16 RF electrode screening had a conservative rod-surface peak sum about 8.11 MV/m and a neighboring-gap peak difference near a simple planar Paschen comparison. Corona/AC/end/roughness safety margin was not established.
- Exact cylindrical Fourier attenuation is about 0.00182 at (d_z=2) mm, 0.159 at 5 mm, 0.553 at 10 mm, and 0.753 at 15 mm for (R=5) mm and shield-wall radius 5.263 mm. Short axial DC features are strongly smoothed toward axis in this screening geometry.
- The finite-rod FEM is a first RF-only field solve. It does not yet choose DC segment voltages, demonstrate distinct stable (z) wells, compute field Hessians/polarization, or feed trajectories through the solved field. Its transfer factors and mesh sensitivity must be read from the corrected JSON only.
- No P2 complex dielectric, charge-retention experiment, nonlinear breakdown/corona prediction, or experimental confirmation was produced.

## Remaining before claiming a physical operating window

1. FEM mesh audit is complete: corrected 0.42→0.32 mm changes are 4.7% and 5.6%, so transfer convergence remains unresolved. A later continuation should refine gaps/ends and compare MFS/FEM at identical (r,z) locations before using these coefficients.
2. Add boundary groups and separate DC basis solves for the actual axially segmented RF rods; optimize potential coefficients only after verifying (E_x,E_y,E_z), gradients and Hessians in the working volume.
3. Construct DC patterns giving two force-balanced locations. Check Earnshaw-consistent radial components, gravity, charge-specific equilibria, electrode clearance, surface fields and end effects.
4. Reintegrate the two particles in the full FEM field, including Coulomb, Stokes/Oseen validity, gravity and justified polarization. Validate starting phases/velocities and longer windows. No current new map is evidence of axial separation or physical capture probability.
5. If proceeding to experimental feasibility, characterize the complex dielectric response and charge stability of actual particle lots and measure pressure/temperature dependent drag.
6. A physical operating window remains unestablished. The immediate continuation is to add real DC-segment basis-solves to the finite geometry, locate charge-specific axial equilibria, and then validate full-field two-particle trajectories with polarization and drag models supported by measurements or primary data.

## Final package audit

- Output audit: `results/output_audit.json`; all stored CSV row counts, pair-outcome totals, selected trajectory validation, three FEM mesh records, and indexed PNG/PDF/SVG figure files passed integrity assertions.
- Reproduction tests: `4 passed` using `$env:PYTHONPATH = (Resolve-Path code).Path; python -m pytest -q tests` from this directory.
- The SHA-256 file manifest is `MANIFEST.json` and covers every regular study artifact except itself and Python cache files.
- The locally reviewed package is committed only on `research/v17c18b19-axial-stability-diagrams`. Remote branch SHA comparison is reported in the completion record for this session.

## Reproduction

```powershell
$env:PYTHONPATH = "research/v17c/v17c_18b19/code"
python code/run_screening_maps.py
python code/validate_selected.py
python code/make_aux_figures.py
python code/publish_figures.py
python code/finite_3d_rf_fem.py
python -m pytest -q research/v17c/v17c_18b19/tests
```

The screening driver used 10 workers out of 12 logical processors, with about 15.4 GiB available RAM reported by the session. FEM runs serially and should not overlap another heavy field job.
