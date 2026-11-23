# Численная верификация и воспроизводимость

## Проверенные исходные результаты

Источники фиксированы в [`data/source_benchmarks.json`](data/source_benchmarks.json): V17C.17 parent/source `718d9eadf5873cdf1a5b602de31ec20647092f0f`; V17C.16 published commit `6708ec62acec19f11ade7b8bdc3d24c0b6985a33`; V17C.15 theory validation `9716e9a701b4ebd8fc3ae4b72a8aaeda9437289b`. V17C.16 has an independent DOP853 and RK4 step-refinement panel in its immutable source directory. V17C.17 has 15k/50k continuation, DOP853 3k checks, Coulomb=0/nominal/5x, perturbation and torque data; its exact limitations are quoted without relabeling it as finite 3D.

## Новый mathematical screen — завершён

`code/model.py` contains the transparent leading-harmonic equations and pair charge/mass scaling. `code/run_screening_maps.py` completed four 13×13 one-particle sections and a 9×9 two-particle section with four fixed starts per cell: 676 single-particle trajectories + 324 paired trajectories = **1000 trajectories** across **757 grid configurations**. The pair panel changes charge at fixed PS radii/masses/densities; (Q_i=\alpha_iM_i), and Coulomb is recomputed from (Q_1Q_2). It omits z, gravity and polarization; this is a planar mathematical screening map, not an axial-device result.

The baseline is RK4 with 64 steps per RF1 period. In the ν sweep, the step count increases with ν to retain at least the baseline resolution per RF2 period; the actual count is recorded per CSV row. The horizon is 1000 RF1 periods, four deterministic starts per pair-map cell, and common RF phases (0,0). The pair RHS includes the small quasi-steady Oseen correction encoded by the V16 PS diameter-based Reynolds factors; single-particle screens use linear Stokes drag. A `*_SCREEN` sign requires bounded radial motion and at least 0.10 net turn in the final 20%. Bounded trajectories below that threshold are classified as low-turn/nonrotating for this finite window; a slow nonzero branch cannot be excluded. This is a finite-time direction test, not an asymptotic-stability or attractor criterion. Four starts are not a universal capture-probability sample; Wilson intervals describe only that finite panel.

### Итоги карты

| Сечение | POS | NEG | Низкий поворот | Неразрешено | Контакт с апертурой | Всего |
|---|---:|---:|---:|---:|---:|---:|
| ((\alpha,\lambda)) | 95 | 56 | 18 | 0 | 0 | 169 |
| ((\alpha,\nu)) | 80 | 46 | 14 | 2 | 27 | 169 |
| ((\alpha,b/a)) | 86 | 24 | 19 | 40 | 0 | 169 |
| ((\alpha,\delta/\delta_{PS,2})) | 103 | 35 | 12 | 19 | 0 | 169 |

В парном наборе из 324 стартов получены 40 (CR(+−)), 35 (CR(−+)), 108 (CO(++)), 66 (CO(−−)), 35 bound/low-turn, 22 unresolved и 18 aperture-contact исходов; столкновений и случаев потери обеих частиц не было. Доля встречного вращения этого конкретного четырёхстартового пула (75/324=23.15\%) не является вероятностью захвата. По каждой ячейке сетки доли и интервалы Уилсона приведены в CSV; малая выборка делает интервалы широкими.

## Отдельные численные sanity checks

Before map generation, central finite differences at a generic ((x,y,t)) compared the analytical gradient and Hessian to the potential: maximum absolute errors were (1.2\times10^{-12}) and (1.3\times10^{-11}), respectively; the 2D DC slice has (tr(H)=-2\beta), while the full 3D quadrupole adds (H_{zz}=+2\beta), giving zero Laplacian. The optimized scalar RHS and the NumPy vector RHS agree at the checked point. The V16 reference MFS verification independently reported boundary residual (\sim10^{-15}) and leading-transfer agreement (<4\times10^{-7}) relative at order 24; that remains 2D, infinite-rod MFS.

For the frozen PS pair (\(\alpha_1=1,\alpha_2=0.3412603\), Coulomb on, P0), RK4 refinement over 1000 periods gave last-window mean radii \((0.5756323,0.7640904)R\) at 64 steps/period and \((0.5756550,0.7641183)R\) at 256 steps/period; changes were 0.0040% and 0.0037%. Mean angular advances per RF1 period changed from \((+0.001016246,-0.001053686)\) to \((+0.001016406,-0.001053423)\), relative changes 0.0157% and 0.0250%. Independent DOP853 differed from RK4/256 by 0.024% and 0.0079% in the two mean radii; the angular advances differed by less than 0.00002%. The single-particle Stokes-only refinement gave 0.0054% mean-radius and 0.0021% rotation-increment changes from 64 to 256 steps/period. These comparisons demonstrate integrator consistency on this selected reduced-model trajectory, not convergence to a finite-electrode field or asymptotic invariant torus.

`results/screening_summary.json`, raw CSV files and figure files record the completed historical planar trajectory campaign. A frozen independent convergence check is in `results/selected_numerical_validation.json`. The earlier finite RF-only 3D FEM is preserved in `results/finite_3d_rf_fem.json`; it is a separate, superseded geometry check.

## Updated continuous-rod / endcap FEM check

`code/endcap_3d_fem.py` solves four complex RF basis fields (real/imaginary for (k=+2,-4)) and two DC endcap basis fields in a finite 3D geometry. The domain contains 24 continuous RF rods, a grounded 50 mm cylindrical shield, grounded external end plates, and isolated DC disks at (z=\pm23) mm. The boundary classifier reported zero unassigned boundary nodes at every solved level. The h=0.24 mm solve used AMG-preconditioned conjugate gradients; all six relative residuals were below (9.7\times10^{-10}).

Mesh and field observables are preserved in `results/continuous_rods_endcap_fem.json`; the design note contains a table and limitations. Five solved levels are h=0.60, 0.45, 0.34, 0.27, 0.24 mm. The two last refinements, 0.34→0.27 and 0.27→0.24 mm, change the two RF transfer magnitudes by (0.885%,0.930%) and (0.327%,0.604%); the larger force-balance root moves by 0.0059 and 0.0400 mm, respectively. Thus the stated two-consecutive-refinement gate **passes for these two scalar observables**. The earlier 0.45→0.34 mm interval remains non-monotone and has larger changes. This is not convergence certification for full E, derivatives or sharp-edge surface maxima. A 0.22 mm mesh with 509,343 nodes and 3,003,109 tetrahedra was generated, but the solve was stopped before it yielded any fields; this is recorded as an incomplete attempt and is not a converged point.

The 0.24 mm screening DC basis requires about 567.3 kV on each end disk to balance gravity for particle 1 at z=−2.5 mm. The force-balance screen puts particle 2 at z≈−4.563 mm, a formal Δz≈2.063 mm. That voltage, combined with a roughly 0.75 mm nearest DC-to-rod gap, rejects this specific endcap placement by a crude V/d comparison. No converged surface field, corona or breakdown margin is claimed. Although the scalar mesh criterion passes, the physical voltage screen fails, so **no final particle trajectories were integrated in the new finite 3D field**.

## Минимальные тесты

`tests/test_model.py` checks derivatives, Laplace trace of the 3D DC quadrupole, scalar/vector RHS agreement and charge/mass scaling. Four tests pass. These checks validate code identities, not physical correctness of the selected geometry.

### Reproduction

From the study folder run the historical map and validation scripts as needed, then `python code/make_aux_figures.py`, `python code/publish_figures.py`, `python code/endcap_3d_fem.py`, and `python code/audit_endcap_fem.py`. The map script used 10 workers in its original run; the FEM solve runs serially. Figures are saved as PNG 600 dpi, PDF and SVG. Existing planar trajectories do not validate the finite endcap field. The scalar FEM convergence gate passes, but surface-field convergence and electrical feasibility do not.
