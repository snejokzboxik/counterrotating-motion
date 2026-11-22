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

`results/screening_summary.json`, raw CSV files and figure files record the completed trajectory campaign. A frozen independent convergence check is in `results/selected_numerical_validation.json`. The finite RF-only 3D FEM mesh levels and transfer probes are recorded in `results/finite_3d_rf_fem.json`; the 0.42→0.32 mm changes are 4.7% and 5.6%, so the mesh trend is explicitly unresolved. Neither result is a segmented-DC FEM convergence claim.

## Минимальные тесты

`tests/test_model.py` checks derivatives, Laplace trace of the 3D DC quadrupole, scalar/vector RHS agreement and charge/mass scaling. Four tests pass. These checks validate code identities, not physical correctness of the selected geometry.

### Reproduction

From the study folder run `python code/run_screening_maps.py`, `python code/validate_selected.py`, `python code/make_aux_figures.py`, `python code/publish_figures.py`, and `python code/finite_3d_rf_fem.py`. The map script used 10 workers (machine reported 12 logical processors and about 15.4 GiB available RAM) and wrote raw CSV plus metadata JSON. Figures are saved in PNG 600 dpi, PDF and SVG; `FIGS01` is an additional frequency-ratio map. Outputs are not a substitute for a converged finite 3-D segmented-field solve, charge characterization, or experimental validation.
