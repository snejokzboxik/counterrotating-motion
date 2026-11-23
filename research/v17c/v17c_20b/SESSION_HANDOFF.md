# V17C.20B — Передача состояния исследования

## Итог текущей сессии

Все запланированные в этой сессии серии завершены; фоновых расчётов нет. Созданы воспроизводимые результаты для `candidate52`, `Local15`, `refine12_sobol17`:

- общие RK4/64 траектории на 15 000 периодов;
- corrected-drag replay refine12 на 240 000 периодов;
- corrected-drag общие траектории candidate52 и Local15 на 50 000 периодов;
- 504-точечные двумерные сечения, 72 однофакторных и 96 scrambled-Sobol short-screen запусков;
- Coulomb-multiplier sweep `0…5 C0`, 30 коротких запусков;
- DOP853 независимая Python-RHS проверка каждого кандидата на 3 000 периодах;
- RK4/128 step-refinement и сравнение с DOP853;
- условный механический QR на 2 000 периодах и расширенный фазовый QR на 500 периодах;
- 18 малых механических возмущений, 6 возмущений несущих фаз и frozen-start поднабор из 96 запусков.

Короткий verdict: встречные режимы действительно воспроизводятся при включённом Coulomb и сохраняются на длинных горизонтах. Притягивающий режим в строгом смысле пока не доказан: верхний conditional exponent близок к нулю и убывает примерно как `1/T`; конечные траектории малых возмущений не показывают однозначного экспоненциального слияния. Coulomb до `5C0` не меняет screen-направление на выбранном стартовом состоянии и не является необходимым условием встречного вращения, поскольку `C=0` также даёт CR.

## Ключевые файлы

- [Основной отчёт](PARTICLE_ATTRACTOR_SEARCH_RU.md).
- [Покрытие параметров](PARAMETER_COVERAGE.json).
- [Реестр кандидатов](ATTRACTOR_CANDIDATES.json).
- [Бассейны и ограничения выборки](BASINS_OF_ATTRACTION_RU.md).
- [QR/Ляпунов и возмущения](LYAPUNOV_VALIDATION_RU.md).
- [Параметры, MFS и исходные состояния](results/candidate_parameters.json), [start_conditions.json](results/start_conditions.json).
- [DOP853](results/independent_dop853_validation.json), [уточнение RK4](results/step_refinement_3000p.json), [вариационная валидация](results/variational_audit.json), [96-run frozen panel](results/basin_ensemble_summary.json).
- Визуализации находятся в `figures/`; массовые JSONL/NPZ и resumable chunks — в `results/`.

## Точные контрольные значения

Общий rest-start: seed `20261850`, исходный ensemble `rest`, start index `7`; начальные координаты и две RF-фазы записаны в `results/start_conditions.json`. Восстановленные размерные отображения: candidate52 `f1=45 Hz`, Local15 и refine12 `f1=50 Hz`, `R=10 mm`; частоты второй несущей и физические/безразмерные параметры полностью перечислены в `results/candidate_parameters.json`.

Длинная refine12 переинтеграция использует исторический V12B rest-start index 54 и фазу `[2.3066744440, 5.2344800999] rad`. Исходный архив содержал initial state и summary, но не полную историю состояния; переинтеграция не подменяет неизменённую историческую запись.

## Воспроизведение

Из корня репозитория:

```powershell
python research/v17c/v17c_20b/code/attractor_search.py --stage config
python research/v17c/v17c_20b/code/attractor_search.py --stage baseline
python research/v17c/v17c_20b/code/attractor_search.py --stage grid
python research/v17c/v17c_20b/code/attractor_search.py --stage coulomb
python research/v17c/v17c_20b/code/univariate_scan.py
python research/v17c/v17c_20b/code/basin_scan.py
python research/v17c/v17c_20b/code/long_replay.py --candidate refine12_sobol17 --periods 240000 --steps 64 --chunk-periods 5000 --transient 20000 --start refine12_240k_seed --tag refine12_240k_corrected_drag
python research/v17c/v17c_20b/code/stability_analysis.py
python research/v17c/v17c_20b/code/full_phase_qr.py
python research/v17c/v17c_20b/code/dop853_independent.py
python research/v17c/v17c_20b/code/step_refinement.py
python research/v17c/v17c_20b/code/plot_results.py
python research/v17c/v17c_20b/code/build_registries.py
```

Long-running scans append JSONL/checkpoint state and can be resumed by rerunning the same command. Baseline builders rewrite deterministic outputs; preserve source archive files. Для повторного long replay checkpoint path не удалять, если нужно продолжение; задавать те же candidate, step, tag и seed.

## Что осталось

1. Расширить frozen-start выборки до хотя бы `N=128` на ансамбль и кандидата; текущие 8 систематических стартов на ансамбль — только скрининг.
2. Продлить QR/механические возмущения хотя бы до 10–20 тысяч периодов, отслеживать сходимость почти нулевого механического показателя и сравнить несколько инвариантных фазовых точек.
3. Проверить явно сосуществование CR/CO при полностью одинаковом векторе параметров и RF-фазах, меняя несколько широких механических стартов.
4. Завершить недостающие парные параметрические сечения и решить, имеет ли отдельная ось `Q` смысл после фиксации `alpha1`, `alpha2`, `mu`, `C12`.
5. Выполнить физически связанный скан, который меняет реальные `q`, `a`, плотности/материал и drag совместно; legacy размерные отображения не подтверждают совместимость двух микросфер.
6. Передавать кандидаты V17C.20A только после согласования полной конвенции безразмерных коэффициентов и drag.
7. Проверить конечную 3D FEM-геометрию отдельно; текущая модель использует 2D поперечный MFS плюс идеальный DC-квадруполь, а не торцевое поле реальной ловушки.

## Изоляция и публикация

Исходная ветка/commit: `research/v17c18b19-axial-stability-diagrams` / `45866d0559db1615a5e2aebc521ddae90f3a0101`. Ветка этой работы: `research/v17c20b-particle-attractor-search`. Не изменять master, V17C и ветки V17C.18B–20A. После фиксации итогового commit записать его SHA в этот раздел и проверить удалённую ветку через `git ls-remote`.
