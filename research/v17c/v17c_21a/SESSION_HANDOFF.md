# SESSION HANDOFF — V17C.21.1

## Состояние исследования

- Рабочая ветка: `research/v17c21a-m3-convergence-stability`.
- База ветки: `05a8d8ce811c9858ab1e74b67b15139c557018b1` (`research/v17c21-analytical-counterrotation`).
- Все новые материалы находятся только в `research/v17c/v17c_21a/`.
- Исторические V20A/V20B/V21 материалы не изменялись.
- Вычислительные процессы остановлены; новых траекторий, Sobol-поиска или оптимизации нет.

## Научный итог

1. Внутренняя M3 гармоническая система теперь имеет явный нелинейный residual и fail-fast gate. Для трудной первой частицы fixed-point diverges, adaptive relaxation stalls, Anderson converges slowly, JFNK converges robustly.
2. Исправлена вещественно-согласованная derivative-конвенция для self-conjugate чётно-сеточных Nyquist-мод.
3. Все четыре обязательные сетки повторно решили полный шестипеременный slow-root M3. Заданные сеточные пороги сходимости выполнены.
4. Fast/slow cutoff (0.02)–(0.10) меняет множество мод и знаменатели, но практически не сдвигает номинальный root.
5. 2× nonlinear quadrature на замороженном fine-grid root даёт малые изменения средних, но не доказывает полного Fourier-tail convergence.
6. M3 nominal отклоняется от архивной full RK4 динамики на (-0.123\%), (-0.024\%) по радиусам и (+0.136\%), (+0.590\%) по частотам. На четырёх hold-out точках максимальная ошибка частоты около (2\%).
7. Вычислен локальный адиабатический M3 slow-flow спектр с десятью устойчивыми модами; он не доказывает аттрактор полной RF-системы.
8. В одномерных isolated-particle сечениях ω=0 RF-момент меняет знак в интервалах (r/R=[0.673516,0.673750]) и ([0.670938,0.671172]), примерно на 10% выше leading estimate (0.610413R). Средний drag torque в точках RF-zero остаётся ненулевым; это не граница смены вращения.

Основная интерпретация и ограничения: `M3_CONVERGENCE_AND_STABILITY_RU.md`.

## Основные файлы

- `NONLINEAR_SOLVER_COMPARISON_RU.md` — benchmark и диагноз solver failure.
- `FOURIER_CONVERGENCE_RU.md` — полные roots по сеткам и aliasing/Nyquist checks.
- `FAST_SLOW_SEPARATION_RU.md` — cutoff scan.
- `M3_SLOW_FLOW_DERIVATION.tex` и `M3_STABILITY_RU.md` — приближённый slow-flow и его локальный спектр.
- `ANALYTICAL_VALIDATION_RU.md` — M2/M3/full-RK4 сравнение и finite-window аудит.
- `results/` — JSON и NPZ checkpoints; `tables/` — CSV исходные ряды графиков; `figures/` — FIG01–FIG09 в PDF/SVG/PNG.
- `code/` — solvers, кампании, audit, воспроизводимость, figure generation и manifest builder.
- `tests/test_m3_regression.py` — 5 регрессионных проверок.
- `MANIFEST.json` — SHA-256 список deliverable и внешних эталонов; сам manifest в список не включён.

## Воспроизведение

Из корня репозитория:

```powershell
python -m compileall -q research/v17c/v17c_21a/code research/v17c/v17c_21a/tests
python -m unittest discover -s research/v17c/v17c_21a/tests -v
python research/v17c/v17c_21a/code/reference_window_audit.py
python research/v17c/v17c_21a/code/postprocess_figures.py
python research/v17c/v17c_21a/code/build_manifest.py
```

Дополнительный JFNK root check запускает новое решение на сетке 96×8×8 из seed, намеренно смещённого относительно опубликованного корня:

```powershell
python research/v17c/v17c_21a/code/reproducibility_check.py
```

Он занимает примерно 50 секунд на текущем локальном компьютере. Тяжёлые сеточные и hold-out JSON уже сохранены; для воспроизведения отчёта повторно гонять все долгие расчёты не нужно.

## Ограничения следующего шага

Для более строгой теории нужно либо вывести полноценный нестационарный M3 slow-flow с slow-acceleration correction в fast solve, либо доказать достаточность квазистационарного closure в проверяемом параметрическом neighborhood. Следующий устойчивостный этап должен проверять притяжение полного расширенного состояния с обеими RF-фазами и не использовать отрицательные eigenvalues этого reduced flow как замену Lyapunov/Floquet-анализу полной системы.

Перед любым расширением модели следует повторно проверить источник и SHA ветки, конвенцию drag/масштабов, статус V20B finite-window reference, cutoff gap и то, что новые данные не подменяют исторические корни.
