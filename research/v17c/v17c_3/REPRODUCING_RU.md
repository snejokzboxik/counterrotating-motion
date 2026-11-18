# Воспроизведение V17C.3

Рабочий каталог: `research/v17c/v17c_3/`. Требуются Python 3.12+, NumPy, SciPy, Numba, Matplotlib и `triangle` (Python Triangle). Скрипты запускаются из корня репозитория.

```powershell
python research/v17c/v17c_3/run_experiments.py
python research/v17c/v17c_3/pseudopotential_stationary_scan.py
python research/v17c/v17c_3/confirm_g0.py
python research/v17c/v17c_3/long_step_validation.py
python research/v17c/v17c_3/dop853_validate.py
python research/v17c/v17c_3/continuation_scan.py
python research/v17c/v17c_3/targeted_pair_scan.py
python research/v17c/v17c_3/boundary_shell_scan.py
python research/v17c/v17c_3/confirm_boundary_shell.py
python research/v17c/v17c_3/plot_summary.py
python research/v17c/v17c_3/package_release.py
```

`run_experiments.py` выполняет первичные карты и связанные парные старты; `confirm_g0.py` — длинную контрольную ветвь, возмущения, шаговую сходимость и DC-скан; `dop853_validate.py` — независимую интеграцию. Скрипты граничной полосы запускаются отдельно, поскольку требуют интегрирования около границы свободной апертуры. Основные параметры и результаты повторных запусков сохраняются в JSON/CSV/NPZ в `results/`.

`χ` — масштаб кулоновской силы в принятой безразмерной системе, а не напряжение отдельного электрода. Карты безразмерные; физическая оценка G0 в отчёте условна. Манифест фиксирует SHA-256 исходных файлов и расчётных артефактов.

### Подробная проверка шестиэлектродных G1/G2

Ниже перечислены команды для независимого P1 FEM-решения, полного спектра, точных трансляционных частот и трёхмерного парного поиска. `g1_g2_joint_search.py` выполняет 900 стартов на геометрию за 320 периодов, а `g1_g2_barrier_pair_search.py` — 600 стартов на геометрию до 600 периодов с начальными энергиями, выбранными относительно угловых барьеров.

```powershell
python research/v17c/v17c_3/g1_g2_fem_full_spectrum.py
python research/v17c/v17c_3/g1_g2_floquet_frequency_fit.py
python research/v17c/v17c_3/g1_g2_dop853_frequency_validation.py
python research/v17c/v17c_3/g1_g2_joint_search.py --samples 900 --periods 320 --steps 32
python research/v17c/v17c_3/g1_g2_barrier_pair_search.py --samples 600 --periods 600 --steps 48
python research/v17c/v17c_3/classify_g1_g2_pair_runs.py
python research/v17c/v17c_3/plot_g1_g2_deep_dive.py
```

Последующая строгая классификация применена к сохранённой широкой выборке без повторения 1 800 интегрирований. Одно состояние с противоположными знаками только в двух последних угловых окнах не удовлетворяло десятиоборотному критерию и переклассифицировано как короткий знакопеременный транзиент. Оно не сохраняется как подтверждённый кандидат.

Скрипты сводки и графиков строят диагностические рисунки, а `package_release.py` пересобирает `V17C3_results.zip`, `MANIFEST.json` и sidecar-файл SHA-256. Полный повтор всех длительных запусков не нужен для чтения включённых данных; их параметры и сохранённые ряды находятся в `results/`.
