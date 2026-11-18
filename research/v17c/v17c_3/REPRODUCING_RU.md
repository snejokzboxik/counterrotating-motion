# Воспроизведение V17C.3

Рабочий каталог: `research/v17c/v17c_3/`. Требуются Python 3.12+, NumPy, SciPy, Numba и Matplotlib. Скрипты запускаются из корня репозитория.

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

Последние два скрипта строят сводные графики и пересобирают `V17C3_results.zip`, `MANIFEST.json` и sidecar-файл SHA-256. Полный повтор всех длительных запусков не нужен для чтения включённых данных; их параметры и сохранённые ряды находятся в `results/`.
