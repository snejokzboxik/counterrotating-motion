# Повторение V17C.2

Для MFS исходного архива нужны Python 3, NumPy и SciPy. Скопированные исходники и расчётные таблицы находятся в `input/`; запустить из корня `v17c_2`:

```powershell
python input/scan.py
python input/candidates.py
python input/independent_audit.py
```

Для P1 FEM-аудита дополнительно нужны `triangle`, NumPy, SciPy и Matplotlib. Если `triangle` не установлен в активном Python, укажите каталог установленного пакета в `V17C_TRIANGLE_PATH`, затем выполните:

```powershell
python independent_fem_audit.py
```

Динамические скрипты используют NumPy, SciPy, Numba и Matplotlib. Начать с воспроизведения компактных карт можно так:

```powershell
python averaged_torque.py
python single_particle_maps.py
python synchronous_resonance_scan.py
python outer_phase_search.py
python negative_damping_scan.py
python summarize_results.py
```

Длинные сценарии отдельно запускают DOP853, балансы и 50 000-периодную траекторию. Времена расчёта зависят от CPU и параметров сетки/горизонта. Пользовательский исходный ZIP сохранён локально в проекте; в `V17C2_results.zip` включены извлечённые входные таблицы, но не дублированный архив.
