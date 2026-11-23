# SESSION_HANDOFF — V17C.21.2

## Исходное состояние

Рабочая ветка: research/v17c21b-m3-bifurcation-diagrams. База: 94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8. Все новые файлы этого этапа находятся в research/v17c/v17c_21b/. Master и исторические каталоги не редактировались.

## Итог

Номинальная M3-точка refine12_sobol17 воспроизведена. Выполнены четыре 1D continuation, три sparse 2D M3-карты, три zero-spin augmented solves с сеточным пере-решением, девять выборочных 10D M3 slow-flow spectra и семь независимых полных RF интегрирований. Найдены M3 переходы по нулю omega1 в A, nu и beta; полная RF-модель подтвердила номинальный CR, CO при A=0.8A0 и CR при A=0.95A0. Точные boundary trajectory tests остались UNRESOLVED из-за малого количества оборотов. Три альтернативных начальных приближения сошлись к одному номинальному CR(+−) корню M3; это не доказательство его единственности.

## Модельные оговорки

M3 slow-flow — адиабатическая редуцированная модель, не полная RF устойчивость. Полная RF проверка использует бесконечно длинные поперечные MFS-электроды и идеальный DC-квадруполь, без поляризации и конечных торцевых полей. Аттракторы и бассейны не доказывались.

## Воспроизведение

Основные скрипты: code/m3_continuation.py, code/refine_m3_sign_events.py, code/m3_sparse_transition_maps.py, code/event_grid_convergence.py, code/m3_event_and_stability.py, code/alternative_roots.py, code/full_rf_validation.py, code/plot_continuation.py, code/write_manifest.py. Тесты: tests/test_v21b_guards.py. Пройдены 4 unit tests, строгая загрузка 36 JSON и компиляция всех Python-файлов; PNG имеют 600 dpi. Созданы FIG01–FIG08; FIG09 не построен из-за недостатка независимо разрешённых точек по границе.

Исходные данные: results/continuation_*.json, results/event_omega1_zero_*.json, results/event_grid_convergence*.json, results/selected_m3_stability.json, results/sparse_map_*.json, results/full_rf_validation.json и рядом NPZ-файлы траекторий.

## Следующая сессия

Все расчётные процессы остановлены. FIG01–FIG08, данные, код, тесты и SHA-256 manifest созданы. Временная вычислительная среда удалена из checkout. Никакое продолжение исследования не запускается автоматически.

## Контроль публикации

Исследовательская ветка опубликована обычным fast-forward push. Первый payload commit: 5c47193baa375f548c3e1c0646587e26cff172c8. Удалённый SHA этой ветки проверен через git ls-remote и совпал с локальным HEAD на момент проверки. Master не изменён: удалённый SHA e5e6b2473bf519c5670ee1c48882183fbcfc8f2d до и после публикации совпадает. Force-push не использовался.

## Статус

Публикация подтверждена; публикационная проверка внесена в этот handoff после payload commit. Конечный branch tip сверяется с удалённым SHA при завершении текущей сессии.
