# V17C.22 — передача сессии

## Git

- Ветка: `research/v17c22-counterrotation-attraction-theory`
- Родительский commit: `894c6896ce5927371708037ffa4a4021189d2b54`
- Источник: `research/v17c21b-m3-bifurcation-diagrams`
- `master` не изменялся.
- Публикация ожидает финальный commit/push и проверку удалённого SHA.

## Вывод этой итерации

Самый сильный обоснованный статус полной модели — `LONG_LIVED_COUNTERROTATION`. M3 slow-flow стабилен на своём 10D averaged state; полная система показывает десять сильных отрицательных conditional exponents и два почти нулевых. Нормальное притяжение к геометрически установленному тору не доказано.

Динамические chain-rule/coriolis/euler/coulomb поправки M3+ выведены, но замкнутая и численно сошедшаяся M3+ ещё не реализована. Две короткие DOP853 проверки подтверждают исполнение full RHS и избирательное сжатие отдельных возмущений, но не служат доказательством аттрактора.

## Новые файлы

- Главный текст: `STABLE_COUNTERROTATION_THEORY_RU.md`.
- Вывод: `NONADIABATIC_M3_DERIVATION.tex`.
- Технические главы: `M3_PLUS_ANALYTICAL_MODEL_RU.md`, `FULL_RF_VARIATIONAL_STABILITY_RU.md`, `TRANSVERSE_ATTRACTION_RU.md`, `INVARIANT_SET_GEOMETRY_RU.md`, `PERTURBATION_RECOVERY_RU.md`, `ANALYTICAL_VS_FULL_STABILITY_RU.md`, `NUMERICAL_VALIDATION_RU.md`.
- Код: `code/audit_stability_evidence.py`, `code/full_rf_short_perturbation.py`, `code/test_full_rf_jacobian.py`.
- Результаты и рисунки: `results/`, `figures/`.

## Незакрытые вопросы

1. Два почти нулевых механических направления: внутренние фазы, очень слабое притяжение или переходный процесс.
2. Инвариантное множество и его независимые частоты/размерность.
3. Поперечное подпространство и нормальные показатели.
4. Численное динамическое Fourier-замыкание M3+ с обновляемыми ξ, \(\dot X\), \(\ddot X\) и Coulomb, зависящим от совместных орбитальных фаз.
5. Независимая длинная валидация M3+ после восстановления Numba runtime или ускорения решателя.

## Рекомендуемое продолжение

Сначала восстановить воспроизводимый runtime для V20B QR/RK4, затем вычислить фазово-условленные спектры на более длинном горизонте. Параллельно построить Fourier parameterization (K(\varphi)) для кандидатов с (d_o=0,1,2), сравнить остаток уравнения инвариантности и получить нормальный cocycle. Только после этого проверять предсказания динамической M3+ на сохранённых perturbation cases. Номинальные RF-параметры оставить неизменными.
