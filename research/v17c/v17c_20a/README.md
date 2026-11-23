# V17C.20A — RF attractor search

Рабочая директория исследования динамической природы долгоживущего встречного вращения. Исходная база — опубликованный V20B SHA `d1bf252649e174166becf464269318b5f97acd5d`; результаты предыдущей серии оставлены неизменными.

## Сначала читать

1. [RF_ATTRACTORS_AND_INVARIANT_TORI_RU.md](RF_ATTRACTORS_AND_INVARIANT_TORI_RU.md) — главный вывод и перечень фигур.
2. [LYAPUNOV_LONG_HORIZON_RU.md](LYAPUNOV_LONG_HORIZON_RU.md) — почти нулевые показатели.
3. [TORUS_DIAGNOSTICS_RU.md](TORUS_DIAGNOSTICS_RU.md) — сечения, возмущения и ограничения.
4. [MULTISTABILITY_RU.md](MULTISTABILITY_RU.md) — одинаковофазовые старты.
5. [RF_PARAMETER_SEARCH_RU.md](RF_PARAMETER_SEARCH_RU.md) — покрытие RF-параметров.
6. [NUMERICAL_VALIDATION_RU.md](NUMERICAL_VALIDATION_RU.md) — проверки реализации и воспроизводимость.
7. [SESSION_HANDOFF.md](SESSION_HANDOFF.md) — краткая передача работы.

## Содержимое

- `code/` — исполняемые скрипты QR, torus diagnostics, RF-screen, multistart, candidate packaging and validation.
- `results/` — checkpoints, состояния, CSV/JSON/NPZ, audit outcomes and plotted data.
- `figures/` — FIG01–FIG08 в PNG 600 dpi, PDF и SVG.
- `ATTRACTOR_CANDIDATES.json` — стартовые условия, фазы, безразмерные параметры, горизонты и SHA кода.
- `PARAMETER_COVERAGE.json` — диапазоны, количество точек и классификации.
- `MANIFEST.json` — SHA-256 сохранённых материалов.

## Ключевое ограничение

Исследование использует бесконечный RF-электродный MFS-базис и идеальный осесимметричный DC-квадруполь; polarization disabled. Оно не проверяет FEM, конечную геометрию или физическую конструкцию ловушки.
