# Литературный контекст и границы переноса результатов

## RF-ловушки и усреднение

Классическая теория RF-ловушек описывает удержание как результат временно-периодического поля и усреднённых эффектов микродвижения; поэтому знак/устойчивость равновесия усреднённой модели сам по себе не заменяет анализ полной driven системы. Для нашего вывода это методологическая рамка, а не доказательство применимости конкретного режима к электродам V17C. См. Dehmelt, *Radiofrequency Spectroscopy of Stored Ions I: Storage*, DOI 10.1016/S0065-2199(08)60170-0, и Landa et al., *Modes of Oscillation in Radiofrequency Paul Traps*, DOI 10.1088/1367-2630/14/9/093023.

Landa et al. строят Floquet–Lyapunov описание линеаризованной time-periodic задачи и обсуждают ограничения псевдопотенциала. Это прямо мотивирует обозначать instantaneous axial eigenvalues FIG06 только как снимки Hessian, а не Floquet показатели. Независимый экспериментальный/теоретический анализ плоских ионных кристаллов Kaufmann et al. показал, что pseudopotential approximation может хорошо давать положения и переходы, но хуже предсказывать собственные частоты, тогда как full time-dependent Floquet-Lyapunov расчёт согласуется точнее (DOI 10.1103/PhysRevLett.109.263003). Этот результат не переносится количественно на V17C, но подтверждает необходимость отделять квазистатические корни от динамического спектра.

## Коллективные Coulomb структуры

Dubin разработал анализ структурных переходов в confined Coulomb crystals, включая planar buckling и многослойные структуры, в зависимости от анизотропии удерживающего потенциала (DOI 10.1103/PhysRevLett.71.2753). Fishman et al. рассматривают структурные переходы в низкоразмерных ионных кристаллах (DOI 10.1103/PhysRevB.77.064111). Эти работы подтверждают, что осевые моды и кулоновская Hessian — необходимая часть multi-particle анализа; они не предполагают устойчивое разнесение колец для driven, damped, oppositely rotating multi-species режима V17C.

## Управление вращением в Penning-плазмах: только аналогия

Huang et al. экспериментально показали rotating-wall control и phase-locked вращение crystallized non-neutral plasmas в Penning–Malmberg trap (DOI 10.1103/PhysRevLett.78.875; 10.1103/PhysRevLett.80.73). Там поле вращает коллективную плазму в магнитной ловушке, что физически не тождественно torque, возникающему во временно-периодическом multipole RF поле V17C. Эти публикации используются только как пример экспериментально измеряемого управления вращением и коллективной фазовой фиксации, не как подтверждение встречновращения или механизма данной модели.

## Библиографические источники

Полные записи в BibTeX находятся в `references.bib`. Основные DOI:

1. H. G. Dehmelt, *Radiofrequency Spectroscopy of Stored Ions I: Storage*, Adv. At. Mol. Phys. **3**, 53 (1967), [10.1016/S0065-2199(08)60170-0](https://doi.org/10.1016/S0065-2199(08)60170-0).
2. D. H. E. Dubin, *Theory of Structural Phase Transitions in a Trapped Coulomb Crystal*, Phys. Rev. Lett. **71**, 2753 (1993), [10.1103/PhysRevLett.71.2753](https://doi.org/10.1103/PhysRevLett.71.2753).
3. S. Fishman, G. De Chiara, T. Calarco, G. Morigi, *Structural phase transitions in low-dimensional ion crystals*, Phys. Rev. B **77**, 064111 (2008), [10.1103/PhysRevB.77.064111](https://doi.org/10.1103/PhysRevB.77.064111).
4. H. Landa, M. Drewsen, B. Reznik, A. Retzker, *Modes of oscillation in radiofrequency Paul traps*, New J. Phys. **14**, 093023 (2012), [10.1088/1367-2630/14/9/093023](https://doi.org/10.1088/1367-2630/14/9/093023).
5. H. Kaufmann et al., *Precise experimental investigation of eigenmodes in a planar ion crystal*, Phys. Rev. Lett. **109**, 263003 (2012), [10.1103/PhysRevLett.109.263003](https://doi.org/10.1103/PhysRevLett.109.263003).
6. X.-P. Huang et al., *Steady-State Confinement of Non-neutral Plasmas by Rotating Electric Fields*, Phys. Rev. Lett. **78**, 875 (1997), [10.1103/PhysRevLett.78.875](https://doi.org/10.1103/PhysRevLett.78.875).
7. X.-P. Huang et al., *Phase-Locked Rotation of Crystallized Non-neutral Plasmas by Rotating Electric Fields*, Phys. Rev. Lett. **80**, 73 (1998), [10.1103/PhysRevLett.80.73](https://doi.org/10.1103/PhysRevLett.80.73).

## Ограничение библиографического охвата

Это краткий первичный список для теоретической главы, а не исчерпывающий обзор микрочастичных ловушек, газового пробоя, зарядовой эмиссии или вязкого охлаждения. Новые физические параметры и лабораторный дизайн в V17C.14 не оптимизировались; для них нужно отдельное предметное исследование с материал- и геометрия-специфическими источниками.
