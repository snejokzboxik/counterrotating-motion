# Осевые коллективные моды и гармоника DC четвёртого порядка

## Квадрупольный планарный режим

Для \(\phi_{\rm DC}=\beta(\zeta^2-\rho^2/2)\) осевое возмущение планарной траектории:
\[
\delta\zeta_i''=-\lambda_i\delta\zeta_i'-2\beta\frac{Q_i}{\mu_i}\delta\zeta_i+
\frac{\kappa}{\mu_i}\sum_{j\ne i}Q_iQ_j
\frac{\delta\zeta_i-\delta\zeta_j}{d_{ij}^3(\tau)}.
\tag{1}
\]
Это следует из \(\partial_z\phi_{\rm DC}=2\beta\zeta\) и линейного разложения кулоновской силы по осевой координате. Матричная форма \(\delta z''+\Lambda\delta z'+S(\tau)\delta z=0\): DC вносит положительную diagonal stiffness \(2\beta Q_i/\mu_i\), а repulsive Coulomb уменьшает осевую stiffness вдоль относительных мод.

Для одного изолированного равностороннего треугольника radius \(\rho\), одинаковых \(Q,\mu\), две внутренние осевые моды имеют \(\lambda_{\rm Lap}=1/(\sqrt3\rho^3)\). Frozen-ring stiffness:
\[
\omega_{z,\rm int}^2=2\beta Q/\mu-
\frac{\kappa Q^2}{\sqrt3\,\mu\rho^3}.
\tag{2}
\]
Planar threshold \(\beta_{\rm crit}=\kappa Q/(2\sqrt3\rho^3)\) — точный только для изолированного rigid-radius triangle. Для двух встречновращающихся колец \(d_{ij}(\tau)\) меняются, поэтому (1) — periodic или quasiperiodic matrix; instantaneous positive eigenvalues не заменяют динамические Lyapunov exponents.

Для local15 frozen pair из V17C.13 радиусы \(0.5480,0.7661\), \(\beta=0.0172733\), \(\kappa=C_0/Q_2=3.03503\cdot10^{-6}\). Figure 06 вычисляет mass-symmetrized 6×6 instantaneous axial stiffness во всём относительном C3-фазовом интервале, с полными межкольцевыми парами. Это проверяет знак замороженной stiffness, но не параметрическую устойчивость.

## DC гармоника четвёртого порядка

Регулярный осесимметричный гармонический полином
\[
H_4=\zeta^4-3\rho^2\zeta^2+\frac38\rho^4
\]
удовлетворяет \(\nabla^2H_4=0\). Для \(\phi_{\rm DC}=\beta_2H_2+\beta_4H_4\), \(H_2=\zeta^2-\rho^2/2\), осевая производная и кривизна:
\[
\partial_\zeta\phi=2\beta_2\zeta+
\beta_4(4\zeta^3-6\rho^2\zeta),
\quad
\partial_{\zeta\zeta}\phi=2\beta_2+\beta_4(12\zeta^2-6\rho^2).
\tag{3}
\]
При фиксированном радиусе \(z=0\) axially stable для положительного заряда, если \(\beta_2-3\beta_4\rho^2>0\). Ненулевые extrema:
\[
\zeta_*^2=\frac32\rho^2-\frac{\beta_2}{2\beta_4}.
\tag{4}
\]
Они существуют при правой части \(>0\), а axial curvature там \(8\beta_4\zeta_*^2\), поэтому это axial minima при \(\beta_4>0\). При \(\beta_2>0,\beta_4>0\) интервал
\[
\frac{1}{3\rho_2^2}<\frac{\beta_4}{\beta_2}<\frac{1}{3\rho_1^2},\quad \rho_2>\rho_1
\tag{5}
\]
даёт внутреннему радиусу локально устойчивую центральную плоскость, а внешнему — осево неустойчивый центр и два симметричных off-axis extrema. Для local15 shell radii интервал примерно \(0.567<\beta_4/\beta_2<1.110\).

Уравнение (5) — только фиксированный-\(\rho\) axial criterion. Оно не доказывает полную 3D устойчивость: \(H_4\) одновременно меняет radial force, а Coulomb и RF добавляют coupling. Зеркально-симметричный DC потенциал даёт пару \(\pm\zeta_*\), а не выбирает одну высоту.
