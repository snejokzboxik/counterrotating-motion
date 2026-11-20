# Обобщённые уравнения движения

## Область утверждений

Ниже приведена точная безразмерная модель классических точечных центров с кулоновским взаимодействием и линейным изотропным сопротивлением. Конечные размеры входят в критерии контакта и стенок, но не изменяют поле точечного заряда до контакта. Поляризация, гидродинамическая связь и нелинейное сопротивление не включены в основной RHS.

## Нормировка и поле

Пусть \(\tau=\Omega _1t\), \(\mathbf x_i=\mathbf r_i/R\), \(\mathbf u_i=d\mathbf x_i/d\tau\), \(\mu_i=m_i/m_1\), \(Q_i=q_i/q_1\), а
\[
\phi=\frac{q_1\Phi}{m_1R^2\Omega _1^2},\quad
\lambda_i=\frac{\gamma_i}{m_i\Omega _1},\quad
\kappa=\frac{q_1^2}{4\pi\varepsilon m_1R^3\Omega _1^2}.
\]
Точное уравнение для \(N\) частиц:
\[
\mathbf x_i'=\mathbf u_i,\qquad
\mu_i\mathbf u_i'=-Q_i\nabla\phi(\mathbf x_i,\tau)
-\mu_i\lambda_i\mathbf u_i+
\kappa\sum_{j\ne i}Q_iQ_j
\frac{\mathbf x_i-\mathbf x_j}{|\mathbf x_i-\mathbf x_j|^3}.
\tag{1}
\]
Гравитацию можно добавить как \(\mu_i\mathbf G_i\), где \(\mathbf G_i=\mathbf g_i/(R\Omega _1^2)\). Нелинейное сопротивление заменяет линейный член явно заданной размерно согласованной функцией скорости.

Для двух видов \(\mu_1=Q_1=1\), \(\mu_2=m_2/m_1\), \(Q_2=q_2/q_1\). При \(s=m_1/m_2\) ускорения второго вида содержат \(sQ_2\) в электрическом поле и \(s\) перед межвидовым Coulomb-ускорением, если \(C=\kappa Q_2\). Общий N-body коэффициент для пары всегда содержит оба заряда \(Q_iQ_j\), а ускорение частицы \(i\) дополнительно делится на \(\mu_i\).

## Цилиндрические уравнения

Обозначим через \(f_{\rho i},f_{\theta i},f_{zi}\) компоненты всех сил на единицу массы, кроме сопротивления. Тогда
\[
\rho_i''-\rho_i\theta_i'^2=f_{\rho i}-\lambda_i\rho_i',
\]
\[
\rho_i\theta_i''+2\rho_i'\theta_i'=f_{\theta i}-\lambda_i\rho_i\theta_i',
\qquad
\zeta_i''=f_{zi}-\lambda_i\zeta_i'.
\tag{2}
\]
Здесь \(\rho=r/R\), \(\zeta=z/R\), а \(f\) — безразмерные ускорения.

## Вариационная система

Для \(\mathbf X'=\mathbf F(\mathbf X,\tau)\):
\[
\delta\mathbf X'=J(\mathbf X,\tau)\delta\mathbf X,\qquad
J=D_{\mathbf X}\mathbf F.
\tag{3}
\]
Блоки ускорения имеют вид \(J_{u_i,x_i}=-\alpha_iD^2\phi+J^C_{ii}\), \(J_{u_i,x_j}=J^C_{ij}\), \(J_{u_i,u_j}=-\lambda_iI\delta_{ij}\), где \(\alpha_i=Q_i/\mu_i\). Для \(\mathbf d_{ij}=\mathbf x_i-\mathbf x_j\), \(d=|\mathbf d|\):
\[
J^C_{ii}\mathrel{+}=\frac{\kappa Q_iQ_j}{\mu_i}
\left(\frac I{d^3}-3\frac{\mathbf d\mathbf d^T}{d^5}\right),\quad
J^C_{ij}\mathrel{-}=\frac{\kappa Q_iQ_j}{\mu_i}
\left(\frac I{d^3}-3\frac{\mathbf d\mathbf d^T}{d^5}\right).
\tag{4}
\]
Кулоновский блок симметризуется массовым масштабированием; координатная матрица по частицам в общем случае несимметрична из-за разных масс.

## Энергия и угловой момент

В нормировке \(m_1R^2\Omega _1^2\):
\[
\mathcal E=\frac12\sum_i\mu_i|\mathbf u_i|^2+
\sum_iQ_i\phi(\mathbf x_i,\tau)+
\kappa\sum_{i<j}\frac{Q_iQ_j}{d_{ij}}.
\tag{5}
\]
При постоянных параметрах и только линейном сопротивлении
\[
\mathcal E'=\sum_iQ_i\partial_\tau\phi(\mathbf x_i,\tau)-
\sum_i\mu_i\lambda_i|\mathbf u_i|^2.
\tag{6}
\]
Это баланс явной RF-работы; \(\sum_i(-Q_i\nabla\phi)\cdot\mathbf u_i\) не равна первому слагаемому мгновенно.

Момент \(L_i=\mu_i(\mathbf x_i\times\mathbf u_i)_z=\mu_i\rho_i^2\theta_i'\) удовлетворяет
\[
L_i'=-Q_i\partial_{\theta_i}\phi_{\rm RF}
+M_{C,i}-\lambda_iL_i.
\tag{7}
\]
Осеесимметричный DC-потенциал не даёт прямого момента. В каждой кулоновской паре внутренние моменты взаимно компенсируются, поэтому \(\sum_iM_{C,i}=0\) точно. Взаимодействие всё же перераспределяет момент и меняет групповые частоты.

## Автономное расширение

Для RF-фаз \(\psi_\ell'=\nu_\ell\) расширенная система автономна на фазовом торе. Рационально соизмеримые частоты дают общий период; несоизмеримые — квазипериодическое возбуждение, для которого обычная однопериодная теория Флоке без дополнительного сведения неприменима.
