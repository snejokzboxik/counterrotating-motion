# Усреднённая RF-сила и два момента

Пусть \(\phi=\Re[a\rho^n e^{i(k\theta-\nu\tau)}]\), \(n=|k|\), \(\sigma=\operatorname{sgn}(k)\), \(\alpha=Q/\mu\). Для одного вращающегося гармонического потенциала с линейным сопротивлением (2D frozen-position, leading averaging):
\[
\overline F_\rho=-K_n\rho^{2n-3},\qquad
\overline F_\theta=\sigma K_n\frac{\lambda}{\nu}\rho^{2n-3},\qquad
K_n=\frac{\alpha^2n^2(n-1)|a|^2}{\nu^2+\lambda^2}.
\tag{1}
\]
Для \(n=1\) градиент поля пространственно постоянен, и drift-компоненты исчезают. Радиальная часть остаётся при \(\lambda=0\); направленная часть пропорциональна chirality \(\sigma\) и диссипации.

Момент на медленную координату:
\[
M_{\rm gc}=\mu\rho\overline F_\theta
=\sigma\frac{Q^2}{\mu}n^2(n-1)|a|^2
\frac{\lambda}{\nu(\nu^2+\lambda^2)}\rho^{2n-2}.
\tag{2}
\]
Полный средний RF-момент на микродвижущейся частице другой:
\[
\langle M_{\rm RF}\rangle=
\sigma\frac{Q^2}{\mu}n^3|a|^2
\frac{\lambda}{\nu(\nu^2+\lambda^2)}\rho^{2n-2},
\quad
\langle L_\mu\rangle=
\sigma\frac{Q^2}{\mu}n^2|a|^2
\frac{\rho^{2n-2}}{\nu(\nu^2+\lambda^2)}.
\tag{3}
\]
Они удовлетворяют
\[
\langle M_{\rm RF}\rangle=M_{\rm gc}+\lambda\langle L_\mu\rangle.
\tag{4}
\]
Поэтому равновесие медленной орбиты использует (2), а точный средний баланс полного углового момента использует (3) и сопротивление микродвижения.

При фиксированных \(n,\nu\) drift torque исчезает и при \(\lambda\to0\), и при \(\lambda\to\infty\); функция \(\lambda/(\nu^2+\lambda^2)\) максимальна при \(\lambda=\nu\). Это локальное следствие приближения, не глобальный критерий устойчивости. Для разных независимых frequency families складываются отдельные вклады (1); для одной частоты сначала складываются комплексные амплитуды и сохраняются cross terms.
