# Усреднение RF-микродвижения с линейным сопротивлением

Для замороженной медленной координаты \(\mathbf X\) рассмотрим
\[
\mathbf X''+\lambda\mathbf X'=\mathbf F_{\rm slow}(\mathbf X)+
\Re[\mathbf f(\mathbf X)e^{-i\nu\tau}].
\]
Первый быстрый response удовлетворяет
\[
\boldsymbol\xi''+\lambda\boldsymbol\xi'=
\Re[\mathbf f(\mathbf X)e^{-i\nu\tau}],\qquad
\hat{\boldsymbol\xi}=\frac{\mathbf f(\mathbf X)}{-\nu^2-i\lambda\nu}.
\tag{1}
\]
Раскрытие RF-силы в \(\mathbf X+\boldsymbol\xi\) даёт ведущую ненулевую среднюю поправку
\[
\overline{\mathbf F}_{\rm eff}=
\frac12\Re[(D\mathbf f)\hat{\boldsymbol\xi}^{\,*}]
=\frac12\Re\left[(D\mathbf f)\frac{\mathbf f^*}{-\nu^2+i\lambda\nu}\right].
\tag{2}
\]
Скобка означает матрицу \(D\mathbf f\), умноженную на conjugate-вектор. Конвенция носителя — \(e^{-i\nu\tau}\); при \(e^{+i\nu\tau}\) меняется знак мнимой части знаменателя.

Применение (2) к slow dynamics требует малой локальной поправки
\[
\epsilon_\mu=\frac{\|D\mathbf f\|}{\nu\sqrt{\nu^2+\lambda^2}}\ll1
\]
и медленного изменения \(X\), амплитуд и RF-фаз на одном carrier cycle. После перехода в движущуюся систему нужно также контролировать \(\nu_{\rm eff}=\nu-k\bar\omega\). Здесь нет универсального числового порога для \(\epsilon_\mu\); резонансы, сильная связь частиц и быстрые огибающие требуют отдельного усреднения.

Для набора RF-компонент \(\mathbf F=\sum_a\Re[\mathbf f_a e^{-i\nu_a\tau}]\) отклик первой гармоники равен \(\hat{\boldsymbol\xi}_a=\mathbf f_a/D_a\), где \(D_a=-\nu_a^2-i\lambda\nu_a\). На конечном интервале длины \(T\), центрированном в \(\tau_c\), произведение компоненты с индексом \(a\) и отклика \(b\) имеет среднее
\[
\left\langle\Re[\mathbf A e^{-i\nu_a\tau}]\,\Re[\mathbf B e^{-i\nu_b\tau}]\right\rangle_T
=\frac12\Re\!\left[\mathbf A\mathbf B^*e^{-i(\nu_a-\nu_b)\tau_c}\operatorname{sinc}\!\left(\frac{(\nu_a-\nu_b)T}{2}\right)+\mathbf A\mathbf B e^{-i(\nu_a+\nu_b)\tau_c}\operatorname{sinc}\!\left(\frac{(\nu_a+\nu_b)T}{2}\right)\right],
\]
где \(\operatorname{sinc}x=\sin x/x\), а в векторном среднем \(\mathbf A\mathbf B\) означает нужное действие матрицы \(D\mathbf f_a\) на вектор \(\hat{\boldsymbol\xi}_b\). Поэтому конечное среднее имеет beat-остаток порядка \(1/(|\nu_a-\nu_b|T)\), а вторую сумму нельзя отбрасывать, пока не проверено \((\nu_a+\nu_b)T\gg1\). При разных ненулевых положительных частотах обе части исчезают в бесконечном среднем; при близких частотах соответствующий разностный член сохраняется на медленном временном масштабе.

Для одной и той же ненулевой частоты сумма комплексных амплитуд сначала объединяется: \(\mathbf f_\Sigma=\sum_a\mathbf f_a\), \(D\mathbf f_\Sigma=\sum_aD\mathbf f_a\). Тогда
\[
\overline{\mathbf F}_{\rm eff}=\frac12\Re\!\left[(D\mathbf f_\Sigma)\left(\frac{\mathbf f_\Sigma}{-\nu^2-i\lambda\nu}\right)^*\right]
=\frac12\Re\sum_{a,b}(D\mathbf f_a)\hat{\boldsymbol\xi}_b^*.
\]
Последняя сумма содержит cross terms между пространственными гармониками. Для несоизмеримых частот на расширенном фазовом торе средние cross terms также равны нулю, если ни одна пара несёт одинаковую абсолютную частоту; соизмеримость сама по себе не создаёт постоянного члена в произведении двух первичных гармоник, но определяет общий период, нужный для точного конечного среднего. Более высокие порядки возмущений могут содержать резонансные комбинации вида \(m_1\nu_1+\cdots+m_L\nu_L\approx0\) и требуют отдельного нормального/усреднённого разложения.

Эти утверждения относятся к локальной замороженной координате и линейной системе с постоянными \(\lambda\). При движении по орбите фазовая скорость пространственной моды меняется на \(\nu_{\rm eff}=\nu-k\bar\omega\); если она мала или сравнима с медленной частотой, frozen-carrier формула \(D_a^{-1}\) не является равномерным приближением.

Воспроизводимая символьная проверка и независимая фазовая квадратура лежат в symbolic/ и numerical/. Область применимости и ошибки для local15 указаны в NUMERICAL_VALIDATION_RU.md.
