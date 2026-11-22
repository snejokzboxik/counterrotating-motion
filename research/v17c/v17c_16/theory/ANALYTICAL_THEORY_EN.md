# Analytical Dynamics of Two Counterrotating Charged Microparticles with Low-Reynolds-Number Drag

## Scope and status

This chapter develops a model for two spherical particles in a two-frequency RF field whose leading spatial harmonics are (k=+2) and (k=-4). It does not claim a coupled counterrotating solution for all physical parameters. Existence and stability for a particular pair must be established by independent integration of the full Coulomb-coupled equations with gravity, the selected drag model, and independent RF phases. Local15 is used as a frozen derivation and comparison point, not as a physically validated operating regime.

## 1. Dimensional and dimensionless equations

For spheres with radii (a_i), densities (ho_i), charges (q_i), and masses
\[
m_i=\frac{4\pi}{3}\rho_i a_i^3,
\]
the dimensional equations in quiescent gas are
\[
m_i\ddot{\mathbf r}_i=q_i\mathbf E(\mathbf r_i,t)
+\frac{q_iq_j}{4\pi\varepsilon_0}\frac{\mathbf r_i-\mathbf r_j}{|\mathbf r_i-\mathbf r_j|^3}
+\mathbf F_{d,i}+(m_i-\rho_fV_i)\mathbf g,\qquad j\ne i.
\]
Continuum Stokes drag is \(\mathbf F_{S,i}=-6\pi\eta a_i\mathbf v_i\). The quasi-steady Oseen approximation is
\[
\mathbf F_{O,i}=-6\pi\eta a_i\mathbf v_i\left(1+\frac{3}{16}\mathrm{Re}_{d,i}\right),\qquad
\mathrm{Re}_{d,i}=\frac{2\rho_fa_i|\mathbf v_i|}{\eta}=\frac{2a_i|\mathbf v_i|}{\nu_f}.
\]
Here (Re_d) is based on particle diameter; with a radius-based Reynolds number the coefficient becomes (3/8). Drag uses the full instantaneous velocity relative to the gas.

Set (\tau=\Omega_1t), (\mathbf x_i=\mathbf r_i/R), (\mathbf u_i=d\mathbf x_i/d\tau), (M_i=m_i/m_1), (Q_i=q_i/q_1), (\alpha_i=(q_i/m_i)/(q_1/m_1)=Q_i/M_i), (\lambda_i=6\pi\eta a_i/(m_i\Omega_1)), and
\[
\kappa_0=\frac{q_1^2}{4\pi\varepsilon_0m_1R^3\Omega_1^2},\qquad
\mathrm{Re}_{d,i}=K_i|\mathbf u_i|,\qquad K_i=\frac{2\rho_fa_iR\Omega_1}{\eta}.
\]
The exact selected Oseen model becomes
\[
\mathbf x_i'=\mathbf u_i,
\quad
\mathbf u_i'=-\alpha_i\nabla\phi(\mathbf x_i,\tau)
+\frac{\kappa_0Q_iQ_j}{M_i}\frac{\mathbf x_i-\mathbf x_j}{|\mathbf x_i-\mathbf x_j|^3}
-\lambda_i\mathbf u_i\left(1+\frac{3K_i}{16}|\mathbf u_i|\right)
+\left(1-\frac{\rho_f}{\rho_i}\right)\mathbf g/(R\Omega_1^2).
\]
Equivalently, define (C_{12}=\kappa_0Q_2=q_1q_2/(4\pi\varepsilon_0m_1R^3\Omega_1^2)). The Coulomb accelerations are then (C_{12}/M_1=C_{12}) for particle 1 and (C_{12}/M_2) for particle 2 because (Q_1=1).

The ideal screening potential is
\[
\phi=A\,\Re[(x+iy)^2e^{-i\tau+i\psi_1}]
+B\,\Re[(x-iy)^4e^{-i\nu\tau+i\psi_2}]
+\beta_2\left(z^2-\frac{x^2+y^2}{2}\right).
\]
This is a Laplace potential in the charge-free region; the DC term contains a corresponding radial field. (A) and (B) are dimensionless amplitudes of leading spatial harmonics, not electrode voltages. The MFS transfer factors (h_{+2},h_{-4}) give
\[
U_1=\frac{A m_1R^2\Omega_1^2}{q_1h_{+2}},\quad
U_2=\frac{B m_1R^2\Omega_1^2}{q_1h_{-4}},\quad
V_2=\frac{\beta_2m_1R^2\Omega_1^2}{q_1}.
\]
(V_2) is the coefficient of an ideal DC quadrupole, not an endcap electrode voltage.

## 2. Linear-drag moving-orbit averaging

For one particle, write a slow circular orbit
\[
\mathbf R_i(\tau)=\rho_i(\cos(\omega_i\tau+\theta_{i0}),\sin(\omega_i\tau+\theta_{i0}),z_i),
\quad \mathbf x_i=\mathbf R_i+\boldsymbol\xi_i,
\]
where (\boldsymbol\xi_i) is the rapid RF micromotion. For signed azimuthal mode (k_s), (n_s=|k_s|), and (\sigma_{is}=\Omega_s/\Omega_1-k_s\omega_i), the linear response in co-rotating coordinates is
\[
\mathcal D_{is}(\sigma_{is})\widehat{\xi}_{is}=\widehat{f}_{is},
\]
\[
\mathcal D_{is}=-\sigma_{is}^2I-i\sigma_{is}(\lambda_iI+2\omega_iJ)
-\omega_i^2I+\lambda_i\omega_iJ-\mathcal A_{i,DC},
\quad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]
Here (\widehat f_{is}) is the RF acceleration phasor at the orbit, and (\mathcal A_{i,DC}) is the slow DC acceleration Jacobian. The first-order mean correction from RF-force curvature is
\[
\overline{\mathbf F}_{RF,i}^{(1)}=\frac12\sum_s\Re\left[\mathcal H_{is}\widehat{\xi}_{is}^{*}\right],
\]
where (\mathcal H_{is}) is the complex spatial Jacobian of that RF acceleration. The formula retains the moving-orbit frequency (\sigma_{is}), Coriolis and centrifugal terms, and the DC Jacobian; it was the corrected portion of V17C.15. It does not include Coulomb phase modulation or nonlinear drag.

The zero-orbit-speed Local15 leading-averaging check with Stokes drag gives, for an isolated particle,
\[
K_1=\frac{4\alpha_i^2A^2}{1+\lambda_i^2},\qquad
K_2=\frac{48\alpha_i^2B^2}{\nu^2+\lambda_i^2},
\]
\[
\omega_i=K_1-\frac{K_2}{\nu}\rho_i^4,
\qquad \omega_i^2+\alpha_i\beta_2=K_1+K_2\rho_i^4.
\]
This algebraic leading model assumes (\epsilon_{RF}=|\xi|/\rho\ll1), small slow speed, no Coulomb coupling, and linear Stokes drag. It does not determine a capture probability, prove stability, or treat the pair as two independent circles when Coulomb coupling is appreciable.

## 3. New averaging of the nonlinear Oseen term

Let (\mathbf V_i=\dot{\mathbf R}_i) be the slow orbital velocity and (\mathbf w_i=\dot{\boldsymbol\xi}_i) the fast micromotion velocity. The mean nonlinear drag contains
\[
\left\langle |\mathbf V_i+\mathbf w_i|(\mathbf V_i+\mathbf w_i)\right\rangle.
\]
This is generally not equal to ( |\mathbf V_i|\mathbf V_i\). For two incommensurate RF frequencies, the appropriate first averaging is over the two-torus \((\varphi_1,\varphi_2)\in[0,2\pi)^2\), leaving the slow orbit phase as a separate coordinate:
\[
\mathcal G_i(\mathbf V_i;\rho_i,\omega_i,\theta_i)=\frac{1}{(2\pi)^2}\int_0^{2\pi}\!\int_0^{2\pi}
|\mathbf V_i+\mathbf w_i(\varphi_1,\varphi_2)|
(\mathbf V_i+\mathbf w_i(\varphi_1,\varphi_2))\,d\varphi_1d\varphi_2.
\]
The torus integral is evaluated by quadrature using the RF response. It retains both RF families and their combination frequencies. For a rational frequency ratio, use the actual common-period orbit with its linked phases instead.

The averaged drag is
\[
\overline{\mathbf F}_{d,i}=-\lambda_i\mathbf V_i
-\lambda_i\frac{3K_i}{16}\mathcal G_i.
\]
The first term is linear. The second depends jointly on slow rotation, both micromotion amplitudes and ellipticities, phase statistics, and (\Omega_2/\Omega_1). Radial and tangential forces are projections onto the current (\mathbf e_{r,i},\mathbf e_{\theta,i}). Replacing (\lambda_i\) by a scalar (\lambda_i(1+3Re/16)) in the old formulas does not perform this averaging.

For small (\mathbf V_i) and a symmetric RF-velocity distribution (\mathbf w\mapsto-\mathbf w),
\[
\mathcal G_i(\mathbf V)=\mathsf T_i\mathbf V+O(|V|^3/\langle|w|^2\rangle),\quad
\mathsf T_i=\left\langle |\mathbf w|I+\frac{\mathbf w\mathbf w^T}{|\mathbf w|}\right\rangle.
\]
The tensor (\mathsf T_i) is generally anisotropic. Only uniformly distributed directions in the plane give (\mathsf T_i=(3/2)\langle|w|\rangle I). When (V) is comparable to RF micromotion, use the full torus integral; the small-(V) expansion loses accuracy.

For a self-consistent first-order Oseen correction, first compute the linear response (\xi_i^{(0)}). Then form
\[
\mathbf g_{O,i}(\varphi_1,\varphi_2)=-\lambda_i\frac{3K_i}{16}
|\mathbf V_i+\mathbf w_i^{(0)}|(\mathbf V_i+\mathbf w_i^{(0)}),
\]
expand this nonlinear force in a torus Fourier series, and pass each mode (\mathbf g_{O,i}^{mn}) through the matrix susceptibility at frequency (m\Omega_1+n\Omega_2), including the DC Jacobian. Return the resulting micromotion correction to (\mathcal G_i). This is a computable perturbation scheme when (3Re_d/16\ll1) and (\epsilon_{RF}\ll1), not a fit parameter.

## 4. Coulomb coupling and relative phase

For a pair, average the Coulomb force over RF phases while keeping the slow relative phase:
\[
\overline{\mathbf F}_{C,i}(\rho_1,\rho_2,z_1,z_2,\Delta\theta;
\varphi_1,\varphi_2)=\frac{1}{(2\pi)^2}\int\!\int
\mathbf F_{C,i}(\mathbf R_i+\xi_i-\mathbf R_j-\xi_j)\,d\varphi_1d\varphi_2.
\]
Then integrate slow dynamics in (\Delta\theta); do not fix it a priori. For each particle in polar coordinates,
\[
\dot\rho_i=u_{r,i},\quad \dot\theta_i=u_{\theta,i}/\rho_i,
\quad \dot u_{r,i}=u_{\theta,i}^2/\rho_i+F_{r,i}^{eff},
\quad \dot u_{\theta,i}=-u_{r,i}u_{\theta,i}/\rho_i+F_{\theta,i}^{eff},
\]
where (F^{eff}) includes averaged RF, DC, Stokes/Oseen drag, gravity, and Coulomb. Under counterrotation, (\dot{\Delta\theta}=\omega_1-\omega_2\) is generally nonzero, so Coulomb torque varies in time even for nearly constant radii. Constant radii and frequencies are roots only for phase-locked states or asymptotic means of a quasiperiodic solution. The (\kappa=0) case is the required control for intrinsic single-particle rotation.

## 5. Unsteady Stokes drag: model C

For harmonic velocity (\hat{\mathbf v}e^{-i\omega t}) in an unbounded incompressible fluid, the linear Basset–Boussinesq–Oseen hydrodynamic impedance of a sphere is
\[
\widehat{\mathbf F}_h=-6\pi\eta a\left[1+a\sqrt{\frac{-i\omega}{\nu_f}}
-i\omega\frac{a^2}{9\nu_f}\right]\widehat{\mathbf v},
\]
using the square-root branch with positive real part. The terms are Stokes drag, Basset history, and fluid added mass (m_f/2=2\pi\rho_fa^3/3). The history magnitude is set by αₕ=ᵃ√(ω/νₑ); added mass relative to Stokes drag scales as αₕ²/9. With slow drift and two RF families, evaluate each response frequency σᵢₛ rather than using only the orbit frequency.

This linear frequency-domain result is a low-Re control model, not the complete finite-Re history force. A full time-domain solution requires a history convolution or a verified diffusive state-space approximation. Thus the Oseen ODE (model B) and model C answer different questions: B estimates quasi-steady nonlinear drag; C measures the scale of linear unsteadiness. If the C shift is comparable with the B correction, a 1–2% physical accuracy claim for model B is unsupported.

The Cunningham correction uses an explicit radius convention, `Kn=ℓ/a`, and
the Kim et al. correlation
`C_c=1+Kn[1.165+0.483 exp(−0.997/Kn)]`. It is retained as a separate
linear-Stokes control; it is not multiplied by the Oseen term as though that
product were a controlled composite closure. The specified spheres give
`C_c=1.00216690` and `1.00168537`, much smaller than the unsteady and field-model
uncertainties.

## 6. Stability and an algorithmic criterion

The Oseen drag Jacobian depends on instantaneous velocity:
\[
\frac{\partial\mathbf F_O}{\partial\mathbf u}=-\lambda\left[I+
\frac{3K}{16}\left(|u|I+\frac{uu^T}{|u|}\right)\right],\qquad u\ne0,
\]
and tends to (-\lambda I) as (u\to0). `numerical/two_particle_oseen.py` includes this matrix in the tangent Jacobian together with the RF Hessian and Coulomb blocks.

A quasiperiodic system has no Floquet multipliers unless a genuine period exists. An algorithmic criterion integrates (\dot\delta=J(t)\delta), QR-orthonormalizes on windows, and estimates the largest finite-time Lyapunov exponent. Finite perturbations are tested by changing initial radii, angles, velocities, and the two RF phases. Retention is classified separately from the signs of mean (\omega_i), phase organization, and contact avoidance.

## 7. Preliminary applicability for the specified pair

For 35/45 μm polystyrene spheres in air at 47.5/147.81 Hz, λ₁=0.2122, λ₂=0.1284, and α₂=0.34126, close to the Local15 Stokes values. However, the frozen Local15 moving-orbit micromotion screen gives diameter-based (Re_d) RMS near 0.365 and 0.333, with conservative phase bounds near 0.734 and 0.653. Thus the (3Re_d/16) term is not uniformly tiny, and some RF phases exceed 0.4, the range below which Maxworthy found the steady Oseen formula to match measured sphere drag accurately. The exact new trajectory must replace these provisional velocity estimates.

The unsteady parameter `α_h=a√(ω/ν_f)` at the first/second RF frequencies is 0.156/0.275 for sphere 1 and 0.200/0.353 for sphere 2. These are larger than the average Oseen correction and show that quasi-steady Oseen alone cannot establish 1–2% accuracy. Exact point-charge trajectories give `Re_d,rms≈0.367/0.333`, with sampled maxima `≈0.555/0.476` for the leading-harmonic and MFS fields; the induced-polarization sensitivity reaches `0.579/1.071`. Cunningham factors are close to unity; multiplying them by Oseen does not create a controlled composite theory.

## 9. Induced polarization: a physical sensitivity gate

For a homogeneous isotropic sphere in the linear quasistatic limit,
`α=4πε0a³(εr−1)/(εr+2)` and
`F_pol=(α/2)∇|E|²=α(H_Φ∇Φ)`, where `H_Φ` is the electric-potential Hessian.
The exact dimensionless RHS adds
`a_pol=[α m_1² Ω_1²/(q_1²m_i)] H_φ∇φ` for
`φ=q_1 Φ/(m_1R²Ω_1²)`. The implementation was checked against the gradient of
the MFS field energy; the trajectory force uses the instantaneous sum of both
RF families and the ideal DC quadrupole. With a constant real illustrative `εr=2.6`, the full-MFS
1000-period result changes: particle 2 stays clear of the rods but its mean
rotation approaches zero, and its mean radius shifts approximately
`0.766R→0.939R`. This shows that a point-charge model alone is insufficient
for a material-specific claim. It does not qualify real polystyrene: measured
complex `ε*(ω)`, dielectric loss, charge-polarization coupling, and mutual
induced-dipole forces remain needed. This run was not repeated for hollow glass
or Al₂O₃.

## 8. Questions not resolved by this chapter

MFS surface fields, RF-phase torus maxima, DC-electrode synthesis, discharge inception, charge leakage, contact mechanics, wall corrections, and finite-length axial fields require separate numerical or experimental checks. The ideal RF field omits higher spatial harmonics of the 24-rod geometry. Final radii and frequencies must be compared against new, independent control initial conditions without fitting the theory to them.
