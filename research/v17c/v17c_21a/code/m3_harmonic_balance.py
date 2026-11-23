"""Fourier--Galerkin M3 closure with safeguarded nonlinear solvers.

The legacy V21 implementation is left untouched.  This module reuses its
field convention and MFS data, but reports the nonlinear equation residual
separately from iterate change and never treats a capped iteration as solved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from scipy.signal import resample

from analytical_model import Mode, J, _phase_grid, _rf_field_rotating


class HarmonicBalanceFailure(RuntimeError):
    """Raised when a collocation closure is not converged to its requested tolerance."""


@dataclass
class HBConfig:
    ntheta: int
    nphase: int
    cutoff: float = 0.05
    oversample: int = 1
    method: str = "anderson"  # fixed_point | adaptive | anderson
    tolerance: float = 1e-5
    equation_tolerance: float = 5e-5
    max_iter: int = 240
    relaxation: float = 0.5
    history: int = 6
    sn_scale: float = 1.0
    continuation_scales: tuple[float, ...] = ()

    def validate(self) -> None:
        if self.method not in {"fixed_point", "adaptive", "anderson", "newton_krylov"}:
            raise ValueError(f"unsupported solver method: {self.method}")
        if min(self.ntheta, self.nphase, self.oversample) <= 0:
            raise ValueError("grid sizes and oversampling must be positive")
        if self.cutoff < 0 or not 0 < self.relaxation <= 1 or self.equation_tolerance <= 0:
            raise ValueError("invalid cutoff or relaxation")
        if not 0 <= self.sn_scale <= 1:
            raise ValueError("SN continuation scale must be in [0, 1]")
        if any(not 0 <= value <= 1 for value in self.continuation_scales):
            raise ValueError("continuation scales must lie in [0, 1]")
        if self.continuation_scales and abs(self.continuation_scales[-1] - 1.0) > 1e-14:
            raise ValueError("continuation scales must end at the physical SN scale 1")


def _resample3(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    out = values
    for axis, n in enumerate(shape):
        if out.shape[axis] != n:
            out = resample(out, n, axis=axis)
    return np.asarray(out.real, dtype=float)


def _fourier_coefficients(values: np.ndarray) -> np.ndarray:
    return np.fft.fftn(values, axes=(0, 1, 2)) / np.prod(values.shape[:3])


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.abs(values) ** 2, axis=-1))))


def _real_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values.ravel()) / np.sqrt(values.size))


def solve_harmonic_balance(
    r: float,
    omega: float,
    alpha: float,
    lam: float,
    re_per_speed: float,
    beta: float,
    modes: list[Mode],
    config: HBConfig,
    u_r: float = 0.0,
    v_z: float = 0.0,
    xi0: np.ndarray | None = None,
    require_convergence: bool = True,
) -> dict[str, Any]:
    """Solve one particle's M3 fast balance on a three-phase Fourier grid.

    The unknown is the retained displacement on the base Fourier grid.  For
    oversample > 1, nonlinear RF/drag products are evaluated on a padded
    periodic grid and projected back, separating quadrature aliasing from the
    chosen retained basis.  The linear fast operator is inverted mode by mode.
    """
    config.validate()
    if config.continuation_scales:
        current = None if xi0 is None else np.asarray(xi0, dtype=float)
        stage_rows = []
        final = None
        for scale in config.continuation_scales:
            stage_cfg = replace(config, sn_scale=float(scale), continuation_scales=())
            final = solve_harmonic_balance(
                r, omega, alpha, lam, re_per_speed, beta, modes, stage_cfg,
                u_r=u_r, v_z=v_z, xi0=current, require_convergence=False,
            )
            stage_rows.append({
                "sn_scale": float(scale), "status": final["status"],
                "iterations": final["iterations"],
                "relative_nonlinear_residual": final["relative_nonlinear_residual"],
                "relative_original_fast_equation_residual": final["relative_original_fast_equation_residual"],
            })
            if not final["converged"]:
                break
            current = final["xi_grid"]
        assert final is not None
        final["continuation_scales"] = list(config.continuation_scales)
        final["continuation_stages"] = stage_rows
        final["status"] = (
            "CONVERGED" if final["converged"] else "CONTINUATION_STAGE_FAILED"
        )
        if require_convergence and not final["converged"]:
            raise HarmonicBalanceFailure(
                f"M3 SN continuation failed at scale={stage_rows[-1]['sn_scale']:.3g}; "
                f"map residual={final['relative_nonlinear_residual']:.3e}, "
                f"original residual={final['relative_original_fast_equation_residual']:.3e}"
            )
        return final
    started = perf_counter()
    ntheta, nphase = config.ntheta, config.nphase
    base_shape = (ntheta, nphase, nphase)
    eval_shape = tuple(n * config.oversample for n in base_shape)
    theta, phases = _phase_grid(ntheta, nphase)
    theta_eval, phases_eval = _phase_grid(eval_shape[0], eval_shape[1])

    ktheta = np.fft.fftfreq(ntheta) * ntheta
    kphase = np.fft.fftfreq(nphase) * nphase
    kt, kp1, kp2 = np.meshgrid(ktheta, kphase, kphase, indexing="ij")
    freq1 = next(m.frequency for m in modes if m.family == 0)
    freq2 = next(m.frequency for m in modes if m.family == 1)
    # A real even-grid Fourier series has a self-conjugate Nyquist index in
    # each coordinate. Its derivative contribution must be zero; otherwise
    # kappa(-k) != -kappa(k) on Nyquist planes and the real inverse FFT silently
    # projects away a nonzero imaginary part.
    kt_eff = np.where(kt == -ntheta // 2, 0.0, kt)
    kp1_eff = np.where(kp1 == -nphase // 2, 0.0, kp1)
    kp2_eff = np.where(kp2 == -nphase // 2, 0.0, kp2)
    deriv_freq = kt_eff * omega + kp1_eff * freq1 + kp2_eff * freq2
    fast_mask = ((kp1 != 0) | (kp2 != 0)) & (np.abs(deriv_freq) >= config.cutoff)
    slow_mask = ~fast_mask
    d0 = -deriv_freq**2 + 1j * deriv_freq * lam - omega**2 - alpha * beta
    dJ = 2j * omega * deriv_freq + lam * omega
    den = d0 * d0 + dJ * dJ
    if not np.any(fast_mask):
        raise ValueError("fast-mode mask is empty")
    min_den = float(np.min(np.abs(den[fast_mask])))
    if not np.isfinite(min_den) or min_den < 1e-14:
        raise ArithmeticError(f"singular fast Fourier operator; min |det|={min_den:.3e}")

    if xi0 is None:
        xi = np.zeros(base_shape + (2,), dtype=float)
    else:
        if tuple(xi0.shape[:3]) != base_shape or xi0.shape[-1] != 2:
            xi = _resample3(np.asarray(xi0, dtype=float), base_shape)
        else:
            xi = np.asarray(xi0, dtype=float).copy()

    def map_and_diagnostics(x: np.ndarray, full_metrics: bool = False):
        xhat = _fourier_coefficients(x)
        xdot_hat = 1j * deriv_freq[..., None] * xhat
        xdot_base = np.fft.ifftn(xdot_hat * np.prod(base_shape), axes=(0, 1, 2)).real
        x_eval = x if eval_shape == base_shape else _resample3(x, eval_shape)
        xdot_eval = xdot_base if eval_shape == base_shape else _resample3(xdot_base, eval_shape)

        # Frozen slow velocity in the rotating frame.  The constant radial and
        # axial components enter the nonlinear drag magnitude; their linear
        # Stokes contributions belong to the slow flow.
        v = np.zeros(eval_shape + (3,), dtype=float)
        v[..., 0] = u_r + xdot_eval[..., 0] - omega * x_eval[..., 1]
        v[..., 1] = omega * r + xdot_eval[..., 1] + omega * x_eval[..., 0]
        v[..., 2] = v_z
        speed = np.linalg.norm(v, axis=-1)
        w = (r + x_eval[..., 0]) + 1j * x_eval[..., 1]
        rf_complex = _rf_field_rotating(w, theta_eval, phases_eval, alpha, modes)
        rf = np.stack((rf_complex.real, rf_complex.imag), axis=-1)
        sn = (-0.15 * config.sn_scale * lam *
              (re_per_speed * speed) ** 0.687)[..., None] * v[..., :2]
        forcing_eval = rf + sn
        forcing_base = forcing_eval if eval_shape == base_shape else _resample3(forcing_eval, base_shape)
        force_hat = _fourier_coefficients(forcing_base)
        target_hat = np.empty_like(force_hat)
        fx, fy = force_hat[..., 0], force_hat[..., 1]
        target_hat[..., 0] = (d0 * fx + dJ * fy) / den
        target_hat[..., 1] = (-dJ * fx + d0 * fy) / den
        target_hat[slow_mask, :] = 0.0
        target = np.fft.ifftn(target_hat * np.prod(base_shape), axes=(0, 1, 2)).real
        op0 = d0 * xhat[..., 0] - dJ * xhat[..., 1]
        op1 = dJ * xhat[..., 0] + d0 * xhat[..., 1]
        equation_hat = np.stack((op0 - force_hat[..., 0], op1 - force_hat[..., 1]), axis=-1)
        fast_eq_rms = float(np.sqrt(np.sum(np.abs(equation_hat[fast_mask]) ** 2)))
        fast_force_rms = float(np.sqrt(np.sum(np.abs(force_hat[fast_mask]) ** 2)))
        relative_equation_residual = fast_eq_rms / max(fast_force_rms, 1e-12)
        if not full_metrics:
            return target, force_hat, relative_equation_residual

        # Reconstruct full drag and z-drag on the padded quadrature grid.
        v3 = np.zeros(eval_shape + (3,), dtype=float)
        v3[..., :2] = v[..., :2]
        v3[..., 2] = v_z
        drag = -lam * (1.0 + 0.15 * (re_per_speed * speed) ** 0.687)[..., None] * v3
        position = np.zeros_like(v3)
        position[..., 0] = r + x_eval[..., 0]
        position[..., 1] = x_eval[..., 1]
        torque_rf = position[..., 0] * rf[..., 1] - position[..., 1] * rf[..., 0]
        torque_drag = position[..., 0] * drag[..., 1] - position[..., 1] * drag[..., 0]
        angular_momentum = position[..., 0] * v3[..., 1] - position[..., 1] * v3[..., 0]

        # Residual of the original collocation equation, separately projected
        # onto fast and excluded slow/near-resonant modes.
        fast_eq = equation_hat[fast_mask]
        slow_eq = equation_hat[slow_mask].copy()
        # The zero/mean force is assigned to slow balance, not to displacement.
        zero_is_slow_row = int(np.flatnonzero(slow_mask.ravel() == True)[0]) if slow_mask[0, 0, 0] else -1
        if zero_is_slow_row >= 0:
            slow_eq[zero_is_slow_row] = 0.0
        rms_fast_eq = float(np.sqrt(np.sum(np.abs(fast_eq) ** 2)))
        rms_slow_eq = float(np.sqrt(np.sum(np.abs(slow_eq) ** 2)))

        # Nonlinear-force tail omitted by the retained base grid.  This is an
        # aliasing indicator, not an error bound for the irrational-power law.
        if eval_shape == base_shape:
            # No oversampled estimate was made; zero would incorrectly claim
            # that the nonlinear spectrum has no omitted high-frequency tail.
            tail_rms = None
        else:
            reconstructed = _resample3(forcing_base, eval_shape)
            tail_rms = _rms(forcing_eval - reconstructed)
        return target, force_hat, {
            "mean_rf_force_rotating_xy": np.mean(rf, axis=(0, 1, 2)).tolist(),
            "mean_drag_force_rotating_xyz": np.mean(drag, axis=(0, 1, 2)).tolist(),
            "mean_total_RF_torque": float(np.mean(torque_rf)),
            "mean_total_drag_torque": float(np.mean(torque_drag)),
            "mean_angular_momentum": float(np.mean(angular_momentum)),
            "mean_speed": float(np.mean(speed)),
            "rms_micromotion": float(np.sqrt(np.mean(np.sum(x_eval * x_eval, axis=-1)))),
            "max_micromotion": float(np.max(np.linalg.norm(x_eval, axis=-1))),
            "max_speed": float(np.max(speed)),
            "mean_Re_estimate": float(re_per_speed * np.mean(speed)),
            "max_Re_estimate": float(re_per_speed * np.max(speed)),
            "rms_rf_force": _rms(rf),
            "rms_drag_force": _rms(drag),
            "rms_unresolved_fast_force": rms_fast_eq,
            "relative_original_fast_equation_residual": relative_equation_residual,
            "equation_tolerance": config.equation_tolerance,
            "rms_unresolved_slow_force": rms_slow_eq,
            "rms_nonlinear_force_tail": tail_rms,
            "relative_nonlinear_force_tail": (
                None if tail_rms is None else tail_rms / max(_rms(forcing_eval), 1e-30)
            ),
            "minimum_fast_mode_denominator": min_den,
            "number_of_fast_modes": int(np.count_nonzero(fast_mask)),
            "number_of_modes_with_any_nyquist_index": int(
                np.count_nonzero((kt == -ntheta // 2) | (kp1 == -nphase // 2) | (kp2 == -nphase // 2))
            ),
            "nyquist_derivative_convention": "zero the derivative contribution along each self-conjugate even-grid Nyquist coordinate",
            "number_of_excluded_rf_combination_modes": int(
                np.count_nonzero(slow_mask & ((kp1 != 0) | (kp2 != 0)))
            ),
            "fast_modes_count_by_RF_phase_order": {
                "p1_nonzero": int(np.count_nonzero(fast_mask & (kp1 != 0))),
                "p2_nonzero": int(np.count_nonzero(fast_mask & (kp2 != 0))),
                "both_nonzero": int(np.count_nonzero(fast_mask & (kp1 != 0) & (kp2 != 0))),
            },
            "excluded_rf_combination_modes": [
                [int(a), int(b), int(c), float(w), float(abs(z))]
                for a, b, c, w, z in zip(
                    kt[slow_mask & ((kp1 != 0) | (kp2 != 0))].ravel(),
                    kp1[slow_mask & ((kp1 != 0) | (kp2 != 0))].ravel(),
                    kp2[slow_mask & ((kp1 != 0) | (kp2 != 0))].ravel(),
                    deriv_freq[slow_mask & ((kp1 != 0) | (kp2 != 0))].ravel(),
                    den[slow_mask & ((kp1 != 0) | (kp2 != 0))].ravel(),
                )
            ],
            "phase_grid": list(base_shape),
            "evaluation_grid": list(eval_shape),
            "cutoff": config.cutoff,
            "solver": config.method,
        }

    # Secant/Anderson history for Type-II Anderson acceleration.
    x_hist: list[np.ndarray] = []
    g_hist: list[np.ndarray] = []
    relaxation = config.relaxation
    previous_g_norm = None
    smooth_decreases = 0
    trace = []
    converged = False
    nonfinite_divergence = False
    last_change = float("inf")
    iteration = 0
    if config.method == "newton_krylov":
        from scipy.optimize import NoConvergence, newton_krylov

        callback_points: list[np.ndarray] = []
        scale = max(float(np.max(np.abs(xi))), 1e-8)

        def fixed_map_residual(flat):
            candidate = np.asarray(flat).reshape(base_shape + (2,))
            target, _, _ = map_and_diagnostics(candidate)
            local_scale = max(float(np.max(np.abs(target))), scale, 1e-12)
            return ((target - candidate) / local_scale).ravel()

        def on_newton_step(candidate, residual):
            nonlocal iteration, last_change
            candidate = np.asarray(candidate)
            if callback_points:
                last_change = float(np.linalg.norm(candidate - callback_points[-1]) /
                                    np.sqrt(candidate.size))
            callback_points.append(candidate.copy())
            iteration += 1
            trace.append({
                "iteration": iteration,
                "relative_nonlinear_residual": float(np.max(np.abs(residual))),
                "residual_l2": float(np.linalg.norm(residual)),
                "method": "newton_krylov",
            })

        try:
            solution = newton_krylov(
                fixed_map_residual, xi.ravel(), method="lgmres",
                maxiter=config.max_iter, inner_maxiter=24,
                # scipy's Krylov termination uses the L2 norm; the HB acceptance
                # uses a max-norm relative residual.  A small factor makes the
                # internal stopping rule stricter and final acceptance remains
                # independently checked below.
                f_tol=config.tolerance * np.sqrt(xi.size) * 1e-3,
                line_search="armijo", callback=on_newton_step,
            )
            xi = np.asarray(solution).reshape(base_shape + (2,))
            converged = True
        except NoConvergence as exc:
            xi = np.asarray(exc.args[0]).reshape(base_shape + (2,))
            converged = False
        iteration = max(iteration, 1)
        iteration_range = ()
    else:
        iteration_range = range(1, config.max_iter + 1)
    last_finite_xi = xi.copy()
    for iteration in iteration_range:
        target, _, relative_equation_residual = map_and_diagnostics(xi)
        if not np.all(np.isfinite(target)) or not np.isfinite(relative_equation_residual):
            nonfinite_divergence = True
            xi = last_finite_xi
            trace.append({"iteration": iteration, "status": "NONFINITE_MAP"})
            break
        last_finite_xi = xi.copy()
        g = target - xi
        denom_norm = max(float(np.max(np.abs(target))), 1e-12)
        nonlinear_residual = float(np.max(np.abs(g)) / denom_norm)
        gnorm = _real_norm(g)
        if iteration == 1 or iteration % 5 == 0 or nonlinear_residual < config.tolerance:
            trace.append({"iteration": iteration, "relative_nonlinear_residual": nonlinear_residual,
                          "rms_map_residual": gnorm, "relaxation": relaxation})
        if (nonlinear_residual <= config.tolerance and
                relative_equation_residual <= config.equation_tolerance):
            converged = True
            last_change = _real_norm(g)
            break

        if config.method == "fixed_point":
            step = config.relaxation * g
        elif config.method == "adaptive":
            if previous_g_norm is not None:
                ratio = gnorm / max(previous_g_norm, 1e-30)
                if ratio > 1.03:
                    relaxation = max(0.05, relaxation * 0.5)
                    smooth_decreases = 0
                elif ratio < 0.72:
                    smooth_decreases += 1
                    if smooth_decreases >= 3:
                        relaxation = min(0.85, relaxation * 1.15)
                        smooth_decreases = 0
            step = relaxation * g
            previous_g_norm = gnorm
        else:
            flat_x, flat_g = xi.ravel(), g.ravel()
            x_hist.append(flat_x.copy())
            g_hist.append(flat_g.copy())
            if len(x_hist) > config.history + 1:
                x_hist.pop(0)
                g_hist.pop(0)
            depth = len(x_hist) - 1
            if depth == 0:
                step = config.relaxation * g
            else:
                dx = np.column_stack([x_hist[j + 1] - x_hist[j] for j in range(depth)])
                dg = np.column_stack([g_hist[j + 1] - g_hist[j] for j in range(depth)])
                try:
                    gamma, *_ = np.linalg.lstsq(dg, flat_g, rcond=1e-12)
                    accelerated = flat_g * config.relaxation - (dx + config.relaxation * dg) @ gamma
                    step = accelerated.reshape(xi.shape)
                    # Reject extreme extrapolation; the relaxed Picard step is
                    # the stable fallback and history is reset.
                    if not np.all(np.isfinite(step)) or np.max(np.abs(step)) > 4.0 * max(np.max(np.abs(g)), 1e-14):
                        step = config.relaxation * g
                        x_hist.clear()
                        g_hist.clear()
                except np.linalg.LinAlgError:
                    step = config.relaxation * g
                    x_hist.clear()
                    g_hist.clear()
        xi_next = xi + step
        if not np.all(np.isfinite(xi_next)) or np.max(np.abs(xi_next)) > 2.0:
            nonfinite_divergence = True
            xi = last_finite_xi
            trace.append({"iteration": iteration, "status": "DIVERGENT_UPDATE"})
            break
        last_change = _real_norm(xi_next - xi)
        xi = xi_next

    # Always recompute diagnostics from the returned iterate; no stale trial
    # state is allowed to leak from the outer least-squares evaluation.
    target, force_hat, metrics = map_and_diagnostics(xi, full_metrics=True)
    g = target - xi
    nonlinear_residual = float(np.max(np.abs(g)) / max(float(np.max(np.abs(target))), 1e-12))
    converged = bool(
        nonlinear_residual <= config.tolerance and
        metrics["relative_original_fast_equation_residual"] <= config.equation_tolerance
    )
    elapsed = perf_counter() - started
    result = {
        **metrics,
        "converged": converged,
        "status": ("CONVERGED" if converged else
                   "NONFINITE_OR_DIVERGENT" if nonfinite_divergence else
                   "ITERATION_LIMIT_OR_STALL"),
        "iterations": iteration,
        "wall_time_s": elapsed,
        "relative_nonlinear_residual": nonlinear_residual,
        "relative_original_fast_equation_residual": metrics["relative_original_fast_equation_residual"],
        "equation_tolerance": config.equation_tolerance,
        "fixed_point_change_rms": last_change,
        "tolerance": config.tolerance,
        "max_iter": config.max_iter,
        "relaxation_initial": config.relaxation,
        "anderson_history": config.history if config.method == "anderson" else 0,
        "residual_trace": trace,
        "xi_grid": xi,
        "xi_fourier_coefficients": _fourier_coefficients(xi),
        "nonlinear_force_fourier_coefficients": force_hat,
    }
    if require_convergence and not converged:
        raise HarmonicBalanceFailure(
            f"M3 inner solve failed: method={config.method}, grid={base_shape}, "
            f"cutoff={config.cutoff}, residual={nonlinear_residual:.3e}, "
            f"iterations={iteration}/{config.max_iter}"
        )
    return result
