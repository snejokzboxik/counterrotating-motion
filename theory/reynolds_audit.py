"""Reynolds and Schiller--Naumann audit on the converged M3 response grid."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21/code"))
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21a/code"))
from analytical_model import _phase_grid, load_inputs  # noqa: E402
from m3_harmonic_balance import HBConfig, solve_harmonic_balance  # noqa: E402


def derivative_frequencies(ntheta: int, nphase: int, omega: float,
                           nu: float) -> np.ndarray:
    """Match M3's three-angle derivative, including real-grid Nyquist rules."""
    ktheta = np.fft.fftfreq(ntheta) * ntheta
    kp = np.fft.fftfreq(nphase) * nphase
    kt, kp1, kp2 = np.meshgrid(ktheta, kp, kp, indexing="ij")
    kt_eff = np.where(kt == -ntheta // 2, 0.0, kt)
    kp1_eff = np.where(kp1 == -nphase // 2, 0.0, kp1)
    kp2_eff = np.where(kp2 == -nphase // 2, 0.0, kp2)
    return kt_eff * omega + kp1_eff + nu * kp2_eff


def evaluate() -> dict:
    data = load_inputs()
    archive = json.loads((ROOT / "research/v17c/v17c_21a/results/full_grid_campaign.json").read_text())
    stage = archive["stages"][-1]
    if stage["grid"] != [192, 16, 16] or stage["status"] != "VERIFIED_ROOT":
        raise RuntimeError("Expected published verified M3 nominal root")
    cfg = HBConfig(**stage["config"])
    theta, phases = _phase_grid(cfg.ntheta, cfg.nphase)
    root = stage["root"]
    particles = []
    for i in range(2):
        r = root["radii_R"][i]
        omega = root["omega_over_Omega1"][i]
        alpha, lam, re_star = data["alpha"][i], data["lambda"][i], data["Re_per_speed"][i]
        hb = solve_harmonic_balance(r, omega, alpha, lam, re_star, data["beta"],
                                    data["modes"], cfg, require_convergence=True)
        xi = hb["xi_grid"]
        xhat = np.fft.fftn(xi, axes=(0, 1, 2)) / np.prod(xi.shape[:3])
        # The Fourier derivative frequencies are the same as in the published M3 solver.
        freq2 = next(m.frequency for m in data["modes"] if m.family == 1)
        deriv = derivative_frequencies(cfg.ntheta, cfg.nphase, omega, freq2)
        xdot_hat = 1j * deriv[..., None] * xhat
        xdot = np.fft.ifftn(xdot_hat * np.prod(xi.shape[:3]), axes=(0, 1, 2)).real
        # Frozen circular slow orbit plus fast response, rotated coordinates preserve |v|.
        vx = xdot[..., 0] - omega * xi[..., 1]
        vy = omega * r + xdot[..., 1] + omega * xi[..., 0]
        speed = np.sqrt(vx * vx + vy * vy)
        re = re_star * speed
        factor = 1 + 0.15 * re**0.687
        x = r + xi[..., 0]
        y = xi[..., 1]
        tau_lin = -lam * (x * vy - y * vx)
        tau_sn = factor * tau_lin
        force_lin = lam * speed
        force_sn = factor * force_lin
        particles.append({
            "particle": i + 1, "Re_star": re_star,
            "M3_grid": list(re.shape), "M3_inner_residual": hb["relative_original_fast_equation_residual"],
            "Re_statistics": {"min": float(np.min(re)), "p05": float(np.percentile(re, 5)),
                               "median": float(np.median(re)), "mean": float(np.mean(re)),
                               "p95": float(np.percentile(re, 95)), "max": float(np.max(re))},
            "SN_relative_drag_factor_statistics": {
                "min": float(np.min(factor)), "median": float(np.median(factor)),
                "mean": float(np.mean(factor)), "p95": float(np.percentile(factor, 95)),
                "max": float(np.max(factor))},
            "mean_drag_force_magnitude_linear": float(np.mean(force_lin)),
            "mean_drag_force_magnitude_SN": float(np.mean(force_sn)),
            "mean_SN_increment_over_linear_force": float(np.mean(force_sn - force_lin)),
            "mean_drag_torque_linear_at_M3_state": float(np.mean(tau_lin)),
            "mean_drag_torque_SN_at_M3_state": float(np.mean(tau_sn)),
            "mean_SN_increment_over_linear_torque": float(np.mean(tau_sn - tau_lin)),
            "caveat": "Frozen-orbit M3 collocation statistics include modeled RF micromotion, not independent full-RF sampling; drag comparison holds the M3 state fixed and is not a frequency re-solve."})
    return {
        "source": "V21A verified nominal M3 root; 192x16x16 Fourier collocation",
        "root": root, "model": "Cunningham-corrected Stokes times Schiller-Naumann as used in V20B/V21",
        "particles": particles,
        "interpretation": "Re_* cannot be removed from full physical dynamics by coordinate scaling. These statistics quantify the nonlinear correction on the M3 fast-response grid; they do not establish transient-flow validity or replace RF trajectory data."}


if __name__ == "__main__":
    out = ROOT / "data/reynolds_m3_grid_audit.json"
    out.write_text(json.dumps(evaluate(), indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
