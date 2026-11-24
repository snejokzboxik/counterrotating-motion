"""Finite-difference verification of the full RF mechanical Jacobian."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from full_rf_short_perturbation import rhs


def jacobian(t, y, phase, p, field, qratio, re1, re2, gz):
    J = np.zeros((12, 12))
    for i in range(3):
        J[i, i + 3] = 1.0
        J[i + 6, i + 9] = 1.0
    A, B, nu, beta, s, C, lam1, lam2 = map(float, p)
    c1 = A * np.exp(-1j * (t + phase[0]))
    c2 = B * np.exp(-1j * (nu * t + phase[1]))
    for part, j in ((0, 0), (1, 6)):
        pos = complex(y[j], y[j + 1])
        qom = 1.0 if part == 0 else s * qratio
        hxx = hxy = hyy = 0.0
        for fam, mode, coef in zip(field["family"], field["k"], field["coeff"]):
            degree = abs(int(mode)); base = pos if mode > 0 else pos.conjugate()
            carrier = c1 if fam == 0 else c2
            hess = carrier * coef * degree * (degree - 1) * base ** (degree - 2)
            if mode > 0:
                hxx -= hess.real; hxy += hess.imag; hyy += hess.real
            else:
                hxx -= hess.real; hxy -= hess.imag; hyy += hess.real
        J[j + 3, j] += qom * (hxx + beta); J[j + 3, j + 1] += qom * hxy
        J[j + 4, j] += qom * hxy; J[j + 4, j + 1] += qom * (hyy + beta)
        J[j + 5, j + 2] += qom * (-2.0 * beta)
        v = y[j + 3:j + 6]
        speed = float(np.linalg.norm(v)); lam = lam1 if part == 0 else lam2
        chi = re1 if part == 0 else re2
        fac = 1.0 + 0.15 * (chi * speed) ** 0.687
        coeff_v = 0.0 if speed <= 1e-30 else 0.15 * 0.687 * chi ** 0.687 * speed ** (-1.313)
        J[j + 3:j + 6, j + 3:j + 6] -= lam * (fac * np.eye(3) + coeff_v * np.outer(v, v))
    d = y[:3] - y[6:9]; d2 = float(d @ d)
    if C != 0.0 and d2 > 1e-28:
        H = C * (np.eye(3) / (d2 * np.sqrt(d2)) - 3.0 * np.outer(d, d) / (d2 ** 2.5))
        J[3:6, 0:3] += H; J[3:6, 6:9] -= H
        J[9:12, 0:3] -= s * H; J[9:12, 6:9] += s * H
    return J


def main():
    pth = ROOT / "research/v17c/v17c_20b/results/candidate_parameters.json"
    fieldpth = ROOT / "research/v17c/v17c_12b/model/field_model.json"
    cfg = json.loads(pth.read_text(encoding="utf-8"))
    c = next(x for x in cfg["candidates"] if x["legacy_candidate"] == "refine12_sobol17")
    fd = json.loads(fieldpth.read_text(encoding="utf-8"))
    field = {"family": np.asarray(fd["family"], int), "k": np.asarray(fd["k"], int),
             "coeff": np.asarray(fd["coeff"], float)[:, 0] + 1j * np.asarray(fd["coeff"], float)[:, 1]}
    p = np.asarray(c["p_corrected"], float); q = float(c["qratio"])
    tr = np.load(ROOT / "research/v17c/v17c_20b/results/refine12_240k_corrected_drag_trajectory.npz")
    y0 = np.asarray(tr["states"][-1], float)
    rng = np.random.default_rng(20261124); errors = []
    for k in range(12):
        y = y0.copy()
        y[:3] += rng.normal(0, 2e-3, 3); y[6:9] += rng.normal(0, 2e-3, 3)
        y[3:6] += rng.normal(0, 1e-3, 3); y[9:12] += rng.normal(0, 1e-3, 3)
        ph = np.array([2.3066744441521223, 2.515823851502823])
        t = 0.371 + k * 0.137
        J = jacobian(t, y, ph, p, field, q, c["re1_per_speed"], c["re2_per_speed"], c["gravity_z"])
        Jfd = np.zeros((12, 12))
        for col in range(12):
            h = 2e-7 if col in [3, 4, 5, 9, 10, 11] else 2e-6
            yp = y.copy(); ym = y.copy(); yp[col] += h; ym[col] -= h
            Jfd[:, col] = (rhs(t, yp, ph, p, field, q, c["re1_per_speed"], c["re2_per_speed"], c["gravity_z"])
                           - rhs(t, ym, ph, p, field, q, c["re1_per_speed"], c["re2_per_speed"], c["gravity_z"])) / (2 * h)
        errors.append(float(np.max(np.abs(J - Jfd)) / max(1.0, np.max(np.abs(Jfd)))))
    result = {"test": "full RF 12D state Jacobian vs central differences of independent NumPy RHS",
              "seed": 20261124, "samples": len(errors), "finite_difference_steps": {"position": 2e-6, "velocity": 2e-7},
              "max_scaled_error": max(errors), "errors": errors,
              "expected": "Analytic MFS Hessian, Coulomb derivative, and Schiller-Naumann velocity derivative agree with the RHS finite differences.",
              "pass": bool(max(errors) < 2e-7)}
    if not result["pass"]:
        raise SystemExit(json.dumps(result, indent=2))
    out = ROOT / "research/v17c/v17c_22/results/full_rf_jacobian_fd_test.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
