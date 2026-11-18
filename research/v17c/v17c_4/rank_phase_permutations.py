"""Audit and rank the 120 repeated-six-phase permutations from the archive."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ARCH = HERE / "input" / "archive_contents" / "v17c_six_phase"
OUT = HERE / "results" / "phase_optimization"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def torque(coeff: dict[int, complex], r: float) -> float:
    return float(sum(np.sign(k) * abs(k)**2 * (abs(k)-1) * abs(c)**2 * r**(2*abs(k)-2)
                     for k, c in coeff.items()))


def chirality(coeff: dict[int, complex], r: float) -> float:
    terms = [np.sign(k) * abs(k)**2 * (abs(k)-1) * abs(c)**2 * r**(2*abs(k)-2)
             for k, c in coeff.items()]
    return float(sum(terms) / max(sum(abs(x) for x in terms), 1e-300))


def roots(coeff: dict[int, complex]):
    grid = np.linspace(.03, .98, 2400)
    val = [torque(coeff, x) for x in grid]
    return [float(brentq(lambda r: torque(coeff, r), x, y))
            for x, y, a, b in zip(grid[:-1], grid[1:], val[:-1], val[1:]) if a*b < 0]


def field_profile(coeff: dict[int, complex], r: float, angles: np.ndarray):
    z = r * np.exp(1j*angles)
    fx = np.zeros_like(z, dtype=np.complex128)
    fy = np.zeros_like(z, dtype=np.complex128)
    for k, c in coeff.items():
        n = abs(k)
        base = z if k > 0 else z.conjugate()
        f = -n * c * base**(n-1)
        fx += f
        fy += (-1j*f) if k > 0 else (1j*f)
    power = np.abs(fx)**2 + np.abs(fy)**2
    return power


def angular_metrics(coeff: dict[int, complex], r: float):
    angles = np.arange(1440) * 2*np.pi/1440
    p = field_profile(coeff, r, angles)
    return {"mean_E2": float(np.mean(p)),
            "mean_pseudopotential_per_alpha2": float(np.mean(p)/4),
            "corrugation": float((np.max(p)-np.min(p))/np.mean(p)),
            "min_over_mean": float(np.min(p)/np.mean(p)),
            "barrier_per_alpha2": float((np.max(p)-np.min(p))/4)}


def surface_field_basis(N=12, a=.05, B=1.1, p=64, samples=128):
    theta = np.arange(N)*2*np.pi/N
    centers = np.exp(1j*theta)
    phi = theta[:, None] + np.arange(p)*2*np.pi/p
    z = (centers[:, None]+a*np.exp(1j*phi)).ravel()
    src = (centers[:, None]+.65*a*np.exp(1j*phi)).ravel()
    def green(target):
        return np.log(np.abs(B*B-target[:, None]*src[None, :].conj()) /
                      (B*np.abs(target[:, None]-src[None, :])))
    lu = lu_factor(green(z))
    basis_rhs = np.repeat(np.eye(N), p, axis=0)
    source_coeffs = lu_solve(lu, basis_rhs)
    samp = np.arange(samples)*2*np.pi/samples
    target = (centers[:, None]+a*np.exp(1j*(theta[:, None]+samp[None, :] - theta[:, None]))).ravel()
    # Equivalent, simpler exact electrode angles: electrode angle + sample offset.
    target = (centers[:, None]+a*np.exp(1j*(theta[:, None]+samp[None, :] - theta[:, None]))).ravel()
    sbar = src.conj()[None, :]
    den1 = B*B-target[:, None]*sbar
    diff = target[:, None]-src[None, :]
    gx = np.real(-sbar/den1-1/diff)
    gy = np.real(-1j*sbar/den1-1j/diff)
    ex_basis = -(gx @ source_coeffs)
    ey_basis = -(gy @ source_coeffs)
    return ex_basis, ey_basis


def main():
    K, p, N, a, B = 72, 64, 12, .05, 1.10
    scan_path = ARCH / "scan.py"
    spec = importlib.util.spec_from_file_location("six_phase_scan", scan_path)
    scan = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(scan)
    k, transfer, boundary_matrix_error = scan.transfer(N, a, B, p=p, K=K)
    original = read_csv(ARCH / "permutations.csv")
    phase_index = {c: i for i, c in enumerate("ABCDEF")}
    phase_table = np.asarray([[phase_index[c] for c in row["sequence"]] for row in original])
    voltage = np.exp(1j * phase_table * np.pi/3.0).T
    coeff_matrix = transfer @ voltage
    ex_basis, ey_basis = surface_field_basis(N=N, a=a, B=B, p=p)
    ex = ex_basis @ voltage
    ey = ey_basis @ voltage
    surface_global = np.max(np.sqrt(np.abs(ex)**2+np.abs(ey)**2), axis=0)
    adjacent = np.max(np.abs(np.roll(voltage, -1, axis=0)-voltage), axis=0)
    rows = []
    for j, source in enumerate(original):
        c = {int(mode): complex(value) for mode, value in zip(k, coeff_matrix[:, j])}
        rts = roots(c)
        low = angular_metrics(c, .30)
        band = angular_metrics(c, .55)
        inner_chi, band_chi = chirality(c, .30), chirality(c, .55)
        negative_width = 0.0
        if len(rts) >= 2:
            negative_width = max((rts[ii+1]-rts[ii] for ii in range(len(rts)-1)
                                  if torque(c, .5*(rts[ii+1]+rts[ii])) < 0), default=0.0)
        rows.append({
            "sequence": source["sequence"], "chi_03": inner_chi, "chi_055": band_chi,
            "chi_075": chirality(c, .75), "torque_roots": ";".join(map(str, rts)),
            "negative_band_width": negative_width,
            "field_mean_E2_r03": low["mean_E2"], "field_mean_E2_r055": band["mean_E2"],
            "angular_corrugation_r03": low["corrugation"],
            "angular_corrugation_r055": band["corrugation"],
            "angular_barrier_per_alpha2_r03": low["barrier_per_alpha2"],
            "angular_barrier_per_alpha2_r055": band["barrier_per_alpha2"],
            "surface_Emax_U0_over_D": float(surface_global[j]),
            "max_adjacent_voltage_difference_U0": float(adjacent[j]),
            "MFS_boundary_residual_p64": float(np.max(np.abs(boundary_matrix_error @ voltage[:, j]))),
            "coefficients": [[int(m), float(v.real), float(v.imag)] for m, v in c.items()],
        })
    rows.sort(key=lambda x: (min(x["chi_03"], -x["chi_055"]), x["negative_band_width"],
                             -x["angular_corrugation_r055"]), reverse=True)
    with (OUT/"repeated_six_phase_permutation_rank.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [x for x in rows[0] if x != "coefficients"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "coefficients"})
    (OUT/"repeated_six_phase_spectra_p64.json").write_text(
        json.dumps({r["sequence"]: r["coefficients"] for r in rows}, indent=2), encoding="utf-8")
    good = [r for r in rows if r["chi_03"] > .1 and r["chi_055"] < -.1]
    strong = [r for r in rows if r["chi_03"] > .3 and r["chi_075"] < -.3]
    summary = {
        "permutations": len(rows), "electrode_pattern_scope": "six phase permutation repeated twice; first phase A fixed modulo global RF time shift",
        "independent_MFS": {"p": p, "K": K, "max_boundary_residual": float(max(r["MFS_boundary_residual_p64"] for r in rows))},
        "criteria_positive_r03_negative_r055_abs_gt_0p1": len(good),
        "criteria_positive_r03_gt_0p3_negative_r075_lt_minus0p3": len(strong),
        "top_12": [{k: r[k] for k in r if k != "coefficients"} for r in rows[:12]],
    }
    (OUT/"repeated_six_phase_permutation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
