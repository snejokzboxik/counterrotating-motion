"""Compute A9 electrode-voltage amplitudes at the 250 V peak cap."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M0_TH = 3200.04652261404
AMU = 1.66053906660e-27
Q = 1.602176634e-19
R = 8e-3
OMEGA = 2 * math.pi * 100_000
C2 = D2 = 0.7718255558274212
C4 = 0.5413395699439091
ALPHA, BETA, KAPPA = 0.00625, 0.075, -0.015
V_LIMIT = 250.0


def peak_by_electrode(model: str, scale: float, delta: float = 0.0):
    voltage_scale = M0_TH * AMU * OMEGA**2 * R**2 / Q
    u2 = BETA * voltage_scale / C2
    dc = ALPHA * voltage_scale / D2
    u4 = KAPPA * scale * voltage_scale / C4
    result = []
    for j in range(12):
        theta = math.pi * j / 6
        q2 = math.cos(2 * theta)
        c4, s4 = math.cos(4 * theta), math.sin(4 * theta)
        if model == "Q":
            a, b = u2 * q2, 0.0
        elif model == "SO":
            a, b = u2 * q2 + u4 * c4 * math.cos(delta), u4 * c4 * math.sin(delta)
        elif model in ("RO_PLUS", "RO_MINUS"):
            sigma = 1 if model == "RO_PLUS" else -1
            a = u2*q2 + u4*(c4*math.cos(delta) - sigma*s4*math.sin(delta))
            b = u4*(c4*math.sin(delta) + sigma*s4*math.cos(delta))
        else:
            raise ValueError(model)
        peak = abs(dc * q2) + math.hypot(a, b)
        result.append({"electrode_index": j, "theta_deg": 30*j, "peak_abs_voltage_V": peak,
                       "U2_V": u2, "VDC_V": dc, "U4_V": u4})
    return result


def cap_scale(model: str) -> float:
    lo, hi = 0.0, 1.0
    while max(r["peak_abs_voltage_V"] for r in peak_by_electrode(model, hi)) < V_LIMIT:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if max(r["peak_abs_voltage_V"] for r in peak_by_electrode(model, mid)) <= V_LIMIT:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    scale_by_model = {m: cap_scale(m) for m in ("SO", "RO_PLUS", "RO_MINUS")}
    rows = []
    for model, scale in [("Q", 0.0), *scale_by_model.items()]:
        for row in peak_by_electrode(model, scale):
            rows.append({"configuration": model, "electrode_scale_vs_A9": scale,
                         **row, "within_250V": row["peak_abs_voltage_V"] <= V_LIMIT + 1e-9})
    out = ROOT / "outputs/octupole_comparison_home/mass_filter_v15"
    with (out / "electrode_voltages_A9_equal_peak.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {
        "protocol": "A9 quadrupole voltages fixed; independently maximize SO and RO octupole amplitude under max_j,t |V_j(t)| <= 250 V",
        "delta_deg": 0.0, "target_peak_V": V_LIMIT, "M0_Th": M0_TH, "frequency_Hz": 100000.0,
        "R_m": R, "alpha": ALPHA, "beta": BETA, "base_kappa": KAPPA,
        "c2_d2": C2, "c4": C4, "scale_by_model": scale_by_model,
        "kappa_at_cap": {m: KAPPA*s for m,s in scale_by_model.items()},
        "U4_at_cap_V": {m: peak_by_electrode(m,s)[0]["U4_V"] for m,s in scale_by_model.items()},
        "max_voltage_by_model_V": {m:max(r["peak_abs_voltage_V"] for r in peak_by_electrode(m,s))
                                    for m,s in scale_by_model.items()},
        "caution": "SO equal-RF-intensity scaling relative to RO may exceed 250 V and is excluded from the saturated equal-peak comparison.",
    }
    (out / "equal_peak_voltage_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
