"""Compute a conservative finite-time retention bound for ideal multipoles.

This is an analytical comparison bound, not a trajectory-based acceptance
estimate. It uses only the maximum possible acceleration inside a chosen disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def bound(order: int, kappa: float, horizon_rf: int, aperture_rho: float) -> dict:
    if order < 1 or kappa < 0 or horizon_rf < 1 or aperture_rho <= 0:
        raise ValueError("order, horizon and aperture must be positive; kappa must be nonnegative")
    T = 2.0 * math.pi * horizon_rf
    if order == 1:
        b = aperture_rho
    elif order == 2:
        b = aperture_rho if kappa * T**2 < 1.0 else 0.0
    else:
        stationary = (1.0 / (0.5 * order * (order - 1) * kappa * T**2)) ** (1.0 / (order - 2)) if kappa else math.inf
        b = min(aperture_rho, stationary)
    if b <= 0:
        d = 0.0
    else:
        max_acceleration = order * kappa * b ** (order - 1)
        d = b - 0.5 * max_acceleration * T**2
    d = max(0.0, d)
    a4 = math.pi**2 * d**4 / (6.0 * T**2) if d > 0 else 0.0
    return {
        "order_m": order, "kappa": kappa, "horizon_RF": horizon_rf,
        "aperture_rho": aperture_rho, "bound_disk_radius_b": b,
        "guaranteed_initial_radius_d": d,
        "max_guaranteed_speed_at_center_u": d / T,
        "a4_lower_bound_if_domain_contains_subset": a4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="4,6")
    parser.add_argument("--kappas", required=True,
                        help="comma-separated values corresponding to --orders")
    parser.add_argument("--horizons", default="1000,3000,10000,30000,100000")
    parser.add_argument("--aperture-rho", type=float, default=1.0)
    parser.add_argument("--mass-th", type=float, default=100000.0)
    parser.add_argument("--frequency-hz", type=float, default=100000.0)
    parser.add_argument("--length-mm", type=float, default=8.0)
    parser.add_argument("--injection-radius-mm", type=float, default=7.5)
    parser.add_argument("--max-energy-ev", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    orders = [int(v) for v in args.orders.split(",")]
    kappas = [float(v) for v in args.kappas.split(",")]
    horizons = [int(v) for v in args.horizons.split(",")]
    if len(orders) != len(kappas):
        parser.error("--orders and --kappas must have the same number of entries")

    mass = args.mass_th * AMU
    omega = 2.0 * math.pi * args.frequency_hz
    length = args.length_mm * 1e-3
    physical_factor = length**4 * omega**2
    velocity_max = math.sqrt(2.0 * args.max_energy_ev * E_CHARGE / mass)
    rho_inj = args.injection_radius_mm * 1e-3 / length
    u_max = velocity_max / (length * omega)
    rows = []
    for order, kappa in zip(orders, kappas):
        for horizon in horizons:
            row = bound(order, kappa, horizon, args.aperture_rho)
            d = row["guaranteed_initial_radius_d"]
            row.update({
                "rho_injection_max": rho_inj,
                "u_injection_max": u_max,
                "bound_subset_inside_injection_domain": bool(d <= rho_inj and row["max_guaranteed_speed_at_center_u"] <= u_max),
                "A4_lower_bound_if_domain_contains_subset_m4_s2": row["a4_lower_bound_if_domain_contains_subset"] * physical_factor,
                "physical_max_guaranteed_speed_at_center_m_s": row["max_guaranteed_speed_at_center_u"] * length * omega,
            })
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "orders": orders, "kappas": kappas, "horizons_RF": horizons,
        "aperture_rho": args.aperture_rho, "mass_Th": args.mass_th,
        "frequency_Hz": args.frequency_hz, "length_mm": args.length_mm,
        "injection_radius_mm": args.injection_radius_mm,
        "max_energy_eV": args.max_energy_ev,
        "volume_conversion": "A4 = R^4 Omega^2 a4",
        "method": "maximum acceleration bound on disk b; analytic phase-volume subset",
    }
    args.output.with_suffix(".config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
