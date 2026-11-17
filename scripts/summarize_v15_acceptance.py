"""Convert finite-domain retention estimates into 4D phase-volume acceptance.

The domain is uniform in transverse position over r<=7.5 mm and in the
velocity disk corresponding to E_perp<=5 eV at the reference mass. For the
mass-ray table the physical energy cutoff is held fixed, so the domain's
velocity-space area scales as 1/mu.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v15"
R = 0.008
R0_MAX = 0.0075
F_HZ = 100_000.0
M0_TH = 3200.04652261404
AMU = 1.66053906660e-27
E_CHARGE = 1.602176634e-19
E_MAX_EV = 5.0
OMEGA = 2 * math.pi * F_HZ
M0 = M0_TH * AMU
VMAX2 = 2 * E_MAX_EV * E_CHARGE / M0
PHASE_VOLUME_MU1 = math.pi**2 * R0_MAX**2 * VMAX2
NORM_VOLUME_MU1 = PHASE_VOLUME_MU1 / (R**4 * OMEGA**2)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def append_row(out, *, source, mesh, mu, model, horizon, mean, low, high, n_scrambles):
    volume = PHASE_VOLUME_MU1 / float(mu)
    norm_volume = NORM_VOLUME_MU1 / float(mu)
    out.append({
        "source": source, "mesh_um": mesh, "mu": mu, "model": model,
        "horizon_rf": horizon, "P_ret": mean, "P_ci95_low": low,
        "P_ci95_high": high, "acceptance4_m4_s2": mean * volume,
        "acceptance4_ci95_low": low * volume,
        "acceptance4_ci95_high": high * volume,
        "acceptance4_norm": mean * norm_volume,
        "acceptance4_norm_ci95_low": low * norm_volume,
        "acceptance4_norm_ci95_high": high * norm_volume,
        "domain_phase_volume_mu1_m4_per_s2": PHASE_VOLUME_MU1,
        "domain_normalized_volume_mu1": NORM_VOLUME_MU1,
        "n_scrambles": n_scrambles,
    })


def main():
    rows = []
    ideal_path = BASE / "confirmation_a9_uniform4d/summary_by_phase_model_horizon.csv"
    for r in read_csv(ideal_path):
        if (r["metric"] != "P_total" or float(r["delta_rad"]) != 0
                or float(r["tau0_rad"]) != 0):
            continue
        append_row(rows, source="ideal_harmonic", mesh="ideal", mu=1.0,
                   model=r["config"], horizon=int(r["horizon_rf"]),
                   mean=float(r["mean"]), low=float(r["ci95_low"]),
                   high=float(r["ci95_high"]), n_scrambles=int(r["n_scrambles"]))

    for dirname in ("fem_confirmation_a9", "fem_confirmation_a9_refined",
                    "fem_mass_mu1p2_confirmation_fixed", "fem_equal_peak_a9"):
        path = BASE / dirname / "summary_by_phase_model_horizon.csv"
        if not path.exists():
            continue
        mu = float(json.loads((BASE / dirname / "manifest.json").read_text(encoding="utf8")).get("mu", 1.0))
        for r in read_csv(path):
            if (r["metric"] != "P_total" or float(r["delta_deg"]) != 0
                    or float(r["tau0_deg"]) != 0):
                continue
            append_row(rows, source=dirname, mesh=int(r["mesh_um"]), mu=mu,
                       model=r["config"], horizon=int(r["horizon_rf"]),
                       mean=float(r["mean"]), low=float(r["ci95_low"]),
                       high=float(r["ci95_high"]), n_scrambles=int(r["n_scrambles"]))

    mass_summary = BASE / "mass_ray_a9/retention_summary.csv"
    if mass_summary.exists():
        for r in read_csv(mass_summary):
            if r["metric"] != "P_total":
                continue
            append_row(rows, source="ideal_mass_ray", mesh="ideal", mu=float(r["mu"]),
                       model=r["config"], horizon=int(r["horizon_rf"]),
                       mean=float(r["mean"]), low=float(r["ci95_low"]),
                       high=float(r["ci95_high"]), n_scrambles=int(r["n_scrambles"]))

    path = BASE / "finite_domain_acceptance.csv"
    if not rows:
        raise RuntimeError("No retention summaries found")
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "domain_phase_volume_mu1_m4_per_s2": PHASE_VOLUME_MU1,
           "domain_normalized_volume_mu1": NORM_VOLUME_MU1,
           "output": str(path)})


if __name__ == "__main__":
    main()
