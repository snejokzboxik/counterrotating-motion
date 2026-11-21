"""Window-by-window sign audit for exact-RF Local15 frequency scans."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results/rotation_sign_scan"
PANEL = ROOT / "results/frozen_local15_single_particle_panel.csv"


def main():
    with PANEL.open(newline="", encoding="utf-8") as f:
        starts = list(csv.DictReader(f))
    entries = []
    trajectories = []
    for path in sorted(SCAN.glob("nu_*_species1.npz")):
        if ".tmp." in path.name:
            continue
        with np.load(path, allow_pickle=False) as z:
            nu = float(z["nu"])
            horizon = int(z["step"]) // int(z["steps_per_period"])
            steps = int(z["steps_per_period"])
            angle = z["angle_samples"].copy()
            rho = z["rho_samples"].copy()
            escape = z["escape_period"].copy()
        for H in (5000, 10000, 15000):
            if H > horizon:
                continue
            omegas = (angle[H] - angle[H - 500]) / (2 * np.pi * 500)
            retained = np.isnan(escape) | (escape > H)
            values = omegas[retained]
            counts = {"negative": int(np.sum(values < 0)),
                      "zero": int(np.sum(values == 0)),
                      "positive": int(np.sum(values > 0))}
            entries.append({"nu": nu, "species": "species1", "horizon_periods": H,
                "measurement_window_periods": 500,
                "steps_per_primary_period": steps,
                "retained": int(np.sum(retained)), "escaped_by_horizon": int(len(starts)-np.sum(retained)),
                **counts,
                "mean_omega": None if not len(values) else float(np.mean(values)),
                "min_omega": None if not len(values) else float(np.min(values)),
                "max_omega": None if not len(values) else float(np.max(values)),
                "mean_rho": None if not np.any(retained) else float(np.mean(rho[H-500:H+1, retained]))})
            for i, start in enumerate(starts):
                escaped = bool(np.isfinite(escape[i]) and escape[i] <= H)
                trajectories.append({"nu": nu, "species": "species1", "start_id": start["start_id"],
                    "horizon_periods": H, "escaped": escaped,
                    "escape_period": None if not np.isfinite(escape[i]) else float(escape[i]),
                    "rho_mean_last_500": float(np.mean(rho[H-500:H+1, i])),
                    "omega_last_500": float(omegas[i]),
                    "sign": 0 if escaped else int(np.sign(omegas[i]))})
    entries.sort(key=lambda r: (r["nu"], r["horizon_periods"]))
    trajectories.sort(key=lambda r: (r["nu"], r["horizon_periods"], r["start_id"]))
    A = 0.03716194907442204
    beta = 0.017273303124515367
    alpha = 1.0
    lam = 0.20975541409005094
    k1 = 4 * alpha**2 * A**2 / (1 + lam**2)
    nu_lead_zero = alpha * beta / k1 - 1
    final = [e for e in entries if e["horizon_periods"] == 15000 and e["retained"] > 0]
    all_negative = [e["nu"] for e in final
                    if e["negative"] == e["retained"] and e["positive"] == 0 and e["zero"] == 0]
    all_positive = [e["nu"] for e in final
                    if e["positive"] == e["retained"] and e["negative"] == 0 and e["zero"] == 0]
    transition_band = [max(all_negative), min(all_positive)]
    obj = {"status": "completed", "model_scope": "fixed panel, exact ideal two-mode Local15, species 1; no Coulomb or full MFS",
        "panel_seed": 15152026,
        "panel_sha256": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        "fixed_parameters": {"A": A, "B": 0.1626938199343468, "beta": beta,
            "alpha": alpha, "lambda": lam, "C": 0},
        "integrator": "RK4, 96 steps per primary period",
        "nu_lead_zero": nu_lead_zero,
        "observed_exact_panel_transition_band": transition_band,
        "delta_nu_interval_exact_band_minus_leading_zero":
            [transition_band[0] - nu_lead_zero, transition_band[1] - nu_lead_zero],
        "transition_band_interpretation": "sampled panel-wide all-negative to all-positive bracket; mixed-start outcomes occur inside, and this is not a single asymptotic critical frequency",
        "probability_note": "Fixed covering panel; counts are deterministic outcomes, not iid probabilities.",
        "windows": entries, "trajectories": trajectories}
    out = SCAN / "local15_sign_window_audit.json"
    out.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with (SCAN / "local15_sign_window_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    with (SCAN / "local15_sign_trajectory_windows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectories[0])); writer.writeheader(); writer.writerows(trajectories)
    print(json.dumps({"output": out.name, "windows": len(entries),
        "trajectory_window_rows": len(trajectories), "nu_lead_zero": nu_lead_zero,
        "observed_band": obj["observed_exact_panel_transition_band"]}))


if __name__ == "__main__":
    main()
