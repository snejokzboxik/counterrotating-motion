"""Join adaptive QMC retention estimates to their physical electrode drive."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
V13_1 = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
M0_TH = 3200.04652261404
OMEGA = 2 * math.pi * 100_000.0
RADIUS = 0.008
C2 = D2 = 0.7718255558274212
C4 = 0.5413395699439091


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    selected = {
        f"a{row['alpha_i']}_b{row['beta_i']}_e{row['eta_i']}": row
        for row in read_csv(BASE / "adaptive_candidate_points.csv")
        if row.get("long_horizon_selected", "").lower() == "true"
    }
    summary = read_csv(BASE / "adaptive_refinement_summary.csv")
    electrodes = [
        row for row in read_csv(V13_1 / "electrode_voltage_peaks.csv")
        if row["scheme"] == "Q"
    ]
    scale = M0_TH * AMU * OMEGA**2 * RADIUS**2 / E_CHARGE
    output = []
    for row in summary:
        point = selected.get(row["candidate_id"])
        if point is None or int(row["horizon_RF"]) not in (100, 300, 1000):
            continue
        alpha, beta, eta = (float(point[key]) for key in ("alpha", "beta", "eta"))
        u2 = beta * scale / C2
        vdc = alpha * scale / D2
        u4 = eta * beta * scale / C4
        peaks = [
            abs(vdc * float(e["Q2"]))
            + math.hypot(
                u2 * float(e["Q2"]) + u4 * float(e["C4"]),
                u4 * float(e["S4"]),
            )
            for e in electrodes
        ]
        q_peaks = [abs(vdc * float(e["Q2"])) + abs(u2 * float(e["Q2"])) for e in electrodes]
        output.append({
            "candidate_id": row["candidate_id"],
            "selection_reason": point["selection_reason"],
            "alpha": alpha, "beta": beta, "kappa": float(point["kappa"]), "eta": eta,
            "floquet_status": point["floquet_status"],
            "U2_V": u2, "VDC_V": vdc, "U4_setting_V": u4,
            "max_instantaneous_electrode_abs_V_RO": max(peaks),
            "max_instantaneous_electrode_abs_V_Q": max(q_peaks),
            "electrode_peaks_abs_V_RO": json.dumps(peaks, separators=(",", ":")),
            "phase_deg": float(row["phase_deg"]),
            "horizon_RF": int(row["horizon_RF"]),
            "n_replicates": int(row["n_replicates"]),
            "states_per_replica": int(row["n_per_replica"]),
            "P_RO": float(row["RO_P_ret_rep_mean"]),
            "P_RO_t95_low": float(row["RO_QMC_t95_low"]),
            "P_RO_t95_high": float(row["RO_QMC_t95_high"]),
            "P_Q": float(row["Q_P_ret_rep_mean"]),
            "paired_RO_minus_Q": float(row["paired_RO_minus_Q_rep_mean"]),
            "paired_t95_low": float(row["paired_QMC_t95_low"]),
            "paired_t95_high": float(row["paired_QMC_t95_high"]),
            "unresolved_RO": int(row["unresolved_RO_total"]),
            "uncertainty_method": row["warning"],
        })
    write_csv(BASE / "adaptive_key_regimes.csv", output)
    print(json.dumps({
        "selected_points": len(selected),
        "summary_rows": len(output),
        "max_peak_V": max(float(row["max_instantaneous_electrode_abs_V_RO"]) for row in output),
        "output": str(BASE / "adaptive_key_regimes.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
