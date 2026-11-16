"""Audit paired h50 P4 controls against the matching ideal-atlas starts."""
from __future__ import annotations

import csv
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mesh-um",type=int,choices=(35,50),default=50)
    args=parser.parse_args()
    fem_dir=BASE/("fem_controls" if args.mesh_um==50 else "fem_controls_h35")
    fem_rows = read_csv(fem_dir / "trajectory_results.csv")
    ideal_rows = read_csv(BASE / "atlas_trajectory_results.csv")
    points = read_csv(BASE / "fem_control_points.csv")
    step_path=fem_dir / "step_halving_results.csv"
    step_rows = read_csv(step_path) if step_path.exists() else []
    step_uncertain = {(r["candidate_id"], r["model"], round(float(r["tau0_rad"]), 12), int(r["state_i"]))
                      for r in step_rows if r["coarse_FEM_status"] != r["step_refined_status"]}
    basis = read_csv(ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1/electrode_voltage_peaks.csv")
    electrode_rows = [r for r in basis if r["scheme"] == "Q"]
    fem = {}
    for row in fem_rows:
        key = (row["candidate_id"], row["model"], float(row["tau0_rad"]),
               int(row["state_i"]), int(row["horizon"]))
        if key in fem:
            raise RuntimeError(f"duplicate FEM key {key}")
        fem[key] = row
    ideal = {}
    for row in ideal_rows:
        key = (row["model"], round(float(row["alpha"]), 12), round(float(row["beta"]), 12),
               round(float(row["kappa"]), 12), round(float(row["tau0_rad"]), 12), int(row["state_i"]))
        ideal[key] = row

    point_summaries = []
    pair_rows = []
    continuation_errors = []
    compare_status = Counter()
    missing_ideal = 0
    for point in points:
        cid = point["candidate_id"]
        alpha, beta, kappa = map(float, (point["alpha"], point["beta"], point["kappa"]))
        for model in ("Q", "RO"):
            for phase in (0.0, math.pi / 2):
                for state_i in range(8):
                    row100 = fem.get((cid, model, phase, state_i, 100))
                    if row100 is None:
                        raise RuntimeError(f"missing FEM 100-cycle task {(cid, model, phase, state_i)}")
                    ik = 0.0 if model == "Q" else kappa
                    ir = ideal.get((model, round(alpha, 12), round(beta, 12), round(ik, 12),
                                    round(phase, 12), state_i))
                    if ir is None:
                        missing_ideal += 1
                    else:
                        same = row100["status"] == ir["status"]
                        compare_status["agree" if same else "different"] += 1
                        pair_rows.append({
                            "candidate_id": cid, "model": model,
                            "alpha": alpha, "beta": beta, "kappa": ik,
                            "eta": 0.0 if model == "Q" else ik / beta,
                            "tau0_deg": math.degrees(phase), "state_i": state_i,
                            "ideal_status": ir["status"], "FEM_status": row100["status"],
                            "status_agree": same,
                            "ideal_exit_RF": ir["exit_periods"],
                            "FEM_exit_RF": row100["exit_periods"],
                            "ideal_rho_max": ir["rho_max"],
                            "FEM_rho_max": row100["rho_max_cumulative"],
                            "integrator_ideal": ir["integrator"],
                            "integrator_FEM": row100["integrator"],
                            "note": "status agreement is a small diagnostic comparison; near-boundary discordance is kept explicit",
                        })
                    row300 = fem.get((cid, model, phase, state_i, 300))
                    if row100["status"] == "retained" and row300 is None:
                        raise RuntimeError(f"missing retained continuation {(cid, model, phase, state_i)}")
                    if row300 is not None:
                        # The row stores the segment's start explicitly. Compare it with the 100-cycle endpoint.
                        start = [float(row300[k]) for k in ("x_start_rho", "y_start_rho", "vx_start_rho_tau", "vy_start_rho_tau")]
                        end100 = json.loads(row100["final_state"])
                        continuation_errors.append(max(abs(a - b) for a, b in zip(start, end100)))

        rows = [r for key, r in fem.items() if key[0] == cid]
        max_voltage = max(float(r["max_electrode_peak_abs_V"]) for r in rows if r.get("max_electrode_peak_abs_V")) if any(r.get("max_electrode_peak_abs_V") for r in rows) else math.nan
        # Voltage is reconstructed from the same 12-electrode basis table if not duplicated in rows.
        if not math.isfinite(max_voltage):
            scale = 3200.04652261404 * 1.66053906660e-27 * (2 * math.pi * 100_000) ** 2 * .008 ** 2 / 1.602176634e-19
            u2, vdc, u4 = beta * scale / .7718255558274212, alpha * scale / .7718255558274212, kappa * scale / .5413395699439091
            max_voltage = max(abs(vdc * float(r["Q2"])) + math.hypot(u2 * float(r["Q2"]) + u4 * float(r["C4"]), u4 * float(r["S4"])) for r in electrode_rows)
        for model in ("Q", "RO"):
            for horizon in (100, 300):
                retained = 0
                escaped = 0
                unresolved = 0
                for phase in (0.0, math.pi / 2):
                    for state_i in range(8):
                        at100 = fem[(cid, model, phase, state_i, 100)]
                        uncertain = (cid, model, round(phase, 12), state_i) in step_uncertain
                        if uncertain:
                            status = "unresolved"
                        elif at100["status"] == "escaped":
                            status = "escaped"
                        elif horizon == 100:
                            status = at100["status"]
                        else:
                            row = fem.get((cid, model, phase, state_i, 300))
                            status = row["status"] if row is not None else "unresolved"
                        retained += status == "retained"
                        escaped += status == "escaped"
                        unresolved += status == "unresolved"
                point_summaries.append({
                    "candidate_id": cid, "model": model, "alpha": alpha, "beta": beta,
                    "kappa": 0.0 if model == "Q" else kappa,
                    "eta": 0.0 if model == "Q" else kappa / beta,
                    "horizon_RF": horizon, "n_initial": 16, "n_retained": retained,
                    "n_escaped": escaped, "n_unresolved": unresolved,
                    "P_ret_lower": retained / 16, "P_ret_upper": (retained + unresolved) / 16,
                    "max_electrode_peak_abs_V": max_voltage,
                    "within_250V": max_voltage <= 250.0,
                })

    write_csv(fem_dir / "fem_control_summary.csv", point_summaries)
    write_csv(fem_dir / "ideal_vs_fem_pairs.csv", pair_rows)
    metadata = {
        "field": f"full h{args.mesh_um} P4 FEM superposition on saved full 12-electrode mesh",
        "mesh_um":args.mesh_um,
        "points": [{"candidate_id": r["candidate_id"], "alpha": float(r["alpha"]),
                    "beta": float(r["beta"]), "kappa": float(r["kappa"]),
                    "reason": r["reason"]} for r in points],
        "paired_ideal_vs_FEM_status_pairs": int(sum(compare_status.values())),
        "paired_status_agree": int(compare_status["agree"]),
        "paired_status_disagree": int(compare_status["different"]),
        "missing_matching_ideal_rows": missing_ideal,
        "unresolved_FEM_rows": sum(r["status"] == "unresolved" for r in fem_rows),
        "step_sensitive_trajectory_count": len(step_uncertain),
        "step_sensitive_trajectory_keys": [list(k) for k in sorted(step_uncertain)],
        "max_abs_continuation_state_difference": max(continuation_errors, default=0.0),
        "horizon_reporting": "300-cycle retained fraction uses the original n=16 denominator; earlier escapes remain escaped; cases changing status under step halving are unresolved and contribute to upper/lower retention bounds",
        "limitations": f"Only four representative points, eight initial states and two injection phases. This is a field-model diagnostic, not a statistical atlas estimate. The full mesh retains higher harmonics. Mesh h{args.mesh_um} um P4 controls are reported separately from the other mesh resolution.",
    }
    (fem_dir / "fem_control_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    for ax,point in zip(axes.flat,points):
        cid=point["candidate_id"]
        for model,color,offset,label in (("Q","#355c9a",-.16,"Q"),("RO","#bd4b37",.16,"Q+RO")):
            rs=sorted((r for r in point_summaries if r["candidate_id"]==cid and r["model"]==model),key=lambda r:r["horizon_RF"])
            xs=[0+offset,1+offset];ys=[float(r["P_ret_lower"]) for r in rs]
            lows=[y for y in ys];highs=[float(r["P_ret_upper"]) for r in rs]
            ax.errorbar(xs,ys,yerr=[[0.,0.],[highs[i]-ys[i] for i in range(len(ys))]],fmt="o-",capsize=3,
                        color=color,label=label if cid==points[0]["candidate_id"] else None)
        ax.set_xticks([0,1],["100 RF","300 RF"]);ax.set_ylim(-.05,1.05);ax.grid(axis="y",alpha=.25)
        ax.set_title(cid.replace("_"," "))
    axes[0,0].set_ylabel("retained fraction, n=16")
    axes[1,0].set_ylabel("retained fraction, n=16")
    axes[0,0].legend(frameon=False)
    fig.suptitle(f"Full h{args.mesh_um} P4 FEM controls; dots are sampled estimates, bars show only step-sensitive bounds")
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig(fem_dir/f"fem_retention_controls.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
