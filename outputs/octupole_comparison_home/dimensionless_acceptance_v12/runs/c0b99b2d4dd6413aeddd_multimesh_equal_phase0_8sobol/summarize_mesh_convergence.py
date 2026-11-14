
from __future__ import annotations
import csv, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
assert cfg["protocol"] == "equal"
assert cfg["phase_mode"] == "fixed" and cfg["phase_deg"] == 0.0
assert cfg["horizon_RF"] == 1000
assert cfg["meshes_um"] == [35, 50] and cfg["models"] == ["A", "B", "C"]
assert cfg["scales"] == [1.0]

states = {}
with (ROOT / "initial_states_r00.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sid = f"r{int(row['replicate']):02d}_{int(row['sample_index']):06d}"
        states[sid] = row

records = {}
for path in (ROOT / "checkpoints").rglob("*.json"):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if row.get("complete") and row.get("status") in ("retained", "escaped"):
        records[(row["model"], row["sample_id"])] = row

expected = {f"{kind}_h{mesh}_scale1" for kind in "ABC" for mesh in (35,50)}
assert len(states) == 8, f"expected 8 starts, got {len(states)}"
assert len(records) == 48, f"expected 48 completed checkpoints, got {len(records)}"

rows = []
by_kind = defaultdict(list)
for kind in "ABC":
    for i in range(8):
        sid = f"r00_{i:06d}"
        initial = states[sid]
        r35 = records[(f"{kind}_h35_scale1", sid)]
        r50 = records[(f"{kind}_h50_scale1", sid)]
        assert r35["status"] in ("retained", "escaped") and r50["status"] in ("retained", "escaped")
        assert r35["phase_rad"] == 0.0 and r50["phase_rad"] == 0.0
        assert r35["voltage_V"] == 250.0 and r50["voltage_V"] == 250.0
        t35 = r35["escape_RF"]
        t50 = r50["escape_RF"]
        dt = None if t35 is None or t50 is None else float(t50) - float(t35)
        row = {
            "configuration": kind, "sample_id": sid,
            "x0_m": initial["x0_m"], "y0_m": initial["y0_m"],
            "vx0_m_s": initial["vx0_m_s"], "vy0_m_s": initial["vy0_m_s"],
            "E0_eV": initial["E0_eV"], "phase_deg": 0.0, "voltage_V": 250.0,
            "horizon_RF": 1000,
            "h35_status": r35["status"], "h50_status": r50["status"],
            "same_classification": r35["status"] == r50["status"],
            "h35_escape_RF": t35, "h50_escape_RF": t50,
            "delta_escape_RF_h50_minus_h35": dt,
            "h35_max_radius_sampled_rho": r35.get("max_radius_sampled_rho"),
            "h50_max_radius_sampled_rho": r50.get("max_radius_sampled_rho"),
            "delta_max_radius_sampled_rho_h50_minus_h35": (
                None if r35.get("max_radius_sampled_rho") is None or r50.get("max_radius_sampled_rho") is None
                else float(r50["max_radius_sampled_rho"])-float(r35["max_radius_sampled_rho"])
            ),
            "h35_nfev": r35.get("nfev"), "h50_nfev": r50.get("nfev"),
        }
        rows.append(row)
        by_kind[kind].append(row)

with (ROOT / "mesh_classification_comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary_by_kind = {}
for kind, group in by_kind.items():
    exits = [r["delta_escape_RF_h50_minus_h35"] for r in group if r["delta_escape_RF_h50_minus_h35"] is not None]
    summary_by_kind[kind] = {
        "starts": len(group),
        "h35_retained": sum(r["h35_status"] == "retained" for r in group),
        "h50_retained": sum(r["h50_status"] == "retained" for r in group),
        "h35_escaped": sum(r["h35_status"] == "escaped" for r in group),
        "h50_escaped": sum(r["h50_status"] == "escaped" for r in group),
        "classification_matches": sum(r["same_classification"] for r in group),
        "classification_disagreements": sum(not r["same_classification"] for r in group),
        "both_meshes_escaped": len(exits),
        "mean_signed_escape_time_difference_RF_h50_minus_h35": None if not exits else float(np.mean(exits)),
        "max_absolute_escape_time_difference_RF": None if not exits else float(np.max(np.abs(exits))),
        "rms_escape_time_difference_RF": None if not exits else float(np.sqrt(np.mean(np.square(exits)))),
    }

overall = {
    "study": "V12 paired N48 h35/h50 mesh-sensitivity pilot",
    "run_tag": ROOT.name,
    "config_hash": cfg["config_hash"],
    "status": "complete",
    "config": {
        "initial_states": len(states), "models": ["A","B","C"],
        "meshes_um": [35,50], "protocol": "equal 250 V",
        "phase_deg": 0.0, "horizon_RF": 1000,
        "sample_domain": cfg["domain"], "solver": cfg["solver"],
        "workers": cfg["compute"]["workers"], "seed": cfg["seed"],
        "sample_power": cfg["sample_power"], "replicates": cfg["replicates"],
    },
    "complete_trajectory_count": len(records),
    "planned_trajectory_count": cfg["replicates"] * (2**cfg["sample_power"]) * len(expected),
    "per_configuration": summary_by_kind,
    "all_classifications_match_across_meshes": all(r["same_classification"] for r in rows),
    "total_classification_disagreements": sum(not r["same_classification"] for r in rows),
    "finite_sample_warning": "Eight Sobol points with one scramble are a mesh-sensitivity pilot only; not a converged acceptance estimate and not evidence of permanent stability.",
    "comparison_csv": "mesh_classification_comparison.csv",
}
(ROOT / "mesh_convergence_summary.json").write_text(json.dumps(overall, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print(json.dumps(overall, indent=2))
