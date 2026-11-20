"""RK4/64 and /128 step check against the saved RK4/96 P7 #292 holdout run."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "strict_loading_holdout"
BASE = HERE.parent / "v17c_9" / "baseline"
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
import basin_scan
import loading_dynamics as ld


def main():
    records = [json.loads(s) for s in (OUT / "holdout_runs.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    row = next(r for r in records if r["protocol_id"] == 4 and r["ensemble"] == "edge"
               and r["category"] == "CR_screen")
    protocols = json.loads((OUT / "holdout_protocols.json").read_text(encoding="utf-8"))
    protocol = np.asarray(protocols[4]["protocol_vector_19"], dtype=np.float64)
    pdoc = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
    p = np.asarray(pdoc["candidate"]["p"], dtype=np.float64)
    qratio = float(pdoc["candidate"]["q2_over_q1"])
    _, family, modes, coeff = basin_scan.load_model()
    skip = int(math.ceil(ld.protocol_duration_cycles(protocol))) + 200
    periods = skip + 15_000
    levels = []
    for steps in (64, 128):
        t0 = time.perf_counter()
        r = ld.integrate(np.asarray(row["initial_state"], dtype=np.float64),
                         np.asarray(row["initial_rf_phase"], dtype=np.float64),
                         p, protocol, family, modes, coeff, periods, steps, skip, qratio,
                         a1_over_R=0.0002, a2_over_R=0.00189)
        levels.append({"steps_per_period": steps, "category": r["category"],
                       "frequency_Hz": r["frequency_Hz"].tolist(),
                       "turns": r["turns"].tolist(), "window_turns": r["window_turns"].tolist(),
                       "r_min": r["r_min"].tolist(), "r_max": r["r_max"].tolist(),
                       "minimum_separation": r["minimum_separation"],
                       "final_state": r["final_state"].tolist(),
                       "elapsed_seconds": time.perf_counter() - t0})
        print(steps, r["category"], r["frequency_Hz"], "elapsed", levels[-1]["elapsed_seconds"], flush=True)
    base_state = np.asarray(row["final_state"], dtype=np.float64)
    for level in levels:
        level["max_abs_final_state_delta_vs_RK4_96"] = float(np.max(np.abs(np.asarray(level["final_state"]) - base_state)))
        level["max_abs_frequency_delta_Hz_vs_RK4_96"] = float(np.max(np.abs(np.asarray(level["frequency_Hz"]) - np.asarray(row["frequency_Hz"]))))
    result = {"protocol": "P7_292", "ensemble": "edge", "start_id": row["start_id"],
              "reference_RK4_steps": 96, "reference_category": row["category"],
              "reference_frequency_Hz": row["frequency_Hz"], "levels": levels}
    (OUT / "step_convergence_P7_292.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("max frequency delta Hz", max(x["max_abs_frequency_delta_Hz_vs_RK4_96"] for x in levels))


if __name__ == "__main__": main()
