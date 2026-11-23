"""Quantify finite-window spread in the archived 240k-period reference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "research" / "v17c" / "v17c_21" / "code"))
import analytical_model as model  # noqa: E402


def main():
    source = ROOT / "research" / "v17c" / "v17c_20b" / "results" / "refine12_240k_corrected_drag_trajectory.npz"
    summary_path = ROOT / "research" / "v17c" / "v17c_20b" / "results" / "refine12_240k_corrected_drag_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = model.load_inputs()
    with np.load(source) as z:
        states = z["states"]
        step_index = z["step_index"]
        periods = step_index / int(summary["steps_per_period"])
    time_s = periods / data["f1_Hz"]
    theta = []
    radius = []
    for i in range(2):
        offset = 6 * i
        x, y = states[:, offset], states[:, offset + 1]
        theta.append(np.unwrap(np.arctan2(y, x)))
        radius.append(np.hypot(x, y))
    rows = []
    for horizon in (20000, 50000, 100000, 200000, 220000):
        mask = periods >= periods[-1] - horizon
        t = time_s[mask]
        row = {"window_periods": horizon, "samples": int(mask.sum()), "particles": []}
        for i in range(2):
            coeff = np.polyfit(t - t.mean(), theta[i][mask], 1)
            freq = float(coeff[0] / (2 * np.pi))
            endpoint_freq = float((theta[i][mask][-1] - theta[i][mask][0]) /
                                  (2 * np.pi * (t[-1] - t[0])))
            row["particles"].append({
                "frequency_Hz": freq,
                "frequency_endpoint_Hz": endpoint_freq,
                "mean_radius_R": float(np.mean(radius[i][mask])),
                "radius_std_R": float(np.std(radius[i][mask], ddof=1)),
                "frequency_delta_from_220k_Hz": None,
                "endpoint_frequency_delta_from_220k_Hz": None,
            })
        rows.append(row)
    ref = rows[-1]
    for row in rows:
        for i in range(2):
            row["particles"][i]["frequency_delta_from_220k_Hz"] = (
                row["particles"][i]["frequency_Hz"] - ref["particles"][i]["frequency_Hz"]
            )
            row["particles"][i]["endpoint_frequency_delta_from_220k_Hz"] = (
                row["particles"][i]["frequency_endpoint_Hz"] -
                ref["particles"][i]["frequency_endpoint_Hz"]
            )
    out = {
        "study": "V17C.21.1 reference finite-window sensitivity",
        "source": "archived V20B corrected-drag nominal trajectory; not reintegrated",
        "requested_periods": summary["requested_periods"],
        "completed_periods": summary["completed_periods"],
        "transient_periods": summary["transient_periods"],
        "steps_per_period": summary["steps_per_period"],
        "sample_stride_periods": float(np.median(np.diff(periods))),
        "f1_Hz": data["f1_Hz"],
        "windows": rows,
    }
    (BASE / "results" / "reference_window_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
