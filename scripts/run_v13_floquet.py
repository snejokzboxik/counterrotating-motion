"""Reproducible first-region Floquet pilot; no V12 files are modified."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from three_phase_trap.mass_filter_v13 import linear_status


ROOT = Path(__file__).resolve().parents[1] / "outputs/octupole_comparison_home/mass_filter_v13"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    alpha_values = np.linspace(-0.12, 0.12, 49)
    beta_values = np.linspace(0, 0.35, 71)
    status_map = np.empty((len(beta_values), len(alpha_values)), dtype=np.uint8)
    with (ROOT / "floquet_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "status", "trace_x", "trace_y", "det_x", "det_y", "margin_x", "margin_y"])
        writer.writeheader()
        for i, beta in enumerate(beta_values):
            for j, alpha in enumerate(alpha_values):
                result = linear_status(float(alpha), float(beta))
                writer.writerow({"alpha": f"{alpha:.12g}", "beta": f"{beta:.12g}", **result})
                status_map[i, j] = {"unstable": 0, "boundary_or_unresolved": 1,
                                    "strictly_stable": 2}[result["status"]]
    fig, ax = plt.subplots(figsize=(7.0, 5.4), layout="constrained")
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cm = ListedColormap(["#f1f1ef", "#e6a23c", "#255c99"])
    ax.pcolormesh(alpha_values, beta_values, status_map, shading="nearest",
                  cmap=cm, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], cm.N))
    ax.set(xlabel=r"$\alpha$", ylabel=r"$\beta$", title="Q: strict linear Floquet stability (first region)")
    ax.text(.02, .02, "Blue: both axes stable   Orange: boundary/unresolved", transform=ax.transAxes,
            fontsize=8, va="bottom")
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(ROOT / f"floquet_stability.{suffix}", dpi=240)
    plt.close(fig)
    (ROOT / "floquet_config.json").write_text(json.dumps({
        "model": "ideal V13 linear Q; kappa cancels", "period_tau": 2*np.pi,
        "method": "DOP853 fundamental matrix", "rtol": 2e-11, "atol": 2e-13,
        "trace_margin_tolerance": 1e-8, "alpha_points": len(alpha_values),
        "alpha_range": [-.12, .12], "beta_points": len(beta_values),
        "beta_range": [0, .35], "classification": "both axes, strictly |trace|<2"
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
