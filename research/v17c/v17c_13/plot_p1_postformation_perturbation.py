"""Plot return of triangular order after post-formation angle perturbations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_postformation_perturbations"


def get_metrics(path: Path) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    with np.load(path) as archive:
        logs = archive["trajectory"]
        skip = int(archive["post_perturb_skip"])
    logs = logs[skip:]
    periods = (logs[:, 0, 0] - logs[0, 0, 0]) / (2.0 * np.pi)
    theta = logs[:, :, 7]
    psi = []
    gap_err = []
    for g in (0, 1):
        th = theta[:, 3 * g:3 * g + 3]
        psi.append(np.abs(np.mean(np.exp(3j * th), axis=1)))
        errs = np.empty(len(th))
        for k, angles in enumerate(th):
            order = np.sort(np.mod(angles, 2.0 * np.pi))
            gaps = np.diff(np.r_[order, order[0] + 2.0 * np.pi])
            errs[k] = np.max(np.abs(np.rad2deg(gaps) - 120.0))
        gap_err.append(errs)
    return periods, psi, gap_err


def main() -> None:
    rows = [json.loads(s) for s in (DATA / "postformation_runs.jsonl").read_text().splitlines() if s.strip()]
    indexed = {(int(r["position_perturbation_deg"]), int(r["C_multiplier"])): r for r in rows}
    colors = ("#1769aa", "#b3261e")
    styles = {0: "--", 1: "-"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, angle in enumerate((10, 20)):
        axp, axg = axes[0, col], axes[1, col]
        for factor in (0, 1):
            row = indexed[(angle, factor)]
            path = DATA / f"{row['task_id']}_trajectory.npz"
            t, psi, gaps = get_metrics(path)
            for g in (0, 1):
                axp.plot(t, psi[g], color=colors[g], ls=styles[factor], lw=0.9,
                         label=f"group {g + 1}, C/C₀={factor}")
                axg.plot(t, gaps[g], color=colors[g], ls=styles[factor], lw=0.9,
                         label=f"group {g + 1}, C/C₀={factor}")
        axp.axhline(0.9, color="black", ls=":", lw=0.8)
        axp.set(title=f"{angle}°: triangular order Ψ₃", xlabel="periods after perturbation", ylabel="Ψ₃", ylim=(0.65, 1.01))
        axp.grid(alpha=0.25)
        axg.axhline(10.0, color="black", ls=":", lw=0.8)
        axg.set(title=f"{angle}°: max interval error", xlabel="periods after perturbation", ylabel="max |gap−120°| (deg)")
        axg.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 1].legend(fontsize=8, ncol=2)
    fig.suptitle("Post-formation perturbations with continuous RF phase; 15,000 periods")
    fig.savefig(HERE / "P1_POSTFORMATION_RECOVERY.png", dpi=180)
    plt.close(fig)
    print(HERE / "P1_POSTFORMATION_RECOVERY.png")


if __name__ == "__main__":
    main()
