"""Plot the sampled N=6 Coulomb continuation and direction transition."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_coulomb_scan" / "coupling_audit.json"


def main() -> None:
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    rows = doc["results"]
    c = np.asarray([r["C_multiplier"] for r in rows], dtype=float)
    f1 = np.asarray([r["mean_frequency_Hz_by_group"][0] for r in rows])
    f2 = np.asarray([r["mean_frequency_Hz_by_group"][1] for r in rows])
    psi1 = np.asarray([r["mean_Psi3_by_group"][0] for r in rows])
    psi2 = np.asarray([r["mean_Psi3_by_group"][1] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(c, f1, "o-", lw=1.7, color="#1769aa", label="group 1")
    ax.plot(c, f2, "s-", lw=1.7, color="#b3261e", label="group 2")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvspan(2, 5, color="#fbbc04", alpha=0.2, label="transition bracket for this start")
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.8)
    ax.set_ylabel("mean angular frequency (Hz)")
    ax.set_title("local15 #505, same ordered start; 15,000 periods after loading")
    ax.grid(alpha=0.25)
    ax.legend()
    ax = axes[1]
    ax.plot(c, psi1, "o-", lw=1.5, color="#1769aa", label="group 1 Ψ₃")
    ax.plot(c, psi2, "s-", lw=1.5, color="#b3261e", label="group 2 Ψ₃")
    ax.axhline(0.9, color="black", lw=0.8, ls="--", label="structural guide 0.9")
    ax.axvspan(2, 5, color="#fbbc04", alpha=0.2)
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.8)
    ax.set_xlabel("C/C₀")
    ax.set_ylabel("mean Ψ₃")
    ax.set_ylim(0.9, 1.005)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(HERE / "P1_COULOMB_CONTINUATION.png", dpi=180)
    plt.close(fig)
    print(HERE / "P1_COULOMB_CONTINUATION.png")


if __name__ == "__main__":
    main()
