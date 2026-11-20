"""Plot one independently integrated P7 #292 trajectory across loading and run."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "strict_loading_holdout"
sys.path.insert(0, str(HERE))
import loading_dynamics as ld


def main():
    recs = [json.loads(s) for s in (OUT / "dop853_representatives.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    rec = next(r for r in recs if r["protocol_label"] == "P7_292")
    z = np.load(OUT / rec["trajectory_file"])
    tau, state, prot = z["tau"], z["state"], z["protocol"]
    cycles = tau / (2 * np.pi)
    pos = state[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
    theta = np.unwrap(np.arctan2(pos[:, :, 1], pos[:, :, 0]), axis=0) / (2 * np.pi)
    turns = theta - theta[0]
    radius = np.linalg.norm(pos[:, :, :2], axis=2)
    env = np.asarray([ld._envelopes(float(t), prot) for t in tau])
    docs = json.loads((OUT / "holdout_protocols.json").read_text(encoding="utf-8"))
    duration = float(docs[4]["duration_cycles"])
    relax_end = np.ceil(duration) + 200.0
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(221)
    ax.plot(pos[:, 0, 0], pos[:, 0, 1], lw=.55, label="particle 1")
    ax.plot(pos[:, 1, 0], pos[:, 1, 1], lw=.55, label="particle 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="x/R", ylabel="y/R", title="DOP853 XY path, entire record")
    ax.legend()
    ax = fig.add_subplot(222)
    ax.plot(cycles, turns[:, 0], lw=.8, label="particle 1")
    ax.plot(cycles, turns[:, 1], lw=.8, label="particle 2")
    ax.axvline(duration, color="black", ls="--", lw=.8, label="load complete")
    ax.axvline(relax_end, color="grey", ls=":", lw=.9, label="relaxation complete")
    ax.set(xlabel="primary RF cycles", ylabel="unwrapped accumulated turns", title="Opposite orbital rotation")
    ax.legend()
    ax = fig.add_subplot(223)
    ax.plot(cycles, radius[:, 0], lw=.8, label="particle 1")
    ax.plot(cycles, radius[:, 1], lw=.8, label="particle 2")
    ax.axvline(duration, color="black", ls="--", lw=.8)
    ax.axvline(relax_end, color="grey", ls=":", lw=.9)
    ax.set(xlabel="primary RF cycles", ylabel="radius / R", title="Free radial motion")
    ax.legend()
    ax = fig.add_subplot(224)
    for i, lab in enumerate(("RF-A", "RF-B", "DC")):
        ax.plot(cycles, env[:, i], lw=1.1, label=lab)
    ax.axvline(duration, color="black", ls="--", lw=.8)
    ax.set_xlim(0, max(10.0, duration + 2.0))
    ax.set(xlabel="primary RF cycles", ylabel="fraction of final coefficient", title="Loading envelopes (early-time view)")
    ax.legend()
    fig.suptitle("V17C.11 P7 #292 — independent DOP853 edge holdout representative")
    fig.tight_layout()
    fig.savefig(OUT / "P7_292_dop853_loading_trajectory.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__": main()
