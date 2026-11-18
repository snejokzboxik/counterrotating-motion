"""Reclassify the saved 120 spectra with a zero-moment-aware root finder."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results" / "phase_optimization"


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    old = read(ROOT / "repeated_six_phase_permutation_rank.csv")
    spectra = json.loads((ROOT / "repeated_six_phase_spectra_p64.json").read_text(encoding="utf-8"))
    rows = []
    for raw in old:
        coeff = {int(k): complex(re, im) for k, re, im in spectra[raw["sequence"]]}

        def chi(r):
            term = [np.sign(k)*abs(k)**2*(abs(k)-1)*abs(c)**2*r**(2*abs(k)-2)
                    for k, c in coeff.items()]
            return float(sum(term)/max(sum(abs(x) for x in term), 1e-300))

        grid = np.linspace(.03, .98, 2400)
        values = np.asarray([chi(x) for x in grid])
        if np.max(np.abs(values)) < 1e-8:
            roots, status = [], "numerically_zero"
        else:
            roots = []
            for x, y, vx, vy in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
                if vx*vy < 0 and max(abs(vx), abs(vy)) > 1e-10:
                    root = float(brentq(lambda rr: sum(np.sign(k)*abs(k)**2*(abs(k)-1)*abs(c)**2*rr**(2*abs(k)-2)
                                                       for k,c in coeff.items()), x, y))
                    if not roots or abs(root-roots[-1]) > 1e-5:
                        roots.append(root)
            status = "resolved" if roots else "single_sign_or_no_resolved_crossing"
        width = 0.0
        if len(roots) > 1:
            for x, y in zip(roots[:-1], roots[1:]):
                if chi(.5*(x+y)) < 0:
                    width = max(width, y-x)
        rows.append({**raw, "chi_03": chi(.3), "chi_055": chi(.55), "chi_075": chi(.75),
                     "torque_roots": ";".join(map(str, roots)), "root_status": status,
                     "negative_band_width": width})
    rows.sort(key=lambda x: (min(float(x["chi_03"]), -float(x["chi_055"])),
                             float(x["negative_band_width"]),
                             -float(x["angular_corrugation_r055"])), reverse=True)
    path = ROOT / "repeated_six_phase_permutation_rank.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    strong = [r for r in rows if float(r["chi_03"])>.3 and float(r["chi_075"])<-.3]
    moderate = [r for r in rows if float(r["chi_03"])>.1 and float(r["chi_055"])<-.1]
    summary = {
        "permutations": len(rows), "strict_root_status_numerically_zero": sum(r["root_status"]=="numerically_zero" for r in rows),
        "moderate_inner_positive_band_negative_count": len(moderate),
        "strong_chi03_gt_0p3_chi075_lt_minus0p3_count": len(strong),
        "top_12": [{k:r[k] for k in ("sequence","chi_03","chi_055","chi_075","torque_roots","negative_band_width",
                                      "angular_corrugation_r03","angular_corrugation_r055","surface_Emax_U0_over_D",
                                      "max_adjacent_voltage_difference_U0")} for r in rows[:12]],
        "criteria": "radially averaged torque from the complete p=64, K=72 MFS spectrum; roots suppressed when normalized chirality is numerically zero",
    }
    (ROOT/"repeated_six_phase_permutation_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
