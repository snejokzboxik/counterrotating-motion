"""FIG05: discrete RK4 step-refinement errors against independent DOP853."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results/local15_exact_rf/numerical_convergence_S01.json"
doc = json.loads(src.read_text(encoding="utf-8"))
if not str(doc.get("status", "")).startswith("completed"):
    raise RuntimeError("FIG05 requires a completed independent DOP853 comparison")

rows = []
for species in ("species1", "species2"):
    for r in doc["results"]:
        if r["species"] != species or r["method"] != "RK4":
            continue
        rows.append({
            "species": species,
            "steps_per_period": r["steps_per_primary_period"],
            "abs_relative_radius_error_percent": abs(r["relative_rho_error_vs_DOP853_percent"]),
            "abs_relative_frequency_error_percent": abs(r["relative_abs_omega_error_vs_DOP853_percent"]),
            "rk4_radius": r["rho_mean_last_500"],
            "rk4_abs_omega": abs(r["omega_mean_last_500"]),
            "dop853_radius": next(x["rho_mean_last_500"] for x in doc["results"] if x["species"]==species and x["method"]=="DOP853"),
            "dop853_abs_omega": abs(next(x["omega_mean_last_500"] for x in doc["results"] if x["species"]==species and x["method"]=="DOP853")),
        })
csv_path = ROOT / "results/figure_data/fig05_method_convergence.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
    "axes.labelsize":9, "axes.titlesize":9, "mathtext.fontset":"dejavusans",
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
    "axes.spines.top":False,"axes.spines.right":False})
fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.1), sharex="col", constrained_layout=True)
metrics = [
    ("abs_relative_radius_error_percent", r"$|\Delta\langle\rho\rangle|/\langle\rho\rangle_{\rm DOP853}$ (%)", "Radius"),
    ("abs_relative_frequency_error_percent", r"Relative error in $|\bar\omega|$ (%)", "Mean-frequency magnitude"),
]
colors = {"species1":"#176b87", "species2":"#bf5726"}
for col, species in enumerate(("species1", "species2")):
    for row, (key, ylabel, title) in enumerate(metrics):
        ax = axs[row, col]
        subset = [r for r in rows if r["species"] == species]
        x = np.array([r["steps_per_period"] for r in subset])
        y = np.array([r[key] for r in subset])
        ax.scatter(x, y, s=34, color=colors[species], edgecolor="white", linewidth=.5, zorder=3)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3g}", (xi, yi), xytext=(0,5), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#333333")
        ax.set_yscale("log")
        ax.set_xticks([64,96,128], labels=["64","96","128"])
        ax.set_ylabel(ylabel)
        panel = ("a" if row == 0 and col == 0 else
                 "b" if row == 0 and col == 1 else
                 "c" if row == 1 and col == 0 else "d")
        ax.set_title(f"({panel}) {title} — {species}", loc="left")
        ax.grid(True, which="major", color="#d8dde3", lw=.45)
        ax.set_xlim(56,136)
        if row == 1:
            ax.set_xlabel("RK4 steps per primary RF period")
fig.suptitle("S01, 15,000 periods; independent DOP853 reference (rtol=$2\\times10^{-11}$, atol=$2\\times10^{-13}$)", fontsize=9.5)
for ext,dpi in (("pdf",None),("svg",None),("png",600)):
    kw = {} if dpi is None else {"dpi":dpi}
    fig.savefig(ROOT/f"figures/FIG05_LOCAL15_METHOD_CONVERGENCE.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print(json.dumps({"figure":"FIG05_LOCAL15_METHOD_CONVERGENCE","points":len(rows),"data":str(csv_path)}))
