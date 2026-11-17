"""Summarize matched V15 A9 trajectory checkpoints on h50/h35 FEM fields."""
from __future__ import annotations

import csv
import json
import math
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v15/fem_confirmation_a9"
CFG_ORDER = ["Q", "SO_equal_harmonic", "SO_equal_rms", "RO_PLUS", "RO_MINUS"]
COLORS = {"Q": "#333333", "SO_equal_harmonic": "#2a6fbb", "SO_equal_rms": "#75aadb",
          "RO_PLUS": "#cf4c4c", "RO_MINUS": "#8d2d8d"}


def interval(values):
    a = np.asarray(values, dtype=float)
    mean = float(np.mean(a))
    if len(a) < 2 or np.std(a, ddof=1) == 0:
        return mean, mean, mean
    half = float(t.ppf(.975, len(a)-1) * sem(a))
    return mean, mean-half, mean+half


def read_rows():
    path = BASE / "checkpoint.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate FEM trajectory keys")
    return rows


def main():
    global BASE
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-name",default="fem_confirmation_a9",help="directory under mass_filter_v15")
    args=parser.parse_args()
    if Path(args.input_name).name!=args.input_name: parser.error("--input-name must be one directory name")
    BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v15"/args.input_name
    rows = read_rows()
    if not rows:
        raise RuntimeError("FEM checkpoint is empty")
    horizons = sorted({int(r["horizon_rf"]) for r in rows})
    reps = sorted({int(r["replica"]) for r in rows})
    out_by_group = defaultdict(list)
    for r in rows:
        out_by_group[(int(r["mesh_um"]), r["config"], float(r["delta_rad"]),
                      float(r["tau0_rad"]), int(r["horizon_rf"]))].append(r)

    # Group values per independent Sobol scramble; infer uncertainty across scrambles.
    rep_values = {}
    metrics = ("P_plus", "P_minus", "P_total", "Delta_chi", "Delta_ell_all",
               "Delta_ell_survivors", "Delta_ell_escaped", "median_exit_rf",
               "max_abs_torque_residual", "unresolved_fraction")
    for g, group in out_by_group.items():
        mesh, config, delta, tau0, horizon = g
        for rep in reps:
            rr = [r for r in group if int(r["replica"]) == rep]
            if not rr:
                continue
            plus = [r for r in rr if int(r["Lz_sign"]) > 0]
            minus = [r for r in rr if int(r["Lz_sign"]) < 0]
            kept = lambda rs: np.asarray([r["status"] == "retained" for r in rs], dtype=float)
            pp = float(np.mean(kept(plus))) if plus else math.nan
            pm = float(np.mean(kept(minus))) if minus else math.nan
            ret = [r for r in rr if r["status"] == "retained"]
            esc = [r for r in rr if r["status"] == "escaped"]
            exits = [float(r["exit_rf"]) for r in esc if r["exit_rf"] is not None]
            vals = {
                "P_plus": pp, "P_minus": pm,
                "P_total": float(np.mean(kept(rr))),
                "Delta_chi": pp-pm,
                "Delta_ell_all": float(np.mean([r["Delta_Lz"] for r in rr])),
                "Delta_ell_survivors": float(np.mean([r["Delta_Lz"] for r in ret])) if ret else math.nan,
                "Delta_ell_escaped": float(np.mean([r["Delta_Lz"] for r in esc])) if esc else math.nan,
                "median_exit_rf": float(np.median(exits)) if exits else math.nan,
                "max_abs_torque_residual": max(abs(float(r["torque_residual"])) for r in rr),
                "unresolved_fraction": float(np.mean([r["status"] == "unresolved" for r in rr])),
            }
            for metric, val in vals.items():
                rep_values[(g, rep, metric)] = val

    summary = []
    for g in sorted(out_by_group):
        mesh, config, delta, tau0, horizon = g
        for metric in metrics:
            vals = [rep_values.get((g, rep, metric), math.nan) for rep in reps]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                mean = low = high = math.nan
            else:
                mean, low, high = interval(vals)
                bounds = {"P_plus": (0., 1.), "P_minus": (0., 1.), "P_total": (0., 1.),
                          "Delta_chi": (-1., 1.), "unresolved_fraction": (0., 1.)}.get(metric)
                if bounds:
                    low, high = max(bounds[0], low), min(bounds[1], high)
            summary.append({"mesh_um": mesh, "config": config, "delta_deg": math.degrees(delta),
                            "tau0_deg": math.degrees(tau0), "horizon_rf": horizon, "metric": metric,
                            "mean": mean, "ci95_low": low, "ci95_high": high,
                            "n_scrambles": len(vals), "replicate_values": json.dumps(vals)})
    with (BASE / "summary_by_phase_model_horizon.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # Pair model outcomes on identical starts within each scramble; uncertainty is across scrambles.
    lookup = {}
    for r in rows:
        key = (int(r["mesh_um"]), float(r["delta_rad"]), float(r["tau0_rad"]),
               int(r["horizon_rf"]), int(r["replica"]), int(r["pair_index"]), int(r["Lz_sign"]))
        lookup.setdefault(key, {})[r["config"]] = float(r["status"] == "retained")
    pair_names = [("RO_PLUS", "Q", "RO_PLUS-Q"),
                  ("RO_PLUS", "SO_equal_harmonic", "RO_PLUS-SO_equal_harmonic"),
                  ("RO_PLUS", "SO_equal_rms", "RO_PLUS-SO_equal_rms"),
                  ("RO_PLUS", "RO_MINUS", "RO_PLUS-RO_MINUS")]
    paired = []
    groups_pair = sorted({key[:4] + (key[4],) for key in lookup})
    for mesh, delta, tau0, horizon, rep in groups_pair:
        for ca, cb, name in pair_names:
            vals = [d[ca]-d[cb] for k, d in lookup.items()
                    if k[:5] == (mesh, delta, tau0, horizon, rep) and ca in d and cb in d]
            if vals:
                paired.append((mesh, delta, tau0, horizon, rep, name, float(np.mean(vals))))
    p_summary = []
    for mesh, delta, tau0, horizon, name in sorted({(r[0], r[1], r[2], r[3], r[5]) for r in paired}):
        vals = [r[6] for r in paired if (r[0], r[1], r[2], r[3], r[5]) == (mesh, delta, tau0, horizon, name)]
        mean, low, high = interval(vals)
        p_summary.append({"mesh_um": mesh, "delta_deg": math.degrees(delta), "tau0_deg": math.degrees(tau0),
                          "horizon_rf": horizon, "contrast": name, "mean_difference_pp": 100*mean,
                          "ci95_low_pp": 100*low, "ci95_high_pp": 100*high,
                          "replicate_differences_pp": json.dumps([100*v for v in vals]),
                          "n_scrambles": len(vals)})
    if p_summary:
        with (BASE / "paired_retention_differences.csv").open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(p_summary[0])); w.writeheader(); w.writerows(p_summary)

    # Exact symmetry regression: Q/SO mirror, RO+ mirrored state vs RO- original state.
    idx = {}
    for r in rows:
        key = (int(r["mesh_um"]), r["config"], float(r["delta_rad"]), float(r["tau0_rad"]),
               int(r["horizon_rf"]), int(r["replica"]), int(r["pair_index"]), int(r["Lz_sign"]))
        idx[key] = r
    sym = {}
    for model in ("Q", "SO_equal_harmonic", "SO_equal_rms"):
        mismatch, n, max_exit, max_torque = 0, 0, 0.0, 0.0
        for key, r in idx.items():
            mesh, config, delta, tau0, hor, rep, pair, sign = key
            if config != model:
                continue
            mate = idx.get((mesh, config, delta, tau0, hor, rep, pair, -sign))
            if mate is None: continue
            n += 1
            mismatch += int(r["status"] != mate["status"])
            if r["exit_rf"] is not None and mate["exit_rf"] is not None:
                max_exit = max(max_exit, abs(float(r["exit_rf"])-float(mate["exit_rf"])))
            max_torque = max(max_torque, abs(float(r["Delta_Lz"])+float(mate["Delta_Lz"])))
        sym[model] = {"mirror_pairs": n, "status_mismatches": mismatch,
                      "max_exit_rf_difference": max_exit, "max_abs_mirror_Delta_ell_sum": max_torque}
    mismatch, n, max_exit, max_torque = 0, 0, 0.0, 0.0
    for key, r in idx.items():
        mesh, config, delta, tau0, hor, rep, pair, sign = key
        if config != "RO_PLUS": continue
        mate = idx.get((mesh, "RO_MINUS", delta, tau0, hor, rep, pair, -sign))
        if mate is None: continue
        n += 1
        mismatch += int(r["status"] != mate["status"])
        if r["exit_rf"] is not None and mate["exit_rf"] is not None:
            max_exit = max(max_exit, abs(float(r["exit_rf"])-float(mate["exit_rf"])))
        max_torque = max(max_torque, abs(float(r["Delta_Lz"])+float(mate["Delta_Lz"])))
    sym["RO_PLUS_to_RO_MINUS_reflection"] = {"mirror_pairs": n, "status_mismatches": mismatch,
        "max_exit_rf_difference": max_exit, "max_abs_mirror_Delta_ell_sum": max_torque}
    # Detailed symmetry diagnostics by mesh and horizon. A small number of
    # edge-sensitive paths can diverge strongly even when almost all pairs agree.
    sym_detailed = {}
    for mesh in sorted({int(r["mesh_um"]) for r in rows}):
      for horizon in horizons:
       entries = {}
       for key, r in idx.items():
        if key[0] == mesh and key[4] == horizon:
            entries[key] = r
       for model in ("Q", "SO_equal_harmonic", "SO_equal_rms", "RO_PLUS_to_RO_MINUS_reflection"):
        status_mismatch=0;exits=[];torques=[];n=0
        for key,r in entries.items():
            _,config,delta,tau0,hor,rep,pair,sign=key
            if model=="RO_PLUS_to_RO_MINUS_reflection":
                if config!="RO_PLUS": continue
                mate=entries.get((mesh,"RO_MINUS",delta,tau0,hor,rep,pair,-sign))
            else:
                if config!=model: continue
                mate=entries.get((mesh,model,delta,tau0,hor,rep,pair,-sign))
            if mate is None: continue
            n+=1;status_mismatch+=int(r["status"]!=mate["status"])
            if r["exit_rf"] is not None and mate["exit_rf"] is not None:
                exits.append(abs(float(r["exit_rf"])-float(mate["exit_rf"])))
            torques.append(abs(float(r["Delta_Lz"])+float(mate["Delta_Lz"])))
        quant=lambda xs: ({"median":float(np.quantile(xs,.5)),"p95":float(np.quantile(xs,.95)),
                           "p99":float(np.quantile(xs,.99)),"max":float(np.max(xs))} if xs else None)
        sym_detailed[f"h{mesh}_N{horizon}_{model}"]={"mirror_pairs":n,"status_mismatches":status_mismatch,
          "exit_time_abs_difference_rf":quant(exits),"mirror_torque_balance_abs_error":quant(torques)}
    val = {"rows": len(rows), "unique_keys": len(set(r["key"] for r in rows)),
           "n_scrambled_replicates": len(reps), "groups_present": len(out_by_group),
           "horizons_present": horizons, "unresolved_rows": sum(r["status"] == "unresolved" for r in rows),
           "rows_over_250V": sum(float(r["peak_electrode_V"]) > 250 for r in rows),
           "max_abs_torque_residual": max(abs(float(r["torque_residual"])) for r in rows),
           "symmetry_global": sym,"symmetry_by_mesh_horizon":sym_detailed,
           "field_hashes": json.loads((BASE / "manifest.json").read_text(encoding="utf8"))["source_hashes"]}
    (BASE / "validation.json").write_text(json.dumps(val, indent=2, ensure_ascii=False)+"\n", encoding="utf8")

    # Compact publication plots: only computed points, uncertainty from independent scrambles.
    figdir = BASE / "figures"; figdir.mkdir(exist_ok=True)
    phases = sorted({(float(r["delta_deg"]), float(r["tau0_deg"])) for r in summary
                     if r["metric"] == "P_total"})
    ncols = 2 if len(phases) > 1 else 1
    nrows = max(1, math.ceil(len(phases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4.5*nrows),
                             sharey=True, squeeze=False)
    for ax, (ddeg, tdeg) in zip(axes.flat, phases):
        d, t0 = math.radians(ddeg), math.radians(tdeg)
        for mesh, ls in ((50, "-"), (35, "--")):
            for cfg in CFG_ORDER:
                for hor, marker in ((100, "o"), (300, "s")):
                    rs = [r for r in summary if r["mesh_um"] == mesh and r["config"] == cfg
                          and r["metric"] == "P_total" and r["horizon_rf"] == hor
                          and abs(r["delta_deg"]-math.degrees(d)) < 1e-8
                          and abs(r["tau0_deg"]-math.degrees(t0)) < 1e-8]
                    if not rs: continue
                    r = rs[0]
                    ax.errorbar([f"{cfg}\nh{mesh} N{hor}"], [100*r["mean"]],
                                yerr=[[100*(r["mean"]-r["ci95_low"])], [100*(r["ci95_high"]-r["mean"])]],
                                fmt=marker, color=COLORS[cfg], ms=5, capsize=2, linestyle="none")
        ax.set_title(f"δ={ddeg:.0f}°, τ₀={tdeg:.0f}°")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.grid(axis="y", alpha=.25)
    for ax in axes.flat[len(phases):]:
        ax.set_visible(False)
    axes[0,0].set_ylabel("Удержание к горизонту (%)")
    if nrows > 1:
        axes[-1,0].set_ylabel("Удержание к горизонту (%)")
    fig.suptitle("A9: сравнение Q/SO/RO на полном P4 FEM, точки и интервалы по scramble")
    fig.tight_layout(); fig.savefig(figdir/"v15_fem_a9_retention.png", dpi=220); fig.savefig(figdir/"v15_fem_a9_retention.svg"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mesh in zip(axes, (50, 35)):
        for cfg in CFG_ORDER:
            for hor, mk in ((100, "o"), (300, "s")):
                rs = [r for r in summary if r["mesh_um"] == mesh and r["config"] == cfg
                      and r["metric"] == "Delta_chi" and r["horizon_rf"] == hor
                      and r["delta_deg"] == 0 and r["tau0_deg"] == 0]
                if not rs: continue
                r=rs[0]
                ax.errorbar([f"{cfg} N{hor}"],[100*r["mean"]],
                    yerr=[[100*(r["mean"]-r["ci95_low"])],[100*(r["ci95_high"]-r["mean"])]],
                    fmt=mk,color=COLORS[cfg],capsize=3,ms=5)
        ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.25); ax.tick_params(axis="x",rotation=75,labelsize=8)
        ax.set_title(f"h{mesh} FEM")
    axes[0].set_ylabel("P+ − P− (процентные пункты)")
    fig.suptitle("A9: хиральный контраст на зеркальных парах, delta=tau0=0°")
    fig.tight_layout(); fig.savefig(figdir/"v15_fem_a9_chirality.png",dpi=220);fig.savefig(figdir/"v15_fem_a9_chirality.svg");plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, mesh in zip(axes, (50, 35)):
        for cfg in CFG_ORDER:
            for hor, marker in ((100, "o"), (300, "s")):
                rs=[r for r in summary if r["mesh_um"]==mesh and r["config"]==cfg and r["metric"]=="Delta_ell_all"
                    and r["horizon_rf"]==hor and r["delta_deg"]==0 and r["tau0_deg"]==0]
                if not rs: continue
                r=rs[0]
                ax.errorbar([f"{cfg} N{hor}"],[r["mean"]],
                    yerr=[[r["mean"]-r["ci95_low"]],[r["ci95_high"]-r["mean"]]],
                    fmt=marker,color=COLORS[cfg],capsize=3,ms=5)
        ax.axhline(0,color="black",lw=.8);ax.grid(alpha=.25);ax.tick_params(axis="x",rotation=75,labelsize=8)
        ax.set_title(f"h{mesh} FEM")
    axes[0].set_ylabel(r"Среднее $\Delta\ell$ до выхода или горизонта")
    fig.suptitle("A9: направленный перенос углового момента, delta=tau0=0°")
    fig.tight_layout();fig.savefig(figdir/"v15_fem_a9_angular_momentum.png",dpi=220);fig.savefig(figdir/"v15_fem_a9_angular_momentum.svg");plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,mesh in zip(axes,(50,35)):
        for cfg in CFG_ORDER:
            rr=[r for r in rows if int(r["mesh_um"])==mesh and r["config"]==cfg and float(r["delta_rad"])==0
                and float(r["tau0_rad"])==0 and int(r["horizon_rf"])==300 and r["status"]=="escaped" and r["exit_rf"] is not None]
            exits=np.sort(np.asarray([float(r["exit_rf"]) for r in rr]))
            if not len(exits):continue
            ax.step(exits,np.arange(1,len(exits)+1)/len(rr),where="post",color=COLORS[cfg],label=f"{cfg} (n={len(rr)})")
        ax.set_xlim(0,100);ax.set_ylim(0,1.02);ax.set_xlabel("Первое время выхода (RF-периодов)")
        ax.set_title(f"h{mesh} FEM");ax.grid(alpha=.25)
    axes[0].set_ylabel("Доля вышедших среди вышедших за 300 RF")
    axes[1].legend(fontsize=7)
    fig.suptitle("A9: эмпирические распределения времени первого выхода, phase=0°")
    fig.tight_layout();fig.savefig(figdir/"v15_fem_a9_exit_time.png",dpi=220);fig.savefig(figdir/"v15_fem_a9_exit_time.svg");plt.close(fig)

    print(json.dumps({"rows": val["rows"], "groups_present": val["groups_present"],
                      "unresolved": val["unresolved_rows"], "max_torque_residual": val["max_abs_torque_residual"],
                      "symmetry": sym}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
