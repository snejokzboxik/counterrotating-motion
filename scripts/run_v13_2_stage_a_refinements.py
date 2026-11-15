"""Targeted V13.2 mass refinements and initial-amplitude stratification."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability"
sys.path.insert(0, str(ROOT / "scripts"))
import run_v13_2_stage_a_mass_scan as scan  # noqa: E402


def stratified_states() -> tuple[np.ndarray, list[dict]]:
    radii = (0.05, 0.12, 0.22, 0.34)
    angles = (0.0, math.pi / 6, math.pi / 3, math.pi / 4)
    categories = ("zero", "radial_out", "radial_in", "tangential_positive_Lz", "tangential_negative_Lz")
    states, meta = [], []
    speed = 0.03
    for ir, (radius, theta) in enumerate(zip(radii, angles)):
        ex, ey = math.cos(theta), math.sin(theta)
        vx = {
            "zero": (0., 0.),
            "radial_out": (speed*ex, speed*ey),
            "radial_in": (-speed*ex, -speed*ey),
            "tangential_positive_Lz": (-speed*ey, speed*ex),
            "tangential_negative_Lz": (speed*ey, -speed*ex),
        }
        x, y = radius*ex, radius*ey
        for category in categories:
            ux, uy = vx[category]
            states.append((x, y, ux, uy))
            meta.append({"index": len(states)-1, "rho0": radius, "theta0_rad": theta,
                         "velocity_class": category, "speed0": math.hypot(ux, uy),
                         "angular_momentum_z": x*uy-y*ux,
                         "kinetic_energy_dimensionless": 0.5*(ux*ux+uy*uy),
                         "stratum": f"rho{ir}_{category}"})
    arr = np.asarray(states, dtype=float)
    # All positions are inside the V13.1 common rectangular entrance domain.
    if np.max(np.abs(arr[:, :2])) > 0.25 + 1e-12:
        raise RuntimeError("A stratified initial point lies outside |x|,|y|<=0.25")
    return arr, meta


def write_initials(path: Path, states: np.ndarray, meta: list[dict] | None = None) -> None:
    rows = []
    for i, s in enumerate(states):
        row = {"index": i, "x0_rho": float(s[0]), "y0_rho": float(s[1]),
               "vx0_rho_per_tau": float(s[2]), "vy0_rho_per_tau": float(s[3]),
               "rho0": float(np.hypot(s[0], s[1])), "speed0": float(np.hypot(s[2], s[3])),
               "angular_momentum_z": float(s[0]*s[3]-s[1]*s[2]),
               "kinetic_energy_dimensionless": float(0.5*(s[2]**2+s[3]**2))}
        if meta:
            row.update(meta[i])
        rows.append(row)
    scan.write_csv(path, rows)


def run_set(label: str, mus: list[float], states: np.ndarray, workers: int,
            eta: float = -0.05, metadata: dict | None = None) -> list[dict]:
    checkpoint = OUT / f"{label}_checkpoint.jsonl"
    done = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()] if checkpoint.exists() else []
    keys = {scan.stable_key(r) for r in done}
    tasks = []
    for mu in mus:
        for phase in scan.PHASES:
            for model, model_eta in (("Q", 0.0), ("SO", eta), ("RO", eta)):
                for i, state in enumerate(states):
                    key = (round(float(mu), 8), round(float(phase), 12), model,
                           round(float(model_eta), 8), i)
                    if key not in keys:
                        tasks.append((float(mu), float(phase), model, model_eta, i, state.tolist()))
    manifest = {"label": label, "mu": mus, "eta": eta, "n_states": len(states),
                "phases_rad": list(scan.PHASES), "models": [["Q", 0.0], ["SO", eta], ["RO", eta]],
                "horizon_rf_periods": 100, "initial_state_units": "rho and drho/dtau",
                "integrator": {"method": "DOP853", "rtol": scan.RTOL, "atol": scan.ATOL,
                               "max_step_tau": scan.MAX_STEP},
                "checkpoint": checkpoint.name, **(metadata or {})}
    (OUT / f"{label}_parameters.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"{label}: {len(tasks)} new trajectories, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, checkpoint.open("a", encoding="utf-8") as ck:
        futures = [pool.submit(scan.advance, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            done.append(row)
            ck.write(json.dumps(row, separators=(",", ":"))+"\n")
            ck.flush()
            if n % 128 == 0 or n == len(futures):
                print(f"{label}: completed {n}/{len(futures)}", flush=True)
    rows = sorted(done, key=lambda r: (float(r["mu"]), float(r["phase_rad"]), r["model"], int(r["index"])))
    scan.write_csv(OUT / f"{label}_trajectory_results.csv", rows)
    scan.write_csv(OUT / f"{label}_retention_summary.csv", scan.summarize(rows))
    return rows


def make_refinement_figures() -> None:
    import matplotlib.pyplot as plt
    from three_phase_trap.mass_filter_v13 import linear_status

    base = list(csv.DictReader((OUT / "stage_a_retention_summary.csv").open(encoding="utf-8")))
    fine = list(csv.DictReader((OUT / "stage_a_fine_retention_summary.csv").open(encoding="utf-8")))
    combined = base + fine
    floquet_rows=[]
    for mu in sorted({round(float(r["mu"]),8) for r in combined}):
        alpha,beta=scan.ALPHA0/mu,scan.BETA0/mu
        q=linear_status(alpha,beta,tolerance=1e-8)
        floquet_rows.append({"mu":mu,"alpha":alpha,"beta":beta,"kappa_eta_minus_0p05":-.05*beta,
                             "linear_status_both_axes":q["status"],"trace_x":q["trace_x"],"trace_y":q["trace_y"],
                             "det_x":q["det_x"],"det_y":q["det_y"],"margin_x":q["margin_x"],"margin_y":q["margin_y"],
                             "method":"DOP853 one-period monodromy, rtol=2e-11, atol=2e-13"})
    scan.write_csv(OUT/"stage_a_floquet_mass_scan.csv",floquet_rows)
    from scipy.optimize import brentq
    boundaries=[]
    for lo,hi in ((.97,.98),(1.08,1.09)):
        root=brentq(lambda mu:min(linear_status(scan.ALPHA0/mu,scan.BETA0/mu)["margin_x"],
                                 linear_status(scan.ALPHA0/mu,scan.BETA0/mu)["margin_y"]),lo,hi,xtol=1e-12)
        alpha,beta=scan.ALPHA0/root,scan.BETA0/root
        q=linear_status(alpha,beta,tolerance=1e-8)
        axis="x" if abs(q["margin_x"])<abs(q["margin_y"]) else "y"
        boundaries.append({"mu_boundary":root,"alpha":alpha,"beta":beta,"active_axis":axis,
                           "trace_x":q["trace_x"],"trace_y":q["trace_y"],"margin_x":q["margin_x"],
                           "margin_y":q["margin_y"],"status":"boundary_or_unresolved"})
    scan.write_csv(OUT/"stage_a_floquet_mass_boundaries.csv",boundaries)
    colors = {"Q":"#202020","SO":"#d97706","RO":"#2563eb"}
    labels = {"Q":"Q","SO":"Q + SO, η = −0.05","RO":"Q + RO, η = −0.05"}
    for zoom, name in ((False,"stage_a_mass_scan_refined"),(True,"stage_a_mass_scan_zoom")):
        fig, axes = plt.subplots(1,2,figsize=(10.4,4.2),sharey=True)
        for ax,phase in zip(axes,(0.,90.)):
            for model in ("Q","SO","RO"):
                rs=sorted([r for r in combined if r["model"]==model and float(r["phase_deg"])==phase
                           and (not zoom or 1.0<=float(r["mu"])<=1.07)],key=lambda r:float(r["mu"]))
                x=np.array([float(r["mu"]) for r in rs]);p=np.array([float(r["retention_resolved"]) for r in rs])
                lo=np.array([float(r["retention_wilson95_low"]) for r in rs])
                hi=np.array([float(r["retention_wilson95_high"]) for r in rs])
                ax.errorbar(x,p,yerr=np.vstack((p-lo,hi-p)),fmt="o",ms=4,capsize=2,lw=.9,
                            color=colors[model],label=labels[model])
            ax.set_title(f"Начальная фаза {phase:.0f}°")
            ax.set_xlabel("μ=(M/Q)/(M₀/Q₀)");ax.set_ylim(-.04,1.04);ax.grid(alpha=.25)
            if zoom:ax.set_xlim(.995,1.065)
        axes[0].set_ylabel("Удержание за 100 RF периодов")
        axes[1].legend(frameon=False,fontsize=8,loc="best")
        fig.suptitle("Идеальное гармоническое поле; N=32 общих Sobol состояний; показаны только вычисленные μ")
        fig.tight_layout()
        for ext in ("png","pdf","svg"):fig.savefig(OUT/f"{name}.{ext}",dpi=220 if ext=="png" else None)
        plt.close(fig)

    rep=list(csv.DictReader((OUT/"stage_a_replica128_mu104_retention_summary.csv").open(encoding="utf-8")))
    fig,ax=plt.subplots(figsize=(7.2,4.0))
    xpos={("Q",0.):0,("SO",0.):1,("RO",0.):2,("Q",90.):4,("SO",90.):5,("RO",90.):6}
    for r in rep:
        x=xpos[(r["model"],float(r["phase_deg"]))];p=float(r["retention_resolved"])
        lo,hi=float(r["retention_wilson95_low"]),float(r["retention_wilson95_high"])
        ax.errorbar(x,p,yerr=[[p-lo],[hi-p]],fmt="o",ms=6,capsize=3,color=colors[r["model"]])
        ax.text(x,p+.05,f"{r['n_retained']}/128",ha="center",fontsize=8)
    ax.set_xticks([0,1,2,4,5,6],["Q","SO","RO","Q","SO","RO"])
    ax.axvline(3,color="0.7",lw=1);ax.set_xlim(-.7,6.7);ax.set_ylim(0,1.13)
    ax.set_ylabel("Удержание, 100 RF периодов")
    ax.set_title("μ=1.04; независимый Sobol N=128; интервалы Wilson 95%")
    ax.grid(axis="y",alpha=.25)
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"stage_a_replica128_mu104.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)

    initials={int(r["index"]):r for r in csv.DictReader((OUT/"stage_a_amplitude_strata_initial_conditions.csv").open(encoding="utf-8"))}
    trajectories=list(csv.DictReader((OUT/"stage_a_amplitude_strata_trajectory_results.csv").open(encoding="utf-8")))
    categories=["zero","radial_out","radial_in","tangential_positive_Lz","tangential_negative_Lz"]
    catlabels=["0","radial +","radial −","Lz +","Lz −"]
    radius_order=[.05,.12,.22,.34]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True)
    for ax,model in zip(axes,("Q","SO","RO")):
        arr=np.full((4,5),np.nan);escaped=np.zeros((4,5),bool)
        for r in trajectories:
            if float(r["mu"])!=1.04 or float(r["phase_deg"])!=0 or r["model"]!=model:continue
            ic=initials[int(r["index"])]
            iy=radius_order.index(round(float(ic["rho0"]),2));ix=categories.index(ic["velocity_class"])
            arr[iy,ix]=min(float(r["max_rho_100"]),1.0);escaped[iy,ix]=r["status_100"]=="escaped"
        im=ax.imshow(arr,origin="lower",aspect="auto",vmin=0,vmax=1,cmap="viridis")
        for iy in range(4):
            for ix in range(5):
                if escaped[iy,ix]:ax.text(ix,iy,"×",ha="center",va="center",color="white",fontsize=15,weight="bold")
        ax.set_title(model);ax.set_xticks(range(5),catlabels,rotation=25,ha="right",fontsize=8)
        ax.set_yticks(range(4),[f"{r:.2f}" for r in radius_order]);ax.set_xlabel("Класс скорости")
    axes[0].set_ylabel("Начальный радиус ρ₀")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Максимум ρ за 100 периодов",shrink=.82)
    fig.suptitle("Контролируемые старты при μ=1.04, фаза 0°; × — выход; одна траектория на клетку")
    fig.subplots_adjust(left=.08,right=.9,bottom=.19,top=.82,wspace=.25)
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"stage_a_amplitude_strata_mu104.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run", choices=("fine", "replica", "strata", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.run in ("fine", "all"):
        initials = scan.load_initials()
        write_initials(OUT / "stage_a_fine_initial_conditions.csv", initials)
        run_set("stage_a_fine", [1.025, 1.035, 1.045, 1.055], initials, args.workers,
                metadata={"purpose": "0.005 mass spacing between completed 0.01 pilot-grid nodes",
                          "source_initial_conditions": "V13.1 seed131 common 32-state Sobol set"})
    if args.run in ("replica", "all"):
        independent = (qmc.Sobol(4, scramble=True, seed=1322).random_base2(7)-0.5) * np.array([.5,.5,.08,.08])
        write_initials(OUT / "stage_a_replica128_initial_conditions.csv", independent)
        run_set("stage_a_replica128_mu104", [1.04], independent, args.workers,
                metadata={"purpose": "independent scrambled Sobol uncertainty check at pilot trough candidate",
                          "sobol_seed": 1322, "scramble": True,
                          "domain": {"x0_y0": [-.25,.25], "vx0_vy0": [-.04,.04]}})
    if args.run in ("strata", "all"):
        states, meta = stratified_states()
        write_initials(OUT / "stage_a_amplitude_strata_initial_conditions.csv", states, meta)
        run_set("stage_a_amplitude_strata", [1.0, 1.04, 1.10], states, args.workers,
                metadata={"purpose": "controlled radius/speed/Lz response comparison",
                          "radial_levels_rho": [0.05,0.12,0.22,0.34],
                          "velocity_classes": ["zero", "radial_out", "radial_in",
                                              "tangential_positive_Lz", "tangential_negative_Lz"],
                          "nonzero_speed_rho_per_tau": 0.03})
    if args.run == "all":
        make_refinement_figures()


if __name__ == "__main__":
    main()
