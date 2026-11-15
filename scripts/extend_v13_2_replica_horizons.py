"""Continue the independent V13.2 N=128 pilot states without resetting RF phase."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability"
SOURCE = OUT / "stage_a_replica128_mu104_checkpoint.jsonl"
RTOL, ATOL = 2e-10, 2e-12
MAX_STEP = 2 * math.pi / 64


def continuation(task: tuple) -> dict:
    row, start_period, end_period = task
    from three_phase_trap.mass_filter_v13 import acceleration

    out = dict(row)
    previous = start_period
    prior_status = row[f"status_{start_period}"]
    prior_exit = row.get(f"exit_periods_{start_period}")
    prior_max = float(row[f"max_rho_{start_period}"])
    prior_state = row.get(f"state_{start_period}")
    if prior_status != "retained":
        out[f"status_{end_period}"] = prior_status
        out[f"exit_periods_{end_period}"] = prior_exit
        out[f"max_rho_{end_period}"] = prior_max
        out[f"state_{end_period}"] = prior_state
        out[f"nfev_{end_period}"] = 0
        return out

    mu, phase, model = float(row["mu"]), float(row["phase_rad"]), row["model"]
    alpha, beta, kappa = float(row["alpha"]), float(row["beta"]), float(row["kappa"])
    standing = model == "SO"
    y0 = np.asarray(prior_state, dtype=float)

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        return np.r_[y[2:], acceleration(y[0], y[1], tau + phase, kappa, beta, alpha,
                                         standing_octupole=standing)]

    def aperture(tau: float, y: np.ndarray) -> float:
        return float(np.hypot(y[0], y[1]) - 1.0)

    aperture.terminal = True
    aperture.direction = 1
    try:
        sol = solve_ivp(rhs, (2*math.pi*start_period, 2*math.pi*end_period), y0,
                        method="DOP853", rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
                        events=aperture, dense_output=True)
        if not sol.success:
            status, exit_period, max_rho = "unresolved", None, prior_max
        else:
            radius = np.hypot(sol.y[0], sol.y[1])
            max_rho = float(max(prior_max, np.max(radius)))
            if len(sol.t) > 1:
                mid = sol.sol((sol.t[:-1] + sol.t[1:]) / 2)
                max_rho = float(max(max_rho, np.max(np.hypot(mid[0], mid[1]))))
            if len(sol.t_events[0]):
                status, exit_period = "escaped", float(sol.t_events[0][0] / (2*math.pi))
                max_rho = max(max_rho, 1.0)
            else:
                status, exit_period = "retained", None
        out[f"status_{end_period}"] = status
        out[f"exit_periods_{end_period}"] = exit_period if exit_period is not None else prior_exit
        out[f"max_rho_{end_period}"] = max_rho
        out[f"state_{end_period}"] = [float(v) for v in sol.y[:, -1]]
        out[f"nfev_{end_period}"] = int(sol.nfev)
        out[f"continuation_integrator"] = "DOP853 rtol=2e-10 atol=2e-12 max_step=2pi/64"
        return out
    except Exception as exc:
        out[f"status_{end_period}"] = "unresolved"
        out[f"exit_periods_{end_period}"] = prior_exit
        out[f"max_rho_{end_period}"] = prior_max
        out[f"state_{end_period}"] = None
        out[f"nfev_{end_period}"] = None
        out[f"error_{end_period}"] = f"{type(exc).__name__}:{exc}"
        return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"))+"\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run_stage(rows: list[dict], start: int, end: int, workers: int) -> list[dict]:
    ck = OUT / f"stage_a_replica128_mu104_checkpoint_{end}.jsonl"
    done = read_jsonl(ck) if ck.exists() else []
    completed = {(int(r["index"]), float(r["phase_rad"]), r["model"], float(r["eta"])) for r in done}
    tasks = []
    for row in rows:
        key = (int(row["index"]), float(row["phase_rad"]), row["model"], float(row["eta"]))
        if key in completed:
            continue
        # Endpoints for a new segment come from the preceding horizon's checkpoint.
        tasks.append((row, start, end))
    print(f"Continue {start}->{end} RF periods: {len(tasks)} tasks, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, ck.open("a", encoding="utf-8") as f:
        futures = [pool.submit(continuation, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            done.append(row)
            f.write(json.dumps(row, separators=(",", ":"))+"\n")
            f.flush()
            if n % 128 == 0 or n == len(futures):
                print(f"{end}-period continuation completed {n}/{len(futures)}", flush=True)
    return done


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k/n
    den = 1+z*z/n
    c = (p+z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0., c-h), min(1., c+h)


def long_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for horizon in (100, 300, 1000):
        col = f"status_{horizon}"
        for phase in (0., math.pi/2):
            for model, eta in (("Q",0.),("SO",-.05),("RO",-.05)):
                rr = [r for r in rows if abs(float(r["phase_rad"])-phase)<1e-10 and
                      r["model"]==model and float(r["eta"])==eta]
                resolved = [r for r in rr if r[col] in ("retained","escaped")]
                k = sum(r[col]=="retained" for r in resolved)
                lo, hi = wilson(k,len(resolved))
                q = {int(r["index"]):r for r in rows if abs(float(r["phase_rad"])-phase)<1e-10 and r["model"]=="Q"}
                paired = [(int(r[col]=="retained")-int(q[int(r["index"])][col]=="retained"),r)
                          for r in resolved if int(r["index"]) in q and q[int(r["index"])][col] in ("retained","escaped")]
                d = np.asarray([x for x,_ in paired],float)
                pd = float(d.mean()) if len(d) else float("nan")
                se = float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else float("nan")
                summary.append({"horizon_rf":horizon,"phase_deg":float(np.degrees(phase)),"model":model,"eta":eta,
                    "n_total":len(rr),"n_resolved":len(resolved),"n_unresolved":len(rr)-len(resolved),"n_retained":k,
                    "retention":k/len(resolved) if resolved else float("nan"),"wilson95_low":lo,"wilson95_high":hi,
                    "retention_bound_all_n_low":k/len(rr) if rr else float("nan"),
                    "retention_bound_all_n_high":(k+len(rr)-len(resolved))/len(rr) if rr else float("nan"),
                    "paired_n_vs_Q":len(d),"paired_diff_vs_Q":pd,
                    "paired_diff_vs_Q_ci95_low":pd-1.96*se if np.isfinite(se) else float("nan"),
                    "paired_diff_vs_Q_ci95_high":pd+1.96*se if np.isfinite(se) else float("nan"),
                    "paired_gains_vs_Q":int(np.sum(d==1)),"paired_losses_vs_Q":int(np.sum(d==-1))})
    # Add RO-SO paired effect and uncertainty to the RO summaries.
    for row in summary:
        if row["model"]!="RO":
            continue
        col=f"status_{row['horizon_rf']}"
        phase=math.radians(row["phase_deg"])
        ro=[r for r in rows if r["model"]=="RO" and abs(float(r["phase_rad"])-phase)<1e-10]
        so={int(r["index"]):r for r in rows if r["model"]=="SO" and abs(float(r["phase_rad"])-phase)<1e-10}
        d=np.asarray([int(r[col]=="retained")-int(so[int(r["index"])][col]=="retained") for r in ro
                      if int(r["index"]) in so and r[col] in ("retained","escaped") and so[int(r["index"])][col] in ("retained","escaped")],float)
        pd=float(d.mean()) if len(d) else float("nan")
        se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else float("nan")
        row.update({"paired_n_RO_minus_SO":len(d),"paired_diff_RO_minus_SO":pd,
                    "paired_diff_RO_minus_SO_ci95_low":pd-1.96*se if np.isfinite(se) else float("nan"),
                    "paired_diff_RO_minus_SO_ci95_high":pd+1.96*se if np.isfinite(se) else float("nan"),
                    "RO_SO_gains":int(np.sum(d==1)),"RO_SO_losses":int(np.sum(d==-1))})
    return summary


def survival_plot(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    colors={"Q":"#202020","SO":"#d97706","RO":"#2563eb"}
    labels={"Q":"Q","SO":"Q + SO, η = −0.05","RO":"Q + RO, η = −0.05"}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),sharey=True)
    for ax,phase in zip(axes,(0.,math.pi/2)):
        for model in ("Q","SO","RO"):
            group=[r for r in rows if abs(float(r["phase_rad"])-phase)<1e-10 and r["model"]==model]
            event_times=np.asarray([float(r["exit_periods_1000"]) for r in group
                                    if r["status_1000"]=="escaped" and r.get("exit_periods_1000") not in (None,"")])
            surv=1.0; x=[0.0]; y=[1.0]
            for t in sorted(set(event_times.tolist())):
                at_risk=sum((r["status_1000"]=="escaped" and r.get("exit_periods_1000") not in (None,"")
                             and float(r["exit_periods_1000"])>=t) or
                            (r["status_1000"]=="retained") for r in group)
                events=int(np.sum(event_times==t))
                if at_risk:
                    surv*=1-events/at_risk
                x.append(t); y.append(surv)
            x.append(1000.0); y.append(surv)
            ax.step(x,y,where="post",color=colors[model],lw=1.7,label=labels[model])
        ax.set_title(f"Начальная фаза {np.degrees(phase):.0f}°")
        ax.set_xlabel("RF-периоды до первого выхода через ρ=1")
        ax.set_xlim(0,1000); ax.set_ylim(-.03,1.03); ax.grid(alpha=.25)
    axes[0].set_ylabel("Доля без выхода (эмпирическая Kaplan–Meier оценка)")
    axes[1].legend(frameon=False,fontsize=8,loc="best")
    fig.suptitle("μ=1.04, независимый Sobol N=128; удержанные на 1000 периодах цензурированы справа")
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig(OUT/f"stage_a_exit_survival_mu104.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--workers",type=int,default=8); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(SOURCE)
    # Resumed rows are keyed by seed/index, RF phase and model. The original 100-period row is kept as source.
    h300=run_stage(rows,100,300,args.workers)
    h300.sort(key=lambda r:(float(r["phase_rad"]),r["model"],int(r["index"])))
    write_csv(OUT/"stage_a_replica128_mu104_horizon300.csv",h300)
    write_jsonl(OUT/"stage_a_replica128_mu104_checkpoint_300.jsonl",h300)
    h1000=run_stage(h300,300,1000,args.workers)
    h1000.sort(key=lambda r:(float(r["phase_rad"]),r["model"],int(r["index"])))
    write_csv(OUT/"stage_a_replica128_mu104_horizon1000.csv",h1000)
    write_jsonl(OUT/"stage_a_replica128_mu104_checkpoint_1000.jsonl",h1000)
    write_csv(OUT/"stage_a_replica128_mu104_long_summary.csv",long_summary(h1000))
    survival_plot(h1000)
    (OUT/"stage_a_replica128_mu104_long_parameters.json").write_text(json.dumps({
        "mass_mu":1.04,"initial_sample":"independent scrambled Sobol seed 1322, N=128",
        "horizons_rf_periods":[100,300,1000],"continuation":"state and RF phase are continued exactly across intervals",
        "integrator":{"method":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP},
        "field":"ideal three-component V13 model; aperture rho=1",
        "meaning":"finite-time transverse survival, not finite-length transmission"},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
