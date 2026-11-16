"""Build physical masks, exact Floquet references and sampled-data figures."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
V13_1 = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
ALPHAS = np.linspace(-0.05, 0.05, 17)
BETAS = np.linspace(0.05, 0.30, 21)
ETAS = (-0.20, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.20)
E_CHARGE = 1.602176634e-19
ATOMIC_MASS = 1.66053906660e-27
M0_TH = 3200.04652261404
R_M = 0.008
F_HZ = 100_000.0
C2 = D2 = 0.7718255558274212
C4 = 0.5413395699439091
U_MAX_V = 250.0


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"))
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def voltage_and_feasibility() -> list[dict]:
    electrodes = [row for row in read_csv(V13_1 / "electrode_voltage_peaks.csv")
                  if row["scheme"] == "Q"]
    basis = [{key: float(row[key]) for key in ("electrode", "theta_deg", "C4", "S4", "Q2")}
             for row in electrodes]
    m_kg = M0_TH * ATOMIC_MASS
    omega = 2.0 * math.pi * F_HZ
    scale = m_kg * omega**2 * R_M**2 / E_CHARGE
    out = []
    for ai, alpha in enumerate(ALPHAS):
        for bi, beta in enumerate(BETAS):
            for ei, eta in enumerate(ETAS):
                model = "Q" if eta == 0.0 else "Q+RO"
                kappa = 0.0 if eta == 0.0 else float(eta*beta)
                u2, vdc, u4 = beta*scale/C2, alpha*scale/D2, kappa*scale/C4
                peaks = []
                for electrode in basis:
                    q2 = electrode["Q2"]
                    rf_cos = u2*q2 + u4*electrode["C4"]
                    rf_sin = u4*electrode["S4"]
                    dc = vdc*q2
                    peaks.append(abs(dc) + math.hypot(rf_cos, rf_sin))
                vmax = max(peaks)
                out.append({
                    "alpha_i": ai, "beta_i": bi, "eta_i": ei,
                    "alpha": float(alpha), "beta": float(beta), "eta": float(eta),
                    "kappa": kappa, "model": model,
                    "U2_V": float(u2), "VDC_V": float(vdc), "U4_V": float(u4),
                    "electrode_peak_abs_V": peaks, "max_electrode_peak_abs_V": vmax,
                    "within_250V": vmax <= U_MAX_V,
                    "voltage_formula": "abs(DC_j)+hypot(RF_cos_j,RF_sin_j)",
                })
    return out


def floquet_reference() -> list[dict]:
    from three_phase_trap.mass_filter_v13 import linear_status
    out = []
    for ai, alpha in enumerate(ALPHAS):
        for bi, beta in enumerate(BETAS):
            row = linear_status(float(alpha), float(beta), tolerance=1e-8,
                                rtol=2e-11, atol=2e-13)
            out.append({"alpha_i": ai, "beta_i": bi,
                        "alpha": float(alpha), "beta": float(beta),
                        **row, "model_relation": "identical for Q and Q+RO at center"})
    return out


def floquet_boundaries() -> list[dict]:
    from scipy.optimize import brentq
    from three_phase_trap.mass_filter_v13 import monodromy
    out=[]
    scan=np.linspace(-0.05,0.05,81)
    for bi,beta in enumerate(BETAS):
        def margins(alpha):
            mx=monodromy(float(alpha),float(beta),"x",rtol=2e-11,atol=2e-13)
            my=monodromy(float(alpha),float(beta),"y",rtol=2e-11,atol=2e-13)
            tx,ty=float(np.trace(mx)),float(np.trace(my))
            return min(2-abs(tx),2-abs(ty)),tx,ty,float(np.linalg.det(mx)),float(np.linalg.det(my))
        vals=[margins(a)[0] for a in scan]
        roots=[]
        for i in range(len(scan)-1):
            if vals[i]*vals[i+1] < 0:
                root=float(brentq(lambda a:margins(a)[0],float(scan[i]),float(scan[i+1]),
                                  xtol=1e-12,rtol=1e-12))
                if not roots or abs(root-roots[-1])>1e-9: roots.append(root)
        for branch,root in enumerate(roots):
            margin,tx,ty,detx,dety=margins(root)
            axis="x" if 2-abs(tx)<=2-abs(ty) else "y"
            tr=tx if axis=="x" else ty
            out.append({"beta_i":bi,"beta":float(beta),"branch_index_alpha_order":branch,
                        "alpha_boundary":root,"active_axis":axis,
                        "active_trace":tr,"limit_trace":2.0 if tr>=0 else -2.0,
                        "margin_at_root":margin,"det_x":detx,"det_y":dety,
                        "method":"bracketed root of exact monodromy trace-margin"})
    return out


def enrich_summary(summary: list[dict], feasibility: list[dict], floquet: list[dict]) -> list[dict]:
    fidx = {(int(row["alpha_i"]), int(row["beta_i"])): row for row in floquet}
    vidx = {(int(row["alpha_i"]), int(row["beta_i"]), int(row["eta_i"])): row
            for row in feasibility}
    out = []
    for row in summary:
        ai, bi, ei = int(row["alpha_i"]), int(row["beta_i"]), int(row["eta_i"])
        ff = fidx[(ai, bi)]
        vv = vidx[(ai, bi, ei)]
        out.append({**row,
                    "eta": vv["eta"],
                    "floquet_status": ff["status"],
                    "floquet_margin_x": ff["margin_x"],
                    "floquet_margin_y": ff["margin_y"],
                    "det_x": ff["det_x"], "det_y": ff["det_y"],
                    "U2_V": vv["U2_V"], "VDC_V": vv["VDC_V"],
                    "U4_V": vv["U4_V"],
                    "max_electrode_peak_abs_V": vv["max_electrode_peak_abs_V"],
                    "within_250V": vv["within_250V"]})
    return out


def retention_figures(rows: list[dict], boundaries: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    rows_by = {(int(row["alpha_i"]), int(row["beta_i"]), int(row["eta_i"]), int(row["phase_i"])): row
               for row in rows}
    phases = [(0, 0.0), (1, 90.0)]
    for phase_i, phase_deg in phases:
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), sharex=True, sharey=True)
        for ei, eta in enumerate(ETAS):
            ax = axes.flat[ei]
            xs, ys, cs, bad_x, bad_y = [], [], [], [], []
            for ai, alpha in enumerate(ALPHAS):
                for bi, beta in enumerate(BETAS):
                    key_eta = 4 if eta == 0 else ei
                    q = rows_by.get((ai, bi, 4, phase_i))
                    ro = rows_by.get((ai, bi, key_eta, phase_i))
                    if q is None or ro is None:
                        continue
                    target = q if eta == 0 else ro
                    physical = bool(target["within_250V"])
                    p = float(target["P_ret"])
                    if np.isfinite(p):
                        xs.append(float(beta)); ys.append(float(alpha)); cs.append(p)
                        if not physical:
                            bad_x.append(float(beta)); bad_y.append(float(alpha))
            if xs:
                ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=0, vmax=1,
                           marker="s", s=27, linewidths=0)
            if bad_x:
                ax.scatter(bad_x, bad_y, marker="x", s=10, c="#dc2626", linewidths=.5)
            bb=[float(r["beta"]) for r in boundaries]; ba=[float(r["alpha_boundary"]) for r in boundaries]
            if bb: ax.scatter(bb,ba,marker=".",s=13,c="black",alpha=.7)
            ax.set_title(f"η={eta:+.3f}  (κ=ηβ)")
            ax.grid(alpha=.16)
            if ei % 3 == 0:
                ax.set_ylabel("α")
            if ei >= 6:
                ax.set_xlabel("β")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=.025, pad=.02,
                     label=f"P_ret, N=8, τ₀={phase_deg:.0f}°")
        fig.suptitle(f"V13.3 конечновременное удержание Q / Q+RO; τ₀={phase_deg:.0f}°; фактически вычисленные узлы", y=.995)
        fig.savefig(OUT / f"retention_slices_phase_{phase_deg:.0f}deg.png", dpi=220, bbox_inches="tight")
        fig.savefig(OUT / f"retention_slices_phase_{phase_deg:.0f}deg.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"retention_slices_phase_{phase_deg:.0f}deg.svg", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), sharex=True, sharey=True)
        for ei, eta in enumerate(ETAS):
            ax = axes.flat[ei]
            xs, ys, cs, bad_x, bad_y = [], [], [], [], []
            key_eta = 4 if eta == 0 else ei
            for ai, alpha in enumerate(ALPHAS):
                for bi, beta in enumerate(BETAS):
                    q = rows_by.get((ai, bi, 4, phase_i))
                    ro = rows_by.get((ai, bi, key_eta, phase_i))
                    if q is not None and ro is not None and np.isfinite(float(ro["P_ret"])) and np.isfinite(float(q["P_ret"])):
                        xs.append(beta); ys.append(alpha)
                        cs.append(float(ro["P_ret"])-float(q["P_ret"]))
                        if not bool(ro["within_250V"]) or not bool(q["within_250V"]):
                            bad_x.append(beta); bad_y.append(alpha)
            if xs:
                sc = ax.scatter(xs, ys, c=cs, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
                                marker="s", s=29, linewidths=0)
            if bad_x:
                ax.scatter(bad_x, bad_y, marker="x", s=10, c="#dc2626", linewidths=.5)
            bb=[float(r["beta"]) for r in boundaries]; ba=[float(r["alpha_boundary"]) for r in boundaries]
            if bb: ax.scatter(bb,ba,marker=".",s=13,c="black",alpha=.7)
            ax.set_title(f"η={eta:+.3f}")
            ax.grid(alpha=.16)
            if ei % 3 == 0: ax.set_ylabel("α")
            if ei >= 6: ax.set_xlabel("β")
        fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=.025, pad=.02,
                     label="парная разность P(Q+RO)−P(Q)")
        fig.suptitle(f"Нелинейное отличие Q+RO от Q, τ₀={phase_deg:.0f}°; только вычисленные точки", y=.995)
        fig.savefig(OUT / f"ro_minus_q_phase_{phase_deg:.0f}deg.png", dpi=220, bbox_inches="tight")
        fig.savefig(OUT / f"ro_minus_q_phase_{phase_deg:.0f}deg.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"ro_minus_q_phase_{phase_deg:.0f}deg.svg", bbox_inches="tight")
        plt.close(fig)

    # The full 3D atlas shows only measured points; no surface interpolation is used.
    fig = plt.figure(figsize=(13.5, 6.3))
    for panel, phase_i in enumerate((0, 1), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        xx, yy, zz, cc, marker = [], [], [], [], []
        for row in rows:
            if int(row["phase_i"]) != phase_i or not np.isfinite(float(row["P_ret"])):
                continue
            xx.append(float(row["alpha"])); yy.append(float(row["beta"])); zz.append(float(row["kappa"]))
            cc.append(float(row["P_ret"]))
            marker.append(row["model"])
        for model, symbol in (("Q", "o"), ("RO", "^")):
            idx = [i for i, m in enumerate(marker) if m == model]
            sc = ax.scatter(np.array(xx)[idx], np.array(yy)[idx], np.array(zz)[idx],
                            c=np.array(cc)[idx], cmap="viridis", vmin=0, vmax=1,
                            marker=symbol, s=10, alpha=.8, label=model, linewidths=0)
        ax.set_xlabel("α"); ax.set_ylabel("β"); ax.set_zlabel("κ")
        ax.set_title(f"τ₀={90*phase_i}°; sampled nodes only")
        ax.legend(loc="upper left")
    fig.colorbar(sc, ax=fig.axes, shrink=.62, pad=.05, label="P_ret, 8 общих состояний")
    fig.suptitle("Трёхпараметрический атлас конечновременного удержания Q и Q+RO", y=.98)
    fig.savefig(OUT / "three_parameter_atlas_sampled.png", dpi=230, bbox_inches="tight")
    fig.savefig(OUT / "three_parameter_atlas_sampled.pdf", bbox_inches="tight")
    fig.savefig(OUT / "three_parameter_atlas_sampled.svg", bbox_inches="tight")
    plt.close(fig)

    # RF injection-phase dependence is displayed separately from relative RO phase delta.
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), sharex=True, sharey=True)
    for ei, eta in enumerate(ETAS):
        ax = axes.flat[ei]
        xs, ys, cs, bad_x, bad_y = [], [], [], [], []
        for ai, alpha in enumerate(ALPHAS):
            for bi, beta in enumerate(BETAS):
                key_eta = 4 if eta == 0 else ei
                a = rows_by.get((ai,bi,key_eta,0)); b = rows_by.get((ai,bi,key_eta,1))
                if a is None or b is None or not np.isfinite(float(a["P_ret"])) or not np.isfinite(float(b["P_ret"])):
                    continue
                xs.append(beta); ys.append(alpha); cs.append(float(b["P_ret"])-float(a["P_ret"]))
                if not bool(a["within_250V"]) or not bool(b["within_250V"]):
                    bad_x.append(beta); bad_y.append(alpha)
        if xs:
            sc = ax.scatter(xs,ys,c=cs,cmap="coolwarm",norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),
                            marker="s",s=29,linewidths=0)
        bb=[float(r["beta"]) for r in boundaries]; ba=[float(r["alpha_boundary"]) for r in boundaries]
        if bb: ax.scatter(bb,ba,marker=".",s=13,c="black",alpha=.7)
        if bad_x: ax.scatter(bad_x,bad_y,marker="x",s=10,c="#dc2626",linewidths=.5)
        ax.set_title(f"η={eta:+.3f}"); ax.grid(alpha=.16)
        if ei%3==0: ax.set_ylabel("α")
        if ei>=6: ax.set_xlabel("β")
    fig.colorbar(sc,ax=axes.ravel().tolist(),fraction=.025,pad=.02,
                 label="P_ret(τ₀=90°) − P_ret(τ₀=0°)")
    fig.suptitle("Чувствительность нелинейного удержания к фазе инжекции (δ=0 фиксирована)",y=.995)
    for ext in ("png","pdf","svg"):
        fig.savefig(OUT/f"injection_phase_difference.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)

    # Exit-time maps: only actually escaped cases receive a color.
    for phase_i, phase_deg in phases:
        fig, axes = plt.subplots(3,3,figsize=(13.5,10),sharex=True,sharey=True)
        for ei,eta in enumerate(ETAS):
            ax=axes.flat[ei]; xs=[];ys=[];cs=[]
            key_eta=4 if eta==0 else ei
            for ai,alpha in enumerate(ALPHAS):
                for bi,beta in enumerate(BETAS):
                    row=rows_by.get((ai,bi,key_eta,phase_i))
                    if row is None or not np.isfinite(float(row["median_exit_periods"])): continue
                    xs.append(beta);ys.append(alpha);cs.append(float(row["median_exit_periods"]))
            if xs:
                sc=ax.scatter(xs,ys,c=cs,cmap="magma_r",vmin=0,vmax=100,marker="s",s=29,linewidths=0)
            else:
                ax.text(.5,.5,"выходов нет",transform=ax.transAxes,ha="center",va="center",fontsize=8)
            ax.set_title(f"η={eta:+.3f}");ax.grid(alpha=.16)
            bb=[float(r["beta"]) for r in boundaries]; ba=[float(r["alpha_boundary"]) for r in boundaries]
            if bb: ax.scatter(bb,ba,marker=".",s=13,c="cyan",alpha=.8)
            if ei%3==0: ax.set_ylabel("α")
            if ei>=6: ax.set_xlabel("β")
        if xs:
            fig.colorbar(sc,ax=axes.ravel().tolist(),fraction=.025,pad=.02,label="медиана первого выхода, RF-периоды")
        fig.suptitle(f"Медианное время выхода среди вышедших траекторий; τ₀={phase_deg:.0f}°",y=.995)
        for ext in ("png","pdf","svg"):
            fig.savefig(OUT/f"median_exit_time_phase_{phase_deg:.0f}deg.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
        plt.close(fig)


def floquet_and_voltage_figures(feasibility: list[dict], floquet: list[dict]) -> None:
    import matplotlib.pyplot as plt
    fidx={(int(row["alpha_i"]),int(row["beta_i"])):row for row in floquet}
    for ei,eta in enumerate(ETAS):
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=True,sharey=True)
        xx=[];yy=[];stable=[]
        for ai,alpha in enumerate(ALPHAS):
            for bi,beta in enumerate(BETAS):
                row=fidx[(ai,bi)]
                xx.append(beta);yy.append(alpha)
                stable.append(1 if row["status"]=="strictly_stable" else 0 if row["status"]=="unstable" else .5)
        axes[0].scatter(xx,yy,c=stable,cmap="coolwarm",vmin=0,vmax=1,marker="s",s=26,linewidths=0)
        axes[0].set_title("Линейный Floquet-статус центра\nQ и Q+RO одинаковы")
        vidx={(int(row["alpha_i"]),int(row["beta_i"]),int(row["eta_i"])):row for row in feasibility}
        feasible=[];bad=[]
        key_eta=4 if eta==0 else ei
        for ai,alpha in enumerate(ALPHAS):
            for bi,beta in enumerate(BETAS):
                row=vidx[(ai,bi,key_eta)]
                (feasible if bool(row["within_250V"]) else bad).append((beta,alpha))
        if feasible:
            axes[1].scatter([p[0] for p in feasible],[p[1] for p in feasible],c="#059669",marker="s",s=27,label="≤250 V")
        if bad:
            axes[1].scatter([p[0] for p in bad],[p[1] for p in bad],c="#dc2626",marker="x",s=20,label=">250 V")
        axes[1].set_title(f"Напряжения при η={eta:+.3f}")
        for ax in axes:
            ax.set_xlabel("β");ax.grid(alpha=.2)
        axes[0].set_ylabel("α");axes[1].legend(loc="best",fontsize=8)
        fig.suptitle("Точная линейная карта центра и независимая физическая маска",y=1.02)
        fig.tight_layout()
        for ext in ("png","pdf","svg"):
            fig.savefig(OUT/f"floquet_voltage_projection_eta_{eta:+.3f}.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
        plt.close(fig)


def isosurface_figures(rows: list[dict]) -> None:
    """Piecewise-linear Delaunay interpolation, explicitly distinguished from samples."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.interpolate import griddata

    points, values = [], []
    for row in rows:
        if int(row["phase_i"]) != 0 or not np.isfinite(float(row["P_ret"])):
            continue
        points.append([float(row["alpha"]), float(row["beta"]), float(row["kappa"])])
        values.append(float(row["P_ret"]))
    points = np.asarray(points); values = np.asarray(values)
    ag = np.linspace(ALPHAS.min(), ALPHAS.max(), 25)
    bg = np.linspace(BETAS.min(), BETAS.max(), 25)
    kg = np.linspace(-0.20*BETAS.max(), 0.20*BETAS.max(), 41)
    grid = np.stack(np.meshgrid(ag, bg, kg, indexing="ij"), axis=-1)
    scalar = griddata(points, values, grid.reshape(-1, 3), method="linear", rescale=True).reshape(grid.shape[:-1])
    axes = [ag, bg, kg]
    cube = ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))
    tetra = ((0,5,1,6),(0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6))

    def intersection(a, b, va, vb, level):
        f = (level-va)/(vb-va)
        return a + f*(b-a)

    def mesh(level):
        faces=[]
        for i in range(len(ag)-1):
          for j in range(len(bg)-1):
            for k in range(len(kg)-1):
              coords=np.array([[ag[i+di],bg[j+dj],kg[k+dk]] for di,dj,dk in cube])
              vals=np.array([scalar[i+di,j+dj,k+dk] for di,dj,dk in cube])
              if not np.all(np.isfinite(vals)) or level < np.min(vals) or level > np.max(vals):
                  continue
              for t in tetra:
                inside=[u for u in t if vals[u]>=level]
                outside=[u for u in t if vals[u]<level]
                if not inside or not outside: continue
                if len(inside)==1:
                    a=inside[0]; pts=[intersection(coords[a],coords[b],vals[a],vals[b],level) for b in outside]
                    faces.append(pts)
                elif len(inside)==3:
                    a=outside[0]; pts=[intersection(coords[a],coords[b],vals[a],vals[b],level) for b in inside]
                    faces.append(pts)
                else:
                    a,b=inside; c,d=outside
                    ac=intersection(coords[a],coords[c],vals[a],vals[c],level)
                    ad=intersection(coords[a],coords[d],vals[a],vals[d],level)
                    bc=intersection(coords[b],coords[c],vals[b],vals[c],level)
                    bd=intersection(coords[b],coords[d],vals[b],vals[d],level)
                    faces.extend(([ac,ad,bd],[ac,bd,bc]))
        return faces

    fig = plt.figure(figsize=(11.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors={0.5:"#2563eb",0.9:"#d97706"}
    for level in (0.5,0.9):
        faces=mesh(level)
        if faces:
            collection=Poly3DCollection(faces,facecolor=colors[level],edgecolor="none",alpha=.24 if level==.5 else .48)
            ax.add_collection3d(collection)
    ax.scatter(points[:,0],points[:,1],points[:,2],s=2,c=values,cmap="viridis",vmin=0,vmax=1,alpha=.20)
    ax.set_xlim(ag.min(),ag.max()); ax.set_ylim(bg.min(),bg.max()); ax.set_zlim(kg.min(),kg.max())
    ax.set_xlabel("α"); ax.set_ylabel("β"); ax.set_zlabel("κ")
    ax.set_title("Интерполированные изоповерхности P_ret=0.5 и 0.9 (δ=0, τ₀=0°)\n"
                 "линейная Delaunay-интерполяция только между вычисленными узлами; точки — фактические расчёты")
    fig.savefig(OUT / "retention_isosurfaces_interpolated.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "retention_isosurfaces_interpolated.pdf", bbox_inches="tight")
    fig.savefig(OUT / "retention_isosurfaces_interpolated.svg", bbox_inches="tight")
    plt.close(fig)
    (OUT / "isosurface_interpolation_metadata.json").write_text(json.dumps({
        "method": "scipy.griddata linear Delaunay + marching tetrahedra",
        "computed_nodes_used": len(points), "phase_deg": 0,
        "levels": [0.5,0.9], "grid_shape": [len(ag),len(bg),len(kg)],
        "warning": "Isosurfaces are interpolated visual reconstructions, not additional trajectory results."
    }, indent=2), encoding="utf-8")


def main() -> None:
    summary_path = OUT / "atlas_retention_summary.csv"
    if not summary_path.exists():
        raise SystemExit("Atlas summary not found; finish or resume the trajectory run first.")
    summary = read_csv(summary_path)
    voltage = voltage_and_feasibility()
    floquet_path=OUT/"floquet_reference_grid.csv"
    if floquet_path.exists():
        floquet=[]
        for row in read_csv(floquet_path):
            row={**row,"alpha_i":int(row["alpha_i"]),"beta_i":int(row["beta_i"]),
                 "alpha":float(row["alpha"]),"beta":float(row["beta"]),
                 **{k:float(row[k]) for k in ("trace_x","trace_y","det_x","det_y","margin_x","margin_y")}}
            floquet.append(row)
    else:
        floquet = floquet_reference()
    boundaries_path=OUT/"floquet_boundaries.csv"
    if boundaries_path.exists():
        boundaries=[]
        for row in read_csv(boundaries_path):
            boundaries.append({**row,"beta_i":int(row["beta_i"]),
                               "beta":float(row["beta"]),"alpha_boundary":float(row["alpha_boundary"]),
                               "branch_index_alpha_order":int(row["branch_index_alpha_order"]),
                               "active_trace":float(row["active_trace"]),"limit_trace":float(row["limit_trace"]),
                               "margin_at_root":float(row["margin_at_root"]),
                               "det_x":float(row["det_x"]),"det_y":float(row["det_y"])})
    else:
        boundaries = floquet_boundaries()
    write_csv(OUT / "physical_feasibility_grid.csv", voltage)
    write_csv(OUT / "floquet_reference_grid.csv", floquet)
    write_csv(OUT / "floquet_boundaries.csv", boundaries)
    enriched = enrich_summary(summary, voltage, floquet)
    write_csv(OUT / "atlas_retention_physical_summary.csv", enriched)
    floquet_and_voltage_figures(voltage, floquet)
    retention_figures(enriched, boundaries)
    isosurface_figures(enriched)
    meta = {
        "grid_nodes": len(voltage), "floquet_alpha_beta_nodes": len(floquet),
        "physical_points_at_250V": int(sum(r["within_250V"] for r in voltage)),
        "physical_fraction_of_atlas_nodes": float(sum(r["within_250V"] for r in voltage)/len(voltage)),
        "strict_floquet_alpha_beta_nodes": int(sum(r["status"]=="strictly_stable" for r in floquet)),
        "unstable_floquet_alpha_beta_nodes": int(sum(r["status"]=="unstable" for r in floquet)),
        "exact_floquet_boundary_roots_in_plot_window": len(boundaries),
        "voltage_limit_V": U_MAX_V, "reference_mass_Th": M0_TH,
        "frequency_Hz": F_HZ, "R_m": R_M,
        "c2": C2, "d2": D2, "c4": C4,
        "phase_slices_deg": [0, 90],
        "atlas_interpolated": False,
        "isosurfaces_interpolated": True,
    }
    (OUT / "atlas_product_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
