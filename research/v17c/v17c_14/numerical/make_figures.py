"""Reproducible V17C.14 analytic and published-data figures; no trajectory campaign."""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results/v17c13_reference_data.json").read_text(encoding="utf-8"))
OUT = ROOT / "figures"
TABLES = ROOT / "results/figure_data"
OUT.mkdir(exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
P = DATA["local15"]
COL = DATA["coulomb_sweep"]

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.15,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"


def save_figure(fig, stem: str):
    for ext in ("pdf", "svg", "png"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.savefig(OUT / f"{stem}.{ext}", dpi=600 if ext == "png" else None,
                        bbox_inches="tight")
        if ext == "svg":
            svg_path = OUT / f"{stem}.svg"
            svg_text = svg_path.read_text(encoding="utf-8")
            svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                                encoding="utf-8")
        for warning in caught:
            print(f"layout warning [{stem}.{ext}]: {warning.message}")
    plt.close(fig)


def save_csv(stem: str, rows: list[dict]):
    path = TABLES / f"{stem}.csv"
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def leading_terms(alpha, lam, B=None, nu=None, rho=None):
    A = P["A"]
    B = P["B"] if B is None else B
    nu = P["nu"] if nu is None else nu
    k1 = alpha**2 * 4 * A**2 / (1 + lam**2)
    k2 = alpha**2 * 48 * B**2 / (nu**2 + lam**2)
    return k1, k2


def lead_force(alpha, lam, rho, B=None, nu=None):
    k1, k2 = leading_terms(alpha, lam, B, nu)
    fr = -k1 * rho - k2 * rho**5
    ft = lam * (k1 * rho - (k2 / nu) * rho**5)
    return fr, ft


def equilibrium_rhs(state, alpha, lam, beta, B=None, nu=None):
    rho, vr, omega = state
    if rho <= 0:
        rho = 1e-12
    fr, ft = lead_force(alpha, lam, rho, B, nu)
    return np.array([
        vr,
        rho * omega**2 + fr + alpha * beta * rho - lam * vr,
        ft / rho - 2 * vr * omega / rho - lam * omega,
    ])


def fixed_points(alpha, lam, beta, B=None, nu=None, stability=True):
    B = P["B"] if B is None else B
    nu = P["nu"] if nu is None else nu
    k1, k2 = leading_terms(alpha, lam, B, nu)
    # x=rho^4; simultaneous radial and azimuthal balance gives a quadratic.
    roots = np.roots([(k2 / nu)**2,
                      -2 * k1 * k2 / nu - k2,
                      k1**2 + alpha * beta - k1])
    ans = []
    for x in roots:
        if abs(x.imag) > 1e-10 or not (0 < x.real < 1):
            continue
        rho = float(x.real**0.25)
        omega = float(k1 - k2 / nu * x.real)
        eig = np.array([])
        stable = False
        if stability:
            y = np.array([rho, 0.0, omega])
            h = 1e-6
            jac = np.column_stack([
                (equilibrium_rhs(y + np.eye(3)[j] * h, alpha, lam, beta, B, nu)
                 - equilibrium_rhs(y - np.eye(3)[j] * h, alpha, lam, beta, B, nu)) / (2*h)
                for j in range(3)
            ])
            eig = np.linalg.eigvals(jac)
            stable = bool(np.max(eig.real) < 1e-7)
        ans.append({"rho": rho, "omega": omega, "stable": stable,
                    "eigenvalues": eig})
    return sorted(ans, key=lambda q: q["rho"])


def fig01_phase_map():
    """Question: where does leading-only local15 model have stable opposite-sign roots?"""
    ratios = np.linspace(0.5, 7.0, 121)  # B/A; A, beta, lambda_i, alpha_i fixed
    nus = np.linspace(1.2, 5.0, 111)
    z = np.zeros((len(nus), len(ratios)), dtype=np.uint8)
    rows = []
    for iy, nu in enumerate(nus):
        for ix, ratio in enumerate(ratios):
            B = ratio * P["A"]
            roots1 = fixed_points(1.0, P["lambda1"], P["beta"], B, nu)
            roots2 = fixed_points(P["alpha2"], P["lambda2"], P["beta"], B, nu)
            s1 = [r for r in roots1 if r["stable"]]
            s2 = [r for r in roots2 if r["stable"]]
            if s1 and s2:
                opposite = any(a["omega"] * b["omega"] < 0 for a in s1 for b in s2)
                z[iy, ix] = 2 if opposite else 1
            elif roots1 or roots2:
                z[iy, ix] = 3
            rows.append({"B_over_A": ratio, "nu": nu, "class_code": int(z[iy, ix]),
                         "class": {0:"no paired stable roots",1:"stable roots, same sign",
                                   2:"stable opposite-sign pair",3:"partial or unstable roots"}[int(z[iy,ix])]})
    multipole = json.loads((ROOT / "results/generalized_multipole_sweep.json").read_text(encoding="utf-8"))
    candidate_names = ["candidate52", "local15", "refine12", "refine12_sobol17"]
    order_pairs = [(2,2),(2,3),(2,4),(2,6),(3,4),(3,6),(4,6)]
    matrix = np.zeros((len(order_pairs), len(candidate_names)), dtype=np.uint8)
    for iy,(p,s) in enumerate(order_pairs):
        for ix,candidate in enumerate(candidate_names):
            key=f"{candidate}:{p},{s}"
            q=multipole["stable_root_summary_by_candidate_and_orders"][key]
            n1=q["species1_stable_root_count"]; n2=q["species2_stable_root_count"]
            if n1==0 and n2==0: matrix[iy,ix]=0
            elif n1==0 or n2==0: matrix[iy,ix]=3
            elif q["stable_opposite_sign_pair_exists"]: matrix[iy,ix]=2
            else: matrix[iy,ix]=1
    fig, (ax, ax_order) = plt.subplots(1, 2, figsize=(11.0, 5.15),
                                       gridspec_kw={"width_ratios":[1.25,1.0]},
                                       constrained_layout=True)
    cmap = ListedColormap(["#f0f0f0", BLUE, ORANGE, "#999999"])
    mesh = ax.pcolormesh(ratios, nus, z, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest", rasterized=True)
    ax.plot(P["B"] / P["A"], P["nu"], marker="*", markersize=12, color="black", label="local15 point")
    ax.set(xlabel=r"$B/A$", ylabel=r"$\nu=\Omega_2/\Omega_1$",
           title="Leading $p=2$, $s=4$ averaged model")
    ax_order.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
    ax_order.set_xticks(np.arange(len(candidate_names)), ["candidate52","local15","refine12","Sobol17"], rotation=28, ha="right")
    ax_order.set_yticks(np.arange(len(order_pairs)), [f"({p},{s})" for p,s in order_pairs])
    ax_order.set(xlabel="Legacy parameter set", ylabel=r"$(p,s)$",
                 title="Stable-root signs under order changes")
    for iy in range(len(order_pairs)):
        for ix in range(len(candidate_names)):
            label={0:"none",1:"same",2:"opposite",3:"partial"}[int(matrix[iy,ix])]
            ax_order.text(ix,iy,label,ha="center",va="center",fontsize=7,
                          color="white" if matrix[iy,ix] in (1,2,3) else "0.25")
    from matplotlib.patches import Patch
    handles=[Patch(color=cmap(i),label=l) for i,l in enumerate([
        "no paired stable roots", "same-sign stable pair", "counterrotating stable pair", "partial / unstable" ])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5,-0.055), frameon=False, ncol=4)
    ax.text(0.0, -0.22, r"Fixed: local15 $A,\beta,\lambda_i,\alpha_i$; roots $0<\rho<1$. Reduced 3D averaged Jacobian.",
            transform=ax.transAxes, fontsize=7)
    ax_order.text(0.0, -0.22, r"Each candidate keeps its own $A,B,\nu,\beta,\lambda_i,\alpha_i$; leading potential equal at $\rho=1$.",
                  transform=ax_order.transAxes, fontsize=7)
    save_csv("fig01_phase_map", rows)
    save_figure(fig, "FIG01_ANALYTIC_COUNTERROTATION_MAP")


def harmonic_force_hessian(rho, theta, alpha, mode, amp, coeff):
    z = rho * np.exp(1j * theta)
    n = abs(int(mode))
    base = z if mode > 0 else np.conjugate(z)
    orient = 1j if mode > 0 else -1j
    f = np.zeros(2, dtype=complex)
    df = np.zeros((2, 2), dtype=complex)
    if n >= 1:
        gx = amp * coeff * n * base**(n-1)
        gy = orient * gx
        f[:] = -alpha * np.array([gx, gy])
    if n >= 2:
        hxx = amp * coeff * n * (n-1) * base**(n-2)
        hxy = orient * hxx
        df[:] = -alpha * np.array([[hxx, hxy], [hxy, -hxx]])
    return f, df


def mfs_averaged_force(rho, theta, alpha, lam, family):
    modes = P["mfs_modes"]
    families = P["mfs_family"]
    coeffs = [complex(*v) for v in P["mfs_coefficients_real_imag"]]
    amp = P["A"] if family == 0 else P["B"]
    nu = 1.0 if family == 0 else P["nu"]
    total = np.zeros(2)
    den = -nu**2 + 1j * lam * nu
    for k, fam, c in zip(modes, families, coeffs):
        if fam != family:
            continue
        f, df = harmonic_force_hessian(rho, theta, alpha, k, amp, c)
        xihat_star = np.conjugate(f / (-nu**2 - 1j * lam * nu))
        total += 0.5 * np.real(df @ xihat_star)
    er = np.array([np.cos(theta), np.sin(theta)])
    et = np.array([-np.sin(theta), np.cos(theta)])
    return float(total @ er), float(total @ et)


def fig02_averaged_force():
    """Question: do the eight MFS modes preserve and modulate leading averaged forces?"""
    rho_grid = np.linspace(0.2, 0.88, 70)
    theta_grid = np.linspace(0, 2*np.pi/24, 49, endpoint=False)
    rows=[]
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.7), constrained_layout=True)
    for species, alpha, lam, color in [(1,1.0,P["lambda1"],BLUE),(2,P["alpha2"],P["lambda2"],ORANGE)]:
        stats={"rho":rho_grid}
        for fam in (0,1):
            frmat=np.zeros((len(theta_grid),len(rho_grid))); ftmat=frmat.copy()
            for ir,rho in enumerate(rho_grid):
                for it,th in enumerate(theta_grid):
                    frmat[it,ir],ftmat[it,ir]=mfs_averaged_force(rho,th,alpha,lam,fam)
            stats[f"fr{fam}"]=frmat; stats[f"ft{fam}"]=ftmat
        fr=stats["fr0"]+stats["fr1"]; ft=stats["ft0"]+stats["ft1"]
        for ir,rho in enumerate(rho_grid):
            k1,k2=leading_terms(alpha,lam)
            frlead=-k1*rho-k2*rho**5
            ftlead=lam*(k1*rho-k2/P["nu"]*rho**5)
            for label, mat, lead in [("radial",fr,frlead),("azimuthal",ft,ftlead)]:
                lo,med,hi=np.quantile(mat[:,ir],[0.05,0.5,0.95])
                rows.append({"species":species,"rho":rho,"component":label,"leading_only":lead,
                             "full_mfs_q05":lo,"full_mfs_median":med,"full_mfs_q95":hi,
                             "angle_samples":len(theta_grid),"angle_domain":"0 to 2pi/24"})
        for ax,mat,lead,title,ylab in [
            (axs[0],fr,lambda r: -leading_terms(alpha,lam)[0]*r-leading_terms(alpha,lam)[1]*r**5,"Radial mean force",r"$\overline{F}_\rho$"),
            (axs[1],ft,lambda r: lam*(leading_terms(alpha,lam)[0]*r-leading_terms(alpha,lam)[1]/P["nu"]*r**5),"Azimuthal mean force",r"$\overline{F}_\theta$"),
        ]:
            q=np.quantile(mat,[.05,.5,.95],axis=0)
            ax.fill_between(rho_grid,q[0],q[2],color=color,alpha=.15)
            ax.plot(rho_grid,q[1],color=color,label=f"full MFS, species {species}")
            ax.plot(rho_grid,[lead(r) for r in rho_grid],color=color,ls="--",lw=.9)
            ax.axhline(0,color="0.55",lw=.6)
            ax.set(title=title,xlabel=r"$\rho=r/R$",ylabel=ylab)
        # Sum-frequency averaged angular drift: F_theta/(lambda*rho).
        w=ft/(lam*rho_grid[None,:])
        q=np.quantile(w,[.05,.5,.95],axis=0)
        axs[2].fill_between(rho_grid,q[0],q[2],color=color,alpha=.15)
        axs[2].plot(rho_grid,q[1],color=color,label=f"full MFS, species {species}")
        k1,k2=leading_terms(alpha,lam)
        axs[2].plot(rho_grid,k1-k2/P["nu"]*rho_grid**4,color=color,ls="--",lw=.9)
        axs[2].axhline(0,color="0.55",lw=.6)
        axs[2].set(title="Slow angular drift",xlabel=r"$\rho=r/R$",ylabel=r"$\overline{F}_\theta/(\lambda\rho)$")
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=BLUE,label="species 1"),Line2D([0],[0],color=ORANGE,label="species 2"),
             Line2D([0],[0],color="0.25",ls="-",label="full MFS"),Line2D([0],[0],color="0.25",ls="--",label="leading $k=2,-4$"),
             Line2D([0],[0],color="0.5",lw=5,alpha=.2,label="5–95% sector-angle band")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,1.08),ncol=5,frameon=False,fontsize=7)
    for ax in axs: ax.axvline(P["rho1_n6_mean"],color=BLUE,ls=":",lw=.8); ax.axvline(P["rho2_n6_mean"],color=ORANGE,ls=":",lw=.8)
    fig.suptitle("Frozen-position second-order average; angular bands from the 24-fold sector",y=1.13)
    fig.text(.5,-.035,r"Fixed local15 $A,B,\nu,\lambda_i,\alpha_i$; no trajectory starts or orbit integration. Band: 5–95% over $\theta\in[0,2\pi/24)$.",ha="center",fontsize=7.5)
    save_csv("fig02_averaged_force", rows)
    save_figure(fig,"FIG02_AVERAGED_RF_FORCE_TORQUE")


def fig03_branches():
    """Question: how do reduced averaged circular roots change with axial DC strength?"""
    betas=np.linspace(0.005,0.030,301)
    species=[(1,1.0,P["lambda1"],BLUE),(2,P["alpha2"],P["lambda2"],ORANGE)]
    rows=[]
    fig,ax=plt.subplots(figsize=(7.0,4.8))
    for si,alpha,lam,color in species:
        for ib, beta in enumerate(betas):
            for r in fixed_points(alpha,lam,beta):
                rows.append({"species":si,"beta":beta,"rho":r["rho"],"omega":r["omega"],"stable_reduced_model":r["stable"],
                             "max_real_eigenvalue":float(np.max(r["eigenvalues"].real))})
                if ib % 10 == 0 or ib == len(betas)-1:
                    ax.scatter(beta,r["omega"],s=8,alpha=.9,marker="o" if r["stable"] else "x",
                               facecolors=color if r["stable"] else "none",edgecolors=color,lw=.55)
        ax.scatter([],[],s=22,marker="o",facecolor=color,edgecolor=color,label=f"species {si}: stable reduced roots")
        ax.scatter([],[],s=24,marker="x",color=color,label=f"species {si}: unstable reduced roots")
    ax.axvline(P["beta"],color="black",ls=":",lw=.9,label="local15 $\\beta$")
    ax.scatter(P["beta"],P["omega1_n6_mean"],marker="*",s=80,color=BLUE,edgecolor="black",lw=.4,label="V17C.13 N=6, selected start")
    ax.scatter(P["beta"],P["omega2_n6_mean"],marker="*",s=80,color=ORANGE,edgecolor="black",lw=.4)
    ax.axhline(0,color="0.45",lw=.6)
    ax.set(xlabel=r"$\beta$",ylabel=r"$\bar\omega=\omega_{\rm orb}/\Omega_1$",
           title="Circular roots of the leading averaged model")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=7,
               loc="upper center", bbox_to_anchor=(.5,.99))
    fig.subplots_adjust(left=.12, right=.98, bottom=.20, top=.78)
    fig.text(.5,.035,"301 beta/species root evaluations; every tenth root is marked. Fixed local15 A, B, nu, lambda_i, alpha_i; 0<rho<1.",
             ha="center", fontsize=7.2)
    fig.text(.5,.010,"Stars: full-MFS+C N=6 data from one selected ordered start. No trajectory starts or integration horizon in the branch calculation.",
             ha="center", fontsize=7.2)
    save_csv("fig03_rotation_branches",rows)
    save_figure(fig,"FIG03_ROTATIONAL_BRANCHES")


def fig04_coulomb_transition():
    """Question: where does one selected N=6 trajectory change from counter- to co-rotation?"""
    c=np.asarray(COL["C_over_C0"]); x=np.log10(1.0+c)
    f1=np.asarray(COL["group1_mean_frequency_Hz"])/COL["f1_Hz"]
    f2=np.asarray(COL["group2_mean_frequency_Hz"])/COL["f1_Hz"]
    reduced=json.loads((ROOT/"results/reduced_collective_locking.json").read_text(encoding="utf-8"))
    c_lock=float(reduced["fold_prediction"]["Ccrit_over_C0"])
    omega_lock=float(reduced["required_lock"]["common_omega_over_Omega1"])
    rows=[{"C_over_C0":float(x),"omega_group1":float(y1),"omega_group2":float(y2),
           "starts_per_point":COL["runs_per_point"],"horizon_periods":COL["horizon_after_loading_periods"]}
          for x,y1,y2 in zip(c,f1,f2)]
    fig,ax=plt.subplots(figsize=(7.0,4.8))
    ax.axvspan(np.log10(3),np.log10(6),color="#e6e6e6",zorder=0,
               label=r"source sign-change bracket: $2<C/C_0\leq5$")
    ax.axvline(np.log10(1.0+c_lock),color="#555555",ls="--",lw=1.0,
               label=rf"reduced fold: $C/C_0={c_lock:.3f}$")
    ax.axhline(omega_lock,color="#555555",ls=":",lw=1.0,
               label=rf"model lock: $\bar{{\omega}}={omega_lock:+.2e}$")
    ax.scatter(x,f1,color=BLUE,marker="o",label="inner group")
    ax.scatter(x,f2,color=ORANGE,marker="s",label="outer group")
    ax.axhline(0,color="0.4",lw=.6)
    ticks=np.asarray([0,.5,1,2,5,10,20,50],dtype=float)
    ax.set_xticks(np.log10(1+ticks),[f"{v:g}" for v in ticks])
    ax.set(xlim=(-.025,np.log10(55.0)),xlabel=r"$C/C_0$ (nonuniform positions)",
           ylabel=r"$\bar\omega=\omega_{\rm orb}/\Omega_1$")
    fig.suptitle("Discrete Coulomb continuation from one selected 3+3 state",y=.98)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7.1,loc="center",
               bbox_to_anchor=(.5,.875),ncol=3,columnspacing=1.4,handlelength=2.0)
    fig.subplots_adjust(left=.13,right=.98,bottom=.25,top=.79)
    fig.text(.5,.12,r"$x=\log_{10}(1+C/C_0)$; ticks show raw $C/C_0$. Source: one ordered start per $C$; RK4/96, 15,000 post-loading periods; increasing $C$.",
             ha="center",fontsize=7.2)
    fig.text(.5,.065,r"Overlay: fixed-radius exact-torque phase model calibrated at $C=0$; predicted fold, not a full-system boundary.",
             ha="center",fontsize=7.2)
    save_csv("fig04_coulomb_sweep",rows)
    save_figure(fig,"FIG04_COULOMB_ROTATIONAL_TRANSITION")


def fig05_internal_modes():
    """Question: what are frozen-ring internal phase-mode eigenvalues for M=3?"""
    scales=np.linspace(0,12,301)
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.6),constrained_layout=True)
    rows=[]
    for name,Q,mu,rho,lam,color in [
        ("inner",1.0,1.0,P["rho1_n6_mean"],P["lambda1"],BLUE),
        ("outer",P["q2_over_q1"],P["m2_over_m1"],P["rho2_n6_mean"],P["lambda2"],ORANGE)]:
        for c in scales:
            wc2=c*5*P["kappa"]*Q**2/(4*np.sqrt(3)*mu*rho**3)
            disc=lam**2/4-wc2
            root=np.sqrt(complex(disc))
            ev1=-lam/2+root; ev2=-lam/2-root
            rows.append({"ring":name,"C_scale":c,"eigen1_real":ev1.real,"eigen1_imag":ev1.imag,
                         "eigen2_real":ev2.real,"eigen2_imag":ev2.imag,"omega_C_squared":wc2})
        arr=[r for r in rows if r["ring"]==name]
        e1=np.array([r["eigen1_real"]+1j*r["eigen1_imag"] for r in arr]); e2=np.array([r["eigen2_real"]+1j*r["eigen2_imag"] for r in arr])
        axs[0].plot(scales[1:],-e1.real[1:]/lam,color=color,label=f"{name}, slow root")
        axs[0].plot(scales[1:],-e2.real[1:]/lam,color=color,ls="--",label=f"{name}, fast root")
        wc=np.sqrt(np.array([r["omega_C_squared"] for r in arr]))
        axs[1].plot(scales,wc/lam,color=color,label=f"{name} ring")
        j=np.argmin(np.abs(scales-1.0)); axs[0].scatter([1,1],[-e1[j].real/lam,-e2[j].real/lam],color=color,edgecolor="black",s=22,zorder=4)
        axs[1].scatter([1],[wc[j]/lam],color=color,edgecolor="black",s=22,zorder=4)
    axs[0].set_yscale("log")
    axs[0].set(xlabel=r"$C/C_0$",ylabel=r"$-\Re(s)/\lambda$",title="Mode decay rates")
    axs[1].axhline(.5,color="black",ls=":",lw=.8,label=r"$\omega_C/\lambda=1/2$")
    axs[1].set(xlabel=r"$C/C_0$",ylabel=r"$\omega_C/\lambda$",title="Damping regime")
    axs[0].legend(frameon=False,fontsize=6.4)
    axs[1].legend(frameon=False,fontsize=7)
    axs[0].text(.0,-.28,"Analytic frozen-ring roots; c=0 has a neutral phase mode. Coupling scaled uniformly from local15.",transform=axs[0].transAxes,fontsize=6.7)
    axs[1].text(.0,-.28,"Throughout 0<C/C0≤12, ωC/λ<1/2: overdamped in this isolated-ring approximation.",transform=axs[1].transAxes,fontsize=6.7)
    fig.suptitle("Internal phase modes of isolated frozen-radius triangles",y=1.05)
    save_csv("fig05_internal_phase_modes",rows)
    save_figure(fig,"FIG05_INTERNAL_PHASE_MODE_SPECTRUM")


def axial_stiffness_eigs(delta):
    rhos=np.array([P["rho1_n6_mean"]]*3+[P["rho2_n6_mean"]]*3)
    z=np.zeros(6); theta=np.r_[np.arange(3)*2*np.pi/3,delta+np.arange(3)*2*np.pi/3]
    xy=np.c_[rhos*np.cos(theta),rhos*np.sin(theta)]
    Q=np.array([1.0]*3+[P["q2_over_q1"]]*3)
    mu=np.array([1.0]*3+[P["m2_over_m1"]]*3)
    H=np.diag(2*P["beta"]*Q)
    for i in range(6):
        for j in range(i+1,6):
            d=np.linalg.norm(xy[i]-xy[j])
            g=P["kappa"]*Q[i]*Q[j]/d**3
            H[i,i]-=g; H[j,j]-=g; H[i,j]+=g; H[j,i]+=g
    invsqrt=1/np.sqrt(mu)
    S=invsqrt[:,None]*H*invsqrt[None,:]
    return np.linalg.eigvalsh(S)


def fig06_axial_modes():
    """Question: how does frozen axial stiffness depend on the relative phase of two triangles?"""
    phases=np.linspace(0,2*np.pi/3,121)
    vals=np.array([axial_stiffness_eigs(p) for p in phases])
    rows=[]
    for phase,eigs in zip(phases,vals):
        for m,eig in enumerate(eigs): rows.append({"relative_phase_rad":phase,"mode_sorted":m,"stiffness_eigenvalue":eig})
    fig,axs=plt.subplots(1,2,figsize=(9.0,4.7))
    palette=["#0072B2","#56B4E9","#009E73","#E69F00","#D55E00","#CC79A7"]
    for m in range(6):
        axs[0].scatter(phases,vals[:,m],s=5,color=palette[m],label=f"sorted mode {m+1}",rasterized=True)
        axs[1].scatter(phases,vals[:,m]-vals[:,m].mean(),s=5,color=palette[m],rasterized=True)
    axs[0].set(xlabel=r"$\Delta\Theta$ (rad; $0\leq\Delta\Theta\leq2\pi/3$)",ylabel=r"$\chi_j(\Delta\Theta)/\Omega_1^2$",title="Absolute stiffness spectrum")
    axs[1].axhline(0,color="black",lw=.6)
    axs[1].set(xlabel=r"$\Delta\Theta$ (rad)",ylabel=r"$\chi_j-\langle\chi_j\rangle_\Delta$",title="Phase modulation about mode mean")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0],[0],color=palette[m],marker="o",lw=0,markersize=4)
                for m in range(6)], [f"sorted mode {m+1}" for m in range(6)],
               frameon=False,ncol=6,fontsize=7,loc="upper center",bbox_to_anchor=(.5,.96))
    fig.subplots_adjust(left=.10,right=.99,bottom=.20,top=.80,wspace=.32)
    fig.suptitle("Instantaneous mass-symmetrized axial stiffness",y=.995)
    fig.text(.5,.055,"Fixed local15 radii, z=0 and 3+3 point rings; full 15-pair Hessian. Frozen-phase snapshots, not Floquet/Lyapunov exponents.",
             ha="center",fontsize=7.2)
    fig.text(.5,.025,"Relative phase varies on the counterrotating orbit; snapshot eigenvalues alone do not decide stability.",
             ha="center",fontsize=7.2)
    save_csv("fig06_axial_stiffness",rows)
    save_figure(fig,"FIG06_AXIAL_COLLECTIVE_STIFFNESS")


def fig07_h4_map():
    """Question: where can H4 keep an inner ring centered while an outer ring has off-axis axial extrema?"""
    rho=np.linspace(.25,.95,260); eta=np.linspace(.05,2.0,280)
    crit=1/(3*rho**2)
    state=(eta[None,:]>=crit[:,None]).astype(int) # 0: central minimum; 1: off-axis extrema exist
    r1,r2=P["rho1_n6_mean"],P["rho2_n6_mean"]
    eta_lo=1/(3*r2**2); eta_hi=1/(3*r1**2)
    fig,axs=plt.subplots(1,2,figsize=(9.0,4.2))
    cm=ListedColormap(["#56B4E9","#D55E00"])
    axs[0].pcolormesh(rho,eta,state.T,cmap=cm,vmin=-.5,vmax=1.5,shading="nearest",rasterized=True)
    axs[0].plot(rho,crit,color="black",lw=1.0,label=r"$\eta_c=1/(3\rho^2)$")
    axs[0].axvline(r1,color=BLUE,ls=":",label="inner local15 shell")
    axs[0].axvline(r2,color=PURPLE,ls=":",label="outer local15 shell")
    axs[0].axhspan(eta_lo,eta_hi,color="white",alpha=.25)
    axs[0].set(xlabel=r"$\rho$",ylabel=r"$\eta=\beta_4/\beta_2$",title="Fixed-radius axial extrema")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color="#56B4E9",label="central plane stable"),
                           Patch(color="#D55E00",label="off-axis extrema exist"),
                           Line2D([0],[0],color="black",label=r"$\eta_c=1/(3\rho^2)$"),
                           Line2D([0],[0],color=BLUE,ls=":",label="inner shell"),
                           Line2D([0],[0],color=PURPLE,ls=":",label="outer shell")],
                  frameon=False,fontsize=6.4,loc="upper right")
    r1g=np.linspace(.3,.85,140); r2g=np.linspace(.35,.98,150)
    width=np.full((len(r2g),len(r1g)),np.nan)
    for iy,ro in enumerate(r2g):
        for ix,ri in enumerate(r1g):
            if ro>ri: width[iy,ix]=1/(3*ri**2)-1/(3*ro**2)
    pc=axs[1].pcolormesh(r1g,r2g,width,cmap="viridis",shading="auto",rasterized=True)
    axs[1].plot([.3,.85],[.3,.85],color="black",lw=.8)
    axs[1].scatter([r1],[r2],marker="*",s=85,color="white",edgecolor="black",label="local15 radii")
    axs[1].set(xlabel=r"inner radius $\rho_1$",ylabel=r"outer radius $\rho_2$",title=r"Allowed $\eta$-window width")
    axs[1].legend(frameon=False,fontsize=7)
    cb=fig.colorbar(pc,ax=axs[1],shrink=.86); cb.set_label(r"$1/(3\rho_1^2)-1/(3\rho_2^2)$")
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.88,wspace=.34)
    fig.text(.5,.055,r"Assumptions: $\beta_2,\beta_4>0$, positive charge, fixed radius. This is an axial-only criterion; radial force, RF and Coulomb coupling can remove these extrema as full 3D states.",ha="center",fontsize=7.2)
    rows=[]
    for ri in r1g:
        for ro in r2g:
            rows.append({"rho_inner":ri,"rho_outer":ro,"eta_min_for_outer_offaxis":1/(3*ro**2) if ro>0 else "",
                         "eta_max_for_inner_center":1/(3*ri**2) if ri>0 else "",
                         "eta_window_width":(1/(3*ri**2)-1/(3*ro**2)) if ro>ri else "",
                         "ordered_radii":bool(ro>ri)})
    save_csv("fig07_h4_axial_map",rows)
    save_figure(fig,"FIG07_H4_AXIAL_STABILITY_MAP")


def fig08_averaging_validity():
    """Question: where is the local micromotion correction small for the full MFS spectrum?"""
    rho=np.linspace(.05,.9,95); theta=np.linspace(0,2*np.pi/24,49,endpoint=False)
    modes=P["mfs_modes"]; fams=P["mfs_family"]; coeffs=[complex(*v) for v in P["mfs_coefficients_real_imag"]]
    rows=[]
    fig,ax=plt.subplots(figsize=(7.0,4.8))
    for si,alpha,lam,color in [(1,1.0,P["lambda1"],BLUE),(2,P["alpha2"],P["lambda2"],ORANGE)]:
        for family,ls in [(0,"-"),(1,"--")]:
            nu=1.0 if family==0 else P["nu"]; amp=P["A"] if family==0 else P["B"]
            values=[]
            for rr in rho:
                norms=[]
                for th in theta:
                    dftot=np.zeros((2,2),complex)
                    for k,fam,c in zip(modes,fams,coeffs):
                        if fam!=family: continue
                        _,df=harmonic_force_hessian(rr,th,alpha,k,amp,c)
                        dftot+=df
                    norms.append(np.linalg.norm(dftot,2)/(nu*np.sqrt(nu**2+lam**2)))
                values.append(norms)
            arr=np.asarray(values).T
            lo,med,hi=np.quantile(arr,[.05,.5,.95],axis=0)
            ax.fill_between(rho,lo,hi,color=color,alpha=.10)
            ax.plot(rho,med,color=color,ls=ls,label=f"species {si}, RF family {family+1}")
            for rr,mm,ll,hh in zip(rho,med,lo,hi):
                rows.append({"species":si,"family":family+1,"rho":rr,"epsilon_median":mm,
                             "epsilon_q05":ll,"epsilon_q95":hh,"angle_samples":len(theta)})
    ax.axhline(1,color="black",ls=":",lw=.8,label=r"$\epsilon_\mu=1$ (order-one reference)")
    ax.axvline(P["rho1_n6_mean"],color=BLUE,ls=":",lw=.7)
    ax.axvline(P["rho2_n6_mean"],color=ORANGE,ls=":",lw=.7)
    ax.set_yscale("log"); ax.set(xlabel=r"$\rho=r/R$",ylabel=r"$\epsilon_\mu=\|Df\|/[\nu\sqrt{\nu^2+\lambda^2}]$",
        title="Local frozen-position averaging parameter")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=7,
               loc="upper center", bbox_to_anchor=(.5,.96))
    fig.subplots_adjust(left=.13,right=.98,bottom=.20,top=.82)
    fig.text(.5,.055,"Full eight-mode MFS; 5–95% sector-angle band. The reference ε=1 is not a universal validity boundary.",
             ha="center",fontsize=7.2)
    fig.text(.5,.025,"Doppler shift, resonances, slow-envelope variation and Coulomb coupling are not included.",
             ha="center",fontsize=7.2)
    save_csv("fig08_averaging_validity",rows)
    save_figure(fig,"FIG08_AVERAGING_VALIDITY")


def main():
    fig01_phase_map(); fig02_averaged_force(); fig03_branches(); fig04_coulomb_transition()
    fig05_internal_modes(); fig06_axial_modes(); fig07_h4_map(); fig08_averaging_validity()
    print(f"created 8 figure sets in {OUT}; data in {TABLES}")


if __name__ == "__main__":
    main()
