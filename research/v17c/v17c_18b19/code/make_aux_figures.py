"""Make auxiliary axial, geometry, trajectory and physically linked figures."""
from __future__ import annotations

import json
import math
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from scipy.special import i0

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; FIG=ROOT/"figures"
plt.rcParams.update({"font.size":9,"axes.labelsize":10,"axes.titlesize":11,
                     "legend.fontsize":8,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":120})


def save(fig,stem):
    for ext in ("png","pdf","svg"):
        fig.savefig(FIG/f"{stem}.{ext}",dpi=600 if ext=="png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def fig05():
    R=5.0; D=R/0.95
    dz=np.linspace(1.2,20.0,500)
    transfer=1.0/i0(np.pi*D/dz)
    points=np.array([2.,5.,10.,15.])
    vals=1.0/i0(np.pi*D/points)
    fig,ax=plt.subplots(figsize=(6.7,4.7),layout="constrained")
    ax.plot(dz/R,transfer,color="#245b78",lw=1.8,label=r"$1/I_0(\pi D/d_z)$")
    ax.scatter(points/R,vals,color="#d1495b",s=28,zorder=3)
    for d,v in zip(points,vals):
        ax.annotate(f"{d:g} mm\n{v:.3f}",(d/R,v),xytext=(4,7),
                    textcoords="offset points",fontsize=7)
    ax.set(xlabel=r"characteristic axial half-wave $d_z/R$",
           ylabel="axis amplitude / cylindrical-wall amplitude",
           title="FIG05  Laplace attenuation of an axial DC feature")
    ax.set_ylim(0,1.02); ax.set_xlim(dz[0]/R,dz[-1]/R)
    ax.grid(True,alpha=.22); ax.legend(frameon=False)
    ax.text(.02,.03,"Exact infinite-cylinder Fourier solution; R=5 mm, D=R/0.95.\n"
            "Screening only: not the segmented 24-rod field.",transform=ax.transAxes,fontsize=7)
    save(fig,"FIG05_axial_DC_transfer")
    rows=[{"dz_mm":float(d),"dz_over_R":float(d/R),"kD":float(np.pi*D/d),
           "axis_wall_amplitude":float(v),"model":"infinite cylindrical Dirichlet wall; k=pi/dz"}
          for d,v in zip(points,vals)]
    (DATA/"fig05_axial_DC_transfer.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")


def fig06():
    # Publication-sized orthographic schematic. This is deliberately a CAD
    # concept, not a field plot or a solved finite-length boundary problem.
    fig=plt.figure(figsize=(12.6,4.65),layout="constrained")
    grid=fig.add_gridspec(2,3,height_ratios=[1,.12],width_ratios=[1,1.15,1.05])
    ax1,ax2,ax3=(fig.add_subplot(grid[0,i]) for i in range(3))
    footer=fig.add_subplot(grid[1,:]); footer.set_axis_off()
    D=1/0.95; rod_a=.05/0.95; shield=1.10/0.95
    ax1.add_patch(Circle((0,0),shield,fill=False,lw=1.4,color="#34495e"))
    ax1.add_patch(Circle((0,0),1.0,fill=False,lw=.8,color="#8898a6",ls="--"))
    for j in range(24):
        th=2*np.pi*j/24
        ax1.add_patch(Circle((D*np.cos(th),D*np.sin(th)),rod_a,
                            facecolor="#345b75",edgecolor="white",lw=.35))
    ax1.set_aspect("equal"); ax1.set_xlim(-1.28,1.28); ax1.set_ylim(-1.28,1.28)
    ax1.set_xlabel(r"$x/R$"); ax1.set_ylabel(r"$y/R$")
    ax1.set_title("24 shared two-tone RF rods")
    ax1.text(0,-.12,"free aperture R",ha="center",fontsize=8,color="#5f6b73")
    ax1.text(0,1.21,"grounded shield",ha="center",fontsize=8,color="#34495e")
    ax1.grid(False)

    # Axial elevation: 10 equal conceptual cells on each RF rod. All cells
    # retain common RF phase pattern; colors show illustrative independent
    # DC offsets. End plates are drawn as separate boundaries.
    L=10.0; nseg=10; gap=.06; colors=["#728da1","#879daf","#a6b3bd",
       "#d2a568","#cf8b54","#c77852","#c77852","#d2a568","#a6b3bd","#728da1"]
    ax2.add_patch(Rectangle((0,-1.16),L,2.32,fill=False,lw=1.3,edgecolor="#34495e"))
    for j in range(24):
        th=2*np.pi*j/24
        y=1.053*np.sin(th)
        for s in range(nseg):
            x0=s+gap/2
            ax2.add_patch(Rectangle((x0,y-.026),(1-gap),.052,
                facecolor=colors[s],edgecolor="white",lw=.25))
    ax2.axhline(0,color="#202b33",lw=.8,ls="--")
    ax2.axvline(0,color="#34495e",lw=2.2); ax2.axvline(L,color="#34495e",lw=2.2)
    ax2.set_xlim(-.25,L+.25); ax2.set_ylim(-1.38,1.38)
    ax2.set_xlabel(r"$z/R$; total $L/R=10$")
    ax2.set_ylabel(r"rod azimuthal position $y/R$")
    ax2.set_yticks([-1,0,1]); ax2.set_title("finite RF rods / illustrative DC cells")
    ax2.text(5,1.23,"grounded end plates",ha="center",fontsize=8,color="#34495e")
    fem_path=ROOT/"results"/"finite_3d_rf_fem.json"
    if fem_path.exists():
        fem=json.loads(fem_path.read_text(encoding="utf-8"))
        rows=fem["mesh_levels"]
        for idx,(mode,color,label) in enumerate(((2,"#287271",r"$m=+2$"),(-4,"#d1495b",r"$m=-4$"))):
            y=[r["midplane_leading_transfer_sample"][idx]["transfer_magnitude"] for r in rows]
            ax3.plot([r["mesh_hmax_mm"] for r in rows],y,"o-",color=color,lw=1.1,ms=4,label=f"FEM {label}")
            ref={2:.8703168911*0.9**2,-4:.7982172624*0.9**4}[mode]
            ax3.axhline(ref,color=color,lw=.8,ls="--",alpha=.7,label=f"infinite 2D {label}")
        ax3.set_xlabel("tetra mesh $h_{max}$ (mm)")
        ax3.set_ylabel(r"$|V_m(0.9R,z=0)|/U_{rod}$")
        ax3.set_title("RF-only FEM transfer; not converged")
        ax3.legend(frameon=False,fontsize=6.5)
        ax3.grid(True,alpha=.2)
    else:
        ax3.text(.5,.5,"finite 3D FEM pending",ha="center",va="center",transform=ax3.transAxes)
        ax3.set_axis_off()
    fig.suptitle("FIG06  Segmented 24-rod concept and finite-RF FEM check",fontsize=12)
    footer.text(.5,.5,"Left: cross-section. Center: 24 rods projected; DC colors are illustrative, not solved. "
                "Right: finite-rod RF basis only; no segmented-DC solve, end-field refinement, or surface-field certification.",
                ha="center",va="center",fontsize=7.5)
    save(fig,"FIG06_segmented_geometry_schematic")


def fig07():
    f0=np.load(DATA/"ps_mfs1000_p0_source_v17c16.npz")
    f1=np.load(DATA/"ps_mfs1000_p1_eps26_source_v17c16.npz")
    fig=plt.figure(figsize=(10.5,5.2),layout="constrained")
    for col,(data,label) in enumerate(((f0,"P0: no induced polarization"),
                                       (f1,r"P1: real $\epsilon_r=2.6$ screening"))):
        ax=fig.add_subplot(1,2,col+1,projection="3d")
        st=data["state"][::8]
        for i,color in enumerate(("#287271","#d1495b")):
            ax.plot(st[:,i,0],st[:,i,1],st[:,i,2],lw=.65,color=color,
                    label=f"particle {i+1}")
        ax.set(xlabel=r"$x/R$",ylabel=r"$y/R$",zlabel=r"$z/R$",title=label)
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-2.1,.4)
        ax.set_box_aspect((1,1,1.25)); ax.legend(frameon=False,loc="upper left")
    fig.suptitle("FIG07  Inherited 1000-period 2D-MFS + ideal-DC trajectories")
    fig.text(.5,-.01,"V17C.16 source; same frozen start and Coulomb scale. P1 is an illustrative instantaneous real-dielectric model; "
             "neither panel uses finite segmented rods or 3D FEM.",ha="center",fontsize=8)
    save(fig,"FIG07_inherited_pair_trajectories")


def fig08():
    eps0=8.8541878128e-12; e=1.602176634e-19
    rho_air=1.2; eta=1.81e-5; Om=2*np.pi*47.5
    Rref=5e-3; aref=35e-6; rhoref=1050.; qref=400000*e
    mref=4*np.pi*rhoref*aref**3/3
    qmref=qref/mref
    radii=np.geomspace(5e-6,100e-6,220)
    fig,ax=plt.subplots(figsize=(9.3,5.2),layout="constrained")
    # Overlay the mathematical one-particle stability screen as a translucent
    # categorical background; physical curves remain linked through radius,
    # density and assumed surface field.
    with (DATA/"alpha_lambda.csv").open(newline="",encoding="utf-8") as f:
        screen=list(csv.DictReader(f))
    axs=np.array(sorted({float(row["alpha"]) for row in screen}))
    lams=np.array(sorted({float(row["lambda"]) for row in screen}))
    status=np.full((len(lams),len(axs)),-1,dtype=int)
    ix={x:i for i,x in enumerate(axs)}; iy={y:i for i,y in enumerate(lams)}
    for row in screen:
        status[iy[float(row["lambda"])],ix[float(row["alpha"])]]=int(row["status_code"])
    xe=np.r_[axs[0]-(axs[1]-axs[0])/2,(axs[:-1]+axs[1:])/2,axs[-1]+(axs[-1]-axs[-2])/2]
    ye=np.r_[lams[0]-(lams[1]-lams[0])/2,(lams[:-1]+lams[1:])/2,lams[-1]+(lams[-1]-lams[-2])/2]
    from matplotlib.colors import ListedColormap,BoundaryNorm
    screen_colors=["#d9d9d9","#2a9d8f","#457b9d","#e9c46a","#d1495b","#6c757d"]
    ax.pcolormesh(xe,ye,status,cmap=ListedColormap(screen_colors),
                  norm=BoundaryNorm(np.arange(-.5,6.5),6),shading="flat",alpha=.25,zorder=0)
    for rho,label,color in ((1050.,"solid PS, rho=1050 kg m$^{-3}$","#287271"),
                            (112.9237,"hollow-glass density branch","#e09f3e")):
        for Es,ls in ((0.1e6,"--"),(0.5e6,"-"),(1.0e6,":"),(3.0e6,"-.")):
            qm=3*eps0*Es/(rho*radii)
            alpha=qm/qmref
            lam=9*eta/(2*rho*radii**2*Om)
            ax.plot(alpha,lam,color=color,ls=ls,lw=1.0,label="_nolegend_")
    ps=[(1.0,0.21217403,"PS 35 $\\mu$m"),
        (0.3412603,0.12835219,"PS 45 $\\mu$m")]
    for a,l,label in ps:
        ax.scatter(a,l,s=42,marker="o",color="#14213d",zorder=4,label="_nolegend_")
    phys=[(0.0009885101/qmref,0.24778247,"phys0570 Al$_2$O$_3$"),
          (0.00285930394/qmref,2.46236784,"phys0570 hollow glass")]
    for a,l,label in phys:
        ax.scatter(a,l,s=46,marker="D",facecolor="none",edgecolor="#d1495b",lw=1.4,zorder=4,label="_nolegend_")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha=(q/m)/(q/m)_{PS,35\mu m}$")
    ax.set_ylabel(r"$\lambda_S=6\pi\eta a/(m\Omega_1)$")
    ax.set_title("FIG08  Physical q/m–drag linkage; surface-field contours are assumptions")
    ax.grid(True,which="both",alpha=.20)
    from matplotlib.patches import Patch
    screen_handles=[Patch(facecolor=screen_colors[i],alpha=.6,label=label) for i,label in enumerate(
        ("unresolved","POS","NEG","low-turn","aperture contact","numerical failure"))]
    screen_leg=ax.legend(handles=screen_handles,frameon=True,framealpha=.9,fontsize=6.0,
                         ncol=2,loc="upper left",title="background: finite-time map",
                         title_fontsize=6.2)
    ax.add_artist(screen_leg)
    from matplotlib.lines import Line2D
    line_styles=(("--","0.1"),("-","0.5"),(":","1"),("-.","3"))
    handles=[Line2D([],[],color="#287271",lw=1.6,label="solid PS, density 1050"),
             Line2D([],[],color="#e09f3e",lw=1.6,label="hollow glass, density 112.9")]
    handles += [Line2D([],[],color="#444444",lw=1.2,ls=ls,label=rf"$E_s={val}\,\mathrm{{MV/m}}$")
                for ls,val in line_styles]
    handles += [Line2D([],[],color="#14213d",marker="o",ls="none",label="PS 35 μm"),
                Line2D([],[],color="#14213d",marker="o",ls="none",label="PS 45 μm"),
                Line2D([],[],color="#d1495b",marker="D",ls="none",mfc="none",label="phys0570 Al₂O₃"),
                Line2D([],[],color="#d1495b",marker="D",ls="none",mfc="none",label="phys0570 glass")]
    ax.legend(handles=handles,frameon=False,fontsize=5.8,ncol=1,loc="upper left",bbox_to_anchor=(1.01,1.0))
    save(fig,"FIG08_physical_feasibility_linked_parameters")
    rows=[]
    for rho,material in ((1050.,"PS"),(112.9237,"hollow glass density branch")):
        for a in (5e-6,10e-6,20e-6,35e-6,45e-6,65e-6,100e-6):
            for Es in (.1e6,.5e6,1e6,3e6):
                qm=3*eps0*Es/(rho*a)
                lam=9*eta/(2*rho*a*a*Om)
                rows.append({"material_assumption":material,"radius_m":a,"density_kg_m3":rho,
                             "sphere_surface_field_assumption_V_m":Es,"alpha_vs_PS35":qm/qmref,
                             "lambda_Stokes_at_47p5Hz":lam,
                             "charge_e":4*np.pi*eps0*a*a*Es/e,
                             "status":"linked dimensional screen; surface-field contour not a validated charge-stability limit"})
    with (DATA/"fig08_linked_physical_parameter_curves.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__=="__main__":
    FIG.mkdir(parents=True,exist_ok=True)
    fig05(); fig06(); fig07(); fig08()
    print("wrote FIG05-FIG08 and source data")
