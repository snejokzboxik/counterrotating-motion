"""Render named manuscript figures from the archived screening CSV outputs."""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from run_screening_maps import _plot_categorical

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; FIG=ROOT/"figures"

def render_pair():
    with (DATA/"alpha_pair_summary.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    xs=np.array(sorted({float(r["alpha1"]) for r in rows}))
    ys=np.array(sorted({float(r["alpha2"]) for r in rows}))
    lookup={(float(r["alpha1"]),float(r["alpha2"])):r for r in rows}
    frac=np.array([[float(lookup[x,y]["cr_fraction_panel"]) for x in xs] for y in ys])
    width=np.array([[float(lookup[x,y]["wilson95_high"])-float(lookup[x,y]["wilson95_low"])
                     for x in xs] for y in ys])
    dom=np.array([[int(lookup[x,y]["dominant_outcome_code"]) for x in xs] for y in ys])
    ext=[xs[0]-(xs[1]-xs[0])/2,xs[-1]+(xs[1]-xs[0])/2,
         ys[0]-(ys[1]-ys[0])/2,ys[-1]+(ys[1]-ys[0])/2]
    fig,axs=plt.subplots(1,3,figsize=(15,5.1),layout="constrained",
                         gridspec_kw={"width_ratios":[1,1,1.18]})
    for ax,title,data,cmap,vmin,vmax in (
        (axs[0],"(a) Counterrotation fraction",frac,"viridis",0,1),
        (axs[1],"(b) Wilson 95% interval width",width,"magma",0,1)):
        im=ax.imshow(data,origin="lower",aspect="auto",interpolation="nearest",
                     extent=ext,cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_xlabel(r"$\alpha_1=(q_1/m_1)/(q_*/m_*)$")
        ax.set_ylabel(r"$\alpha_2=(q_2/m_2)/(q_*/m_*)$")
        ax.set_title(title)
        cb=fig.colorbar(im,ax=ax)
        cb.set_label("CR starts / 4" if data is frac else "upper − lower bound")
    colors=["#808080","#2a9d8f","#457b9d","#e9c46a","#8f5aa2",
            "#f4a261","#d1495b","#7f0000","#222222","#9c6644","#c7c7c7"]
    cmap=ListedColormap(colors); norm=BoundaryNorm(np.arange(-.5,11.5),cmap.N)
    im=axs[2].imshow(dom,origin="lower",aspect="auto",interpolation="nearest",
                     extent=ext,cmap=cmap,norm=norm)
    axs[2].set_xlabel(r"$\alpha_1=(q_1/m_1)/(q_*/m_*)$")
    axs[2].set_ylabel(r"$\alpha_2=(q_2/m_2)/(q_*/m_*)$")
    axs[2].set_title("(c) Plurality outcome, same 4 starts")
    cb=fig.colorbar(im,ax=axs[2],ticks=np.arange(11))
    cb.ax.set_yticklabels(["unresolved","CR +−","CR −+","CO ++","CO −−",
       "bound/low-turn","one lost","both lost","collision","aperture","mixed/tie"])
    fig.suptitle("Two-particle finite-horizon screening; n=4 starts per grid point",fontsize=12)
    for extn in ("png","pdf","svg"):
        fig.savefig(FIG/f"alpha_pair_panel_fraction.{extn}",dpi=600 if extn=="png" else None,
                    bbox_inches="tight")
    plt.close(fig)

def render_single_maps():
    with (ROOT/"results"/"screening_summary.json").open(encoding="utf-8") as f:
        summary=json.load(f)
    specs={
      "alpha_lambda":("lambda",r"(a) Mathematical screening: $(\alpha,\lambda)$"),
      "alpha_nu":("nu",r"(b) Mathematical screening: $(\alpha,\nu)$"),
      "alpha_b_over_a":("b_over_a",r"(c) Mathematical screening: $(\alpha,b/a)$"),
      "alpha_pol_PS2":("pol_multiplier",r"(d) Polarization sensitivity: $(\alpha,\chi/\chi_{PS,2})$")}
    for name,(ykey,title) in specs.items():
        with (DATA/f"{name}.csv").open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        m=next(x for x in summary["maps"] if x["map"]==name)
        _plot_categorical(rows,"alpha",ykey,"status_code",np.asarray(m["x_values"]),
                          np.asarray(m["y_values"]),title,name)

def main():
    render_single_maps()
    render_pair()
    named={
      "FIG01_single_particle_alpha_lambda":"alpha_lambda",
      "FIG02_RF_parameter_alpha_b_over_a":"alpha_b_over_a",
      "FIG03_two_particle_stability":"alpha_pair_panel_fraction",
      "FIG04_polarization_sensitivity":"alpha_pol_PS2",
      "FIG05_axial_DC_transfer":"FIG05_axial_DC_transfer",
      "FIG06_segmented_geometry":"FIG06_segmented_geometry_schematic",
      "FIG07_representative_trajectories":"FIG07_inherited_pair_trajectories",
      "FIG08_physical_feasibility":"FIG08_physical_feasibility_linked_parameters",
      "FIGS01_frequency_ratio_alpha_nu":"alpha_nu"}
    for dst,src in named.items():
        if dst==src:
            continue
        for ext in ("png","pdf","svg"):
            shutil.copyfile(FIG/f"{src}.{ext}",FIG/f"{dst}.{ext}")
    (DATA/"figure_index.json").write_text(json.dumps({"named_figure_stems":named,
      "formats":["PDF","SVG","PNG (600 dpi)"],
      "additional_screen":"FIGS01_frequency_ratio_alpha_nu is an extra one-particle frequency-ratio section."},indent=2)+"\n",encoding="utf-8")
    print(f"rendered pair panel and wrote {len(named)} named figure sets")

if __name__=="__main__": main()
