"""Render fixed-voltage mass rays over the sampled V13.3 3D atlas."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def main():
    manifest=json.loads((BASE/"mass_ray_manifest.json").read_text(encoding="utf-8"))
    rays=manifest["fixed_voltage_rays"]
    fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(111,projection="3d")
    colors=("#b23a48","#286f9b","#dd8b24","#5b8e52")
    mus=np.linspace(.8,1.2,81)
    for ray,color in zip(rays,colors):
        a0,b0,k0=(float(ray[k]) for k in ("alpha","beta","kappa"))
        a,b,k=a0/mus,b0/mus,k0/mus
        ax.plot(a,b,k,color=color,lw=2.1,label=ray["candidate_id"].replace("_"," "))
        ax.scatter([a[20],a[40],a[60]],[b[20],b[40],b[60]],[k[20],k[40],k[60]],
                   color=color,s=19,depthshade=False)
        ax.text(a[40],b[40],k[40],"μ=1",color=color,fontsize=8)
    # Pure-Q reference ray from the physically selected V13.1 working point.
    qalpha,qbeta=.027,.175
    ax.plot(qalpha/mus,qbeta/mus,np.zeros_like(mus),color="#171717",lw=1.8,ls="--",label="Q reference (κ=0)")
    ax.scatter([qalpha],[qbeta],[0],marker="*",s=90,color="#171717",depthshade=False)
    ax.set_xlabel("α = α₀/μ");ax.set_ylabel("β = β₀/μ");ax.set_zlabel("κ = κ₀/μ")
    ax.set_title("V13.3 fixed-voltage mass rays in (α, β, κ); 0.8 ≤ μ ≤ 1.2")
    ax.legend(loc="upper left",fontsize=8,frameon=False)
    ax.view_init(elev=22,azim=-56);fig.tight_layout()
    stem=BASE/"mass_rays_in_parameter_atlas"
    for ext in ("png","pdf","svg"):fig.savefig(stem.with_suffix("."+ext),dpi=210,bbox_inches="tight")
    plt.close(fig)
    print(stem)

if __name__=="__main__":main()
