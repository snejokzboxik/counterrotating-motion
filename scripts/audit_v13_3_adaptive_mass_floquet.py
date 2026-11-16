"""Compute exact linear Floquet status along atlas-selected mass rays."""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path,rows):
    if not rows:return
    fields=[]
    for row in rows:fields.extend(k for k in row if k not in fields)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--points",default=str(BASE/"fem_nonlinear_gain_points.csv"))
    p.add_argument("--output-name",default="mass_ray_floquet_adaptive")
    p.add_argument("--mu-min",type=float,default=.8)
    p.add_argument("--mu-max",type=float,default=1.2)
    p.add_argument("--mu-step",type=float,default=.005)
    args=p.parse_args()
    if Path(args.output_name).name!=args.output_name:p.error("output-name must be one file stem")
    sys.path.insert(0,str(ROOT/"src"))
    from three_phase_trap.mass_filter_v13 import monodromy
    points=read(Path(args.points))
    mus=np.round(np.arange(args.mu_min,args.mu_max+args.mu_step/2,args.mu_step),12)
    rows=[];edges=[];max_det_error=0.
    for point in points:
        a0,b0,k0=(float(point[k]) for k in ("alpha","beta","kappa"))
        def margins(mu):
            values=[]
            mats=[]
            for axis in ("x","y"):
                matrix=monodromy(a0/mu,b0/mu,axis,rtol=2e-11,atol=2e-13)
                mats.append(matrix);values.append(2.-abs(float(np.trace(matrix))))
            return values,mats
        point_rows=[]
        for mu in mus:
            vals,mats=margins(float(mu));dets=[float(np.linalg.det(m)) for m in mats]
            max_det_error=max(max_det_error,*(abs(d-1.) for d in dets))
            strict=all(v>1e-10 for v in vals)
            boundary=not strict and all(v>=-1e-8 for v in vals)
            eig=[np.linalg.eigvals(m) for m in mats]
            row={"candidate_id":point["candidate_id"],"mu":float(mu),
                 "alpha":a0/float(mu),"beta":b0/float(mu),"kappa_RO":k0/float(mu),
                 "margin_x":vals[0],"margin_y":vals[1],
                 "det_x":dets[0],"det_y":dets[1],
                 "max_multiplier_modulus_x":float(max(abs(v) for v in eig[0])),
                 "max_multiplier_modulus_y":float(max(abs(v) for v in eig[1])),
                 "status":"strictly_stable" if strict else "boundary_or_parabolic" if boundary else "unstable",
                 "method":"V13 monodromy in x and y; kappa has zero linear Jacobian"}
            rows.append(row);point_rows.append(row)
        for left,right in zip(point_rows[:-1],point_rows[1:]):
            stable_left=left["status"]=="strictly_stable"
            stable_right=right["status"]=="strictly_stable"
            if stable_left!=stable_right:
                def fn(mu):return min(margins(mu)[0])
                root=brentq(fn,left["mu"],right["mu"],xtol=1e-11,rtol=1e-11)
                edges.append({"candidate_id":point["candidate_id"],"mu_boundary":root,
                              "alpha_boundary":a0/root,"beta_boundary":b0/root,
                              "transition":f"{left['status']} -> {right['status']}",
                              "note":"identical for Q and Q+RO linearization"})
    stem=BASE/args.output_name
    write(stem.with_name(stem.name+".csv"),rows)
    write(stem.with_name(stem.name+"_boundaries.csv"),edges)
    meta={"points":len(points),"mu_nodes_per_point":len(mus),"boundary_count":len(edges),
          "mu_grid":[float(mus[0]),float(mus[-1]),args.mu_step],
          "max_abs_det_minus_one":max_det_error,
          "interpretation":"exact linear center status; does not classify finite-amplitude nonlinear retention"}
    stem.with_name(stem.name+"_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
