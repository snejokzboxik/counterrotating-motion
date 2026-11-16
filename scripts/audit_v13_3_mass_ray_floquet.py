"""Exact linear Floquet classification along fixed-voltage V13.3 mass rays."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
MUS=np.round(np.arange(.8,1.2000001,.005),12)
def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    cols=[]
    for r in rows:cols.extend(k for k in r if k not in cols)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def main():
    sys.path.insert(0,str(ROOT/"src"))
    from three_phase_trap.mass_filter_v13 import monodromy
    points=read(BASE/"fem_control_points.csv");rows=[]
    for point in points:
      a0,b0,k0=map(float,(point["alpha"],point["beta"],point["kappa"]))
      for mu in MUS:
        a,b=a0/mu,b0/mu
        matrices=[monodromy(a,b,axis,rtol=2e-11,atol=2e-13) for axis in ("x","y")]
        traces=[float(np.trace(m)) for m in matrices]
        dets=[float(np.linalg.det(m)) for m in matrices]
        margins=[2.-abs(x) for x in traces]
        strictly=all(v>1e-10 for v in margins)
        boundary=not strictly and all(v>=-1e-8 for v in margins)
        status="strictly_stable" if strictly else "boundary_or_parabolic" if boundary else "unstable"
        eig=[np.linalg.eigvals(m) for m in matrices]
        rows.append({"candidate_id":point["candidate_id"],"mu":float(mu),"alpha":a,"beta":b,
                     "kappa_RO":k0/mu,"Q_and_RO_linear_status":status,
                     "trace_x":traces[0],"trace_y":traces[1],"det_x":dets[0],"det_y":dets[1],
                     "margin_x":margins[0],"margin_y":margins[1],
                     "max_multiplier_modulus_x":float(max(abs(x) for x in eig[0])),
                     "max_multiplier_modulus_y":float(max(abs(x) for x in eig[1])),
                     "determinant_error_max":max(abs(d-1.) for d in dets),
                     "method":"independent exact 2x2 monodromy in both transverse coordinates; octupole has zero linear Jacobian"})
    write(BASE/"mass_ray_floquet.csv",rows)
    # Bracket all sampled changes of strict stability along each ray by bisection.
    from scipy.optimize import brentq
    edges=[]
    for point in points:
      a0,b0=map(float,(point["alpha"],point["beta"]))
      def status_margin(mu):
        vals=[]
        for axis in ("x","y"):
            mat=monodromy(a0/mu,b0/mu,axis,rtol=2e-11,atol=2e-13)
            vals.append(2.-abs(float(np.trace(mat))))
        return min(vals)
      group=[r for r in rows if r["candidate_id"]==point["candidate_id"]]
      for left,right in zip(group[:-1],group[1:]):
        s0=left["Q_and_RO_linear_status"]=="strictly_stable";s1=right["Q_and_RO_linear_status"]=="strictly_stable"
        if s0!=s1:
            root=brentq(status_margin,float(left["mu"]),float(right["mu"]),xtol=1e-11,rtol=1e-11)
            edges.append({"candidate_id":point["candidate_id"],"mu_boundary":root,
                          "alpha_boundary":a0/root,"beta_boundary":b0/root,
                          "transition":f"{left['Q_and_RO_linear_status']} -> {right['Q_and_RO_linear_status']}",
                          "note":"same boundary for Q and Q+RO at the same α,β; κ does not enter the linearization"})
    write(BASE/"mass_ray_floquet_boundaries.csv",edges)
    metadata={"mu_grid":[float(MUS[0]),float(MUS[-1]),float(MUS[1]-MUS[0])],"nodes":len(rows),
              "boundary_count":len(edges),"max_determinant_error":max(float(r["determinant_error_max"]) for r in rows),
              "source":"V13 Mathieu/Floquet monodromy implementation, both x and y",
              "interpretation":"exact linear center status along fixed-voltage mass rays; not a nonlinear retention classification"}
    (BASE/"mass_ray_floquet_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata,indent=2))
if __name__=="__main__":main()
