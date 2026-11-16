"""Estimate only bracketed retention peaks using independent Sobol replicates."""
from __future__ import annotations
import argparse,csv,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    cols=[]
    for r in rows:cols.extend(k for k in r if k not in cols)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def estimate(values):
    x=np.asarray(values,float);m=float(np.mean(x))
    if len(x)<2:return m,math.nan,math.nan
    h=float(student_t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/math.sqrt(len(x)))
    return m,m-h,m+h
def crossings(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    imax=np.flatnonzero(y==np.max(y));left_i=int(imax[0]);right_i=int(imax[-1]);peak=float(np.max(y));half=peak/2
    li=next((j for j in range(left_i-1,-1,-1) if y[j]<half),None)
    ri=next((j for j in range(right_i+1,len(y)) if y[j]<half),None)
    def interp(i,j):return float(x[i]+(half-y[i])*(x[j]-x[i])/(y[j]-y[i]))
    left=interp(li,li+1) if li is not None else math.nan
    right=interp(ri-1,ri) if ri is not None else math.nan
    return peak,float(np.mean(x[imax])),left,right,(right-left if math.isfinite(left) and math.isfinite(right) else math.nan)
def main():
    p=argparse.ArgumentParser();p.add_argument("--directory",required=True);args=p.parse_args()
    d=Path(args.directory)
    if d.is_absolute() or ".." in d.parts:raise SystemExit("directory must stay below the V13.3 atlas")
    out=BASE/d
    rows=read(out/"replicate_retention.csv")
    group=defaultdict(dict)
    for r in rows:
        group[(r["model"],int(r["replica"]))][(float(r["mu"]),float(r["phase_deg"]))]=float(r["P_lower"])
    models=sorted({k[0] for k in group});replicas=sorted({k[1] for k in group})
    phases=("phase_average",0.,90.)
    per_rep=[]
    for model in models:
      for rep in replicas:
       data=group[(model,rep)];mus=sorted({key[0] for key in data})
       for phase in phases:
        profile=[]
        for mu in mus:
            if phase=="phase_average":val=(data[(mu,0.)]+data[(mu,90.)])/2
            else:val=data[(mu,float(phase))]
            profile.append(val)
        peak,mupeak,left,right,width=crossings(mus,np.asarray(profile,float))
        per_rep.append({"model":model,"replica":rep,"phase":phase,"peak_P_ret":peak,"peak_mu":mupeak,
                        "left_half_max_mu":left,"right_half_max_mu":right,"FWHM_mu":width,
                        "Rm_replica":mupeak/width if width>0 and math.isfinite(width) else math.nan,
                        "bounded_both_sides":math.isfinite(left) and math.isfinite(right),
                        "mass_grid_min":min(mus),"mass_grid_max":max(mus),"grid_step_min":min(np.diff(mus)) if len(mus)>1 else math.nan,
                        "threshold":"half of each independent replicate's own sampled maximum; piecewise linear only between computed mass nodes"})
    write(out/"fwhm_per_replica.csv",per_rep)
    summary=[]
    for model in models:
      for phase in phases:
        rr=[r for r in per_rep if r["model"]==model and r["phase"]==phase]
        bounded=[r for r in rr if r["bounded_both_sides"]]
        if len(bounded)==len(rr):
            pmean,plo,phi=estimate([r["peak_P_ret"] for r in rr])
            mmean,mlo,mhi=estimate([r["peak_mu"] for r in rr])
            wmean,wlo,whi=estimate([r["FWHM_mu"] for r in rr])
            rmean,rlo,rhi=estimate([r["Rm_replica"] for r in rr])
            rm_ratio=mmean/wmean if wmean>0 else math.nan
            status="bounded on both sides for all independent replicates"
        else:
            pmean=plo=phi=mmean=mlo=mhi=wmean=wlo=whi=rmean=rlo=rhi=rm_ratio=math.nan
            status="FWHM not estimated: at least one replicate lacks a computed half-height crossing"
        summary.append({"model":model,"phase":phase,"n_replicates":len(rr),"n_bounded_replicates":len(bounded),
                        "peak_P_ret_mean":pmean,"peak_P_ret_t95_low":plo,"peak_P_ret_t95_high":phi,
                        "peak_mu_mean":mmean,"peak_mu_t95_low":mlo,"peak_mu_t95_high":mhi,
                        "FWHM_mu_mean":wmean,"FWHM_mu_t95_low":wlo,"FWHM_mu_t95_high":whi,
                        "Rm_mean_of_replica_ratios":rmean,"Rm_t95_low":rlo,"Rm_t95_high":rhi,
                        "Rm_ratio_of_mean_peak_mu_to_mean_width":rm_ratio,"status":status,
                        "uncertainty":"Student-t 95% interval over independent scrambled-Sobol replicate estimates; mass interpolation only for crossing between directly computed neighbors"})
    write(out/"fwhm_summary.csv",summary)
    print({"directory":str(d),"replicate_profiles":len(per_rep),"summary_rows":len(summary),"status":[r["status"] for r in summary]})
if __name__=="__main__":main()
