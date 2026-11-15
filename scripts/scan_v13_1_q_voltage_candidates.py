"""Coarse Floquet gate for pure-Q settings under a 250 V peak limit."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.mass_filter_v13 import linear_status
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1"
ALPHA=.027000013157894725; BETA=.175
MUS=np.linspace(.85,1.15,61)
rows=[]
for u2 in np.arange(150.,246.,5.):
  for vdc in np.arange(0.,min(101.,250.-u2)+1e-9,5.):
    if u2+vdc>250.: continue
    a=ALPHA*vdc/29.3143; b=BETA*u2/190.
    center=linear_status(a,b)
    states=np.array([linear_status(a/m,b/m)["status"]=="strictly_stable" for m in MUS])
    i=int(np.argmin(abs(MUS-1.)))
    lo=hi=i
    if states[i]:
      while lo>0 and states[lo-1]:lo-=1
      while hi+1<len(MUS) and states[hi+1]:hi+=1
      low,high=float(MUS[lo]),float(MUS[hi]);width=high-low
    else: low=high=None;width=0.
    rows.append(dict(U2_V=float(u2),VDC_V=float(vdc),peak_V=float(u2+vdc),alpha=float(a),beta=float(b),
       status_mu1=center["status"],margin_x=center["margin_x"],margin_y=center["margin_y"],
       stable_mu_low=low,stable_mu_high=high,stable_mu_width_grid=width))
with (OUT/"q_voltage_candidate_scan.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
eligible=[r for r in rows if r["status_mu1"]=="strictly_stable" and min(r["margin_x"],r["margin_y"])>=.03 and r["stable_mu_width_grid"]>0]
selected=min(eligible,key=lambda r:(r["stable_mu_width_grid"],-r["peak_V"])) if eligible else None
(OUT/"q_voltage_candidate_selected.json").write_text(json.dumps(selected,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"n_candidates":len(rows),"n_eligible":len(eligible),"selected":selected},indent=2))
