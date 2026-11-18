"""Long-horizon checks and perturbations of the two opposite single-particle branches."""
import json
import numpy as np
from single_particle import load_geometry,integrate,classify,RESULTS

def main():
    cases=[('positive_inner',.34,.7,.0005,.12470596599090433),
           ('negative_outer',.61,.02,.0005,-.005445940846130374)]
    summary=[]
    for label,r,af,lam,omega in cases:
        d,modes,coeff=load_geometry('6_ABC_thin');lead=abs(coeff[np.where(modes==d['m'])[0][0]])
        base=.1044/lead*af
        variants=[('nominal',r,omega,base),('position_plus_1pct',r*1.01,omega,base),
                  ('speed_plus_1pct',r,omega*1.01,base),('amplitude_plus_5pct',r,omega,base*1.05)]
        for tag,rr,ww,aa in variants:
            tr,rmax,escaped=integrate(rr,.371,0.,ww,0.,modes,coeff,aa,lam,8000,96,True)
            diag=classify(tr,8000,escaped)
            row=dict(case=label,variant=tag,r0=rr,omega0=ww,amp=aa,lambda_=lam,
                     max_radius=float(rmax),**diag)
            summary.append(row)
            if tag=='nominal':
                angle=np.unwrap(np.arctan2(tr[:,1],tr[:,0]))
                np.savez_compressed(RESULTS/f'{label}_8000RF.npz',tr=tr,angle=angle,
                                    rf_periods=np.arange(len(tr)),row=json.dumps(row))
            print(label,tag,diag['state'],diag['turns_total'],diag['turns_windows'],diag['min_radius'],rmax,flush=True)
    (RESULTS/'single_long_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
