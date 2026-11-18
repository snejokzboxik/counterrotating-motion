"""Refine the predicted negative-torque annulus with lower RF amplitudes."""
import json
import numpy as np
from single_particle import load_geometry,integrate,classify,RESULTS
from guided_search import pseudo_potential,omega_circular

def main():
    RESULTS.mkdir(exist_ok=True);rows=[]
    cfg=[('6_ABC_moderate',.68),('6_ABC_thin',.61),('9_ABC_moderate',.84),('9_ABC_thin',.77),('12_ABC_moderate',.92)]
    for name,r in cfg:
        d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==d['m'])[0][0]])
        base=.1044/lead
        for af in (.02,.05,.1,.2):
            amp=base*af
            w=omega_circular(r,modes,coeff,amp)
            for lam in (.0001,.0005,.002):
                for wf in (.8,1.,1.2):
                    for theta in (.17,1.03):
                        tr,rmax,escaped=integrate(r,theta,0.,-w*wf,0.,modes,coeff,amp,lam,2400,64,True)
                        row=dict(name=name,r0=r,amp_factor=af,lambda_=lam,omega0=-w*wf,
                                 omega_circular=w*wf,max_radius=float(rmax),**classify(tr,2400,escaped))
                        rows.append(row)
        (RESULTS/'outer_branch_checkpoint.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(name,'done',len(rows),flush=True)
    (RESULTS/'outer_branch_search.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    good=[x for x in rows if x['state']=='negative' and abs(x['turns_total'])>=5]
    (RESULTS/'outer_branch_candidates.json').write_text(json.dumps(good,indent=2),encoding='utf-8')
    print('negative candidates',len(good))

if __name__=='__main__':main()
