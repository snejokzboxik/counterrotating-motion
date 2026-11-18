"""Save selected off-attractor trajectories used in the G2 robustness test."""
import json
import numpy as np
from single_particle import RESULTS,load_geometry,integrate,classify

def main():
    d,modes,coeff=load_geometry('6_ABC_thin');lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
    base=dict(r=.34,theta=.371,omega=.12470596599090433,alpha=.07308,lam=.0005)
    cases=[('radius_plus_1pct',base['r']*1.01,base['omega'],base['alpha'],base['theta'],0.),
           ('alpha_plus_0p5pct',base['r'],base['omega'],base['alpha']*1.005,base['theta'],0.)]
    rows=[]
    for label,r,w,a,theta,phase in cases:
        tr,mx,ex=integrate(r,theta,0.,w,phase,modes,coeff,a/lead,base['lam'],8000,96,True)
        ang=np.unwrap(np.arctan2(tr[:,1],tr[:,0]));row=dict(label=label,r0=r,omega0=w,alpha=a,lambda_=base['lam'],theta0=theta,
           phase0=phase,max_radius=float(mx),**classify(tr,8000,ex))
        rows.append(row);np.savez_compressed(RESULTS/f'G2_{label}_8000RF.npz',rf_period=np.arange(len(tr)),state=tr,theta=ang,
           parameters=json.dumps(row))
        print(row,flush=True)
    (RESULTS/'positive_perturbation_tracks.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
