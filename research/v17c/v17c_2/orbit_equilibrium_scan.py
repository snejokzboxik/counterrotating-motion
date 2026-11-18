"""Direct RF trajectories initialized from stable averaged circular points."""
import json
import numpy as np
from single_particle import RESULTS, load_geometry, integrate, classify

def main():
    equilibria=json.loads((RESULTS/'averaged_orbit_equilibria.json').read_text())
    requests=[('6_ABC_thin',1,.40),('6_ABC_thin',-1,.72),
              ('9_ABC_moderate',1,.55),('9_ABC_moderate',-1,.92),
              ('9_ABC_thin',1,.50),('9_ABC_thin',-1,.84)]
    rows=[]; saved={}
    for name,sign,target in requests:
        d,modes,coeff=load_geometry(name)
        cand=[x for x in equilibria if x['name']==name and x['lambda_']==.0005 and x['stable'] and x['torque_sign']==sign]
        if not cand:
            print('no averaged candidate',name,sign,flush=True);continue
        x=min(cand,key=lambda y:abs(y['r']-target)); r=x['r'];amp=x['amplitude'];omega=x['omega'];lam=x['lambda_']
        # Phase/azimuth scan around the mean guiding-center fixed point.
        for theta in (0.,.7,1.4):
          for phase in (0.,2*np.pi/3,4*np.pi/3):
            tr,mx,ex=integrate(r,theta,0.,omega,phase,modes,coeff,amp,lam,300,256,True)
            diag=classify(tr,300,ex)
            row=dict(name=name,sign=sign,requested_r=target,r0=r,amp=amp,lambda_=lam,omega0=omega,
                     theta0=theta,phase0=phase,max_radius=float(mx),**diag)
            rows.append(row)
            if diag['state'] in ('positive','negative') and abs(diag['turns_total'])>=10:
                saved[f'{name}_{sign}_{theta}_{phase}']=tr
            print(name,sign,'theta',theta,'phase',round(phase,3),diag['state'],
                  round(diag['turns_total'],2),round(diag['min_radius'],3),round(mx,3),flush=True)
        (RESULTS/'orbit_equilibrium_scan_checkpoint.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (RESULTS/'orbit_equilibrium_scan.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if saved:
        np.savez_compressed(RESULTS/'orbit_equilibrium_scan_candidates.npz',**saved)

if __name__=='__main__':main()
