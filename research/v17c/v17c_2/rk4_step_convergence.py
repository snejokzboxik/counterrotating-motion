"""Step-doubling convergence and balance residuals for representative runs."""
import json
import numpy as np
from full_rf_balances import integrate_balance
from single_particle import RESULTS,load_geometry

def main():
    cases=[('6_ABC_moderate',.6397,np.pi/2,-.098075855,.05,.0005,800),
           ('6_ABC_thin',.34,.371,.12470596599090433,.07308,.0005,1200)]
    out=[]
    for name,r,theta,omega,alpha,lam,periods in cases:
        d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]]);amp=alpha/lead
        runs={}
        for steps in (64,128,256):
            tr,E,D,T,TD,angle=integrate_balance(r,theta,omega,0.,modes,coeff,amp,lam,periods,steps)
            K=.5*(tr[:,2]**2+tr[:,3]**2);L=tr[:,0]*tr[:,3]-tr[:,1]*tr[:,2]
            runs[steps]=dict(tr=tr,E=E,D=D,T=T,TD=TD,angle=angle,
              er=float((K-K[0]-E+D)[-1]),lr=float((L-L[0]-T-TD)[-1]))
        ref=runs[256]
        row=dict(name=name,periods=periods,steps={})
        for steps in (64,128,256):
            z=runs[steps]
            row['steps'][str(steps)]=dict(max_state_diff_vs_256=float(np.max(np.abs(z['tr']-ref['tr']))),
               final_state_diff=float(np.max(np.abs(z['tr'][-1]-ref['tr'][-1]))),
               turns=float((z['angle'][-1]-z['angle'][0])/(2*np.pi)),
               rf_work=float(z['E'][-1]),drag_loss=float(z['D'][-1]),energy_residual=z['er'],angular_residual=z['lr'])
        out.append(row);print(row,flush=True)
    (RESULTS/'rk4_step_convergence.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
