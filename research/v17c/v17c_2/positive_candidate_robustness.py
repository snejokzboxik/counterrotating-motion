"""Structured perturbation sensitivity of the long G2 positive orbit."""
import json
import numpy as np
from single_particle import RESULTS,load_geometry,integrate,classify

def main():
    name='6_ABC_thin';r=.34;theta=.371;omega=.12470596599090433;alpha=.07308;lam=.0005
    d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]]);amp=alpha/lead
    vars=[]
    for p in (.001,.0025,.005,.01):vars += [(f'r_plus_{p:.4f}',r*(1+p),omega,amp,lam,theta,0.),(f'r_minus_{p:.4f}',r*(1-p),omega,amp,lam,theta,0.)]
    for p in (.001,.005,.01):vars += [(f'omega_plus_{p:.4f}',r,omega*(1+p),amp,lam,theta,0.),(f'omega_minus_{p:.4f}',r,omega*(1-p),amp,lam,theta,0.)]
    for p in (.005,.01,.02,.05):vars += [(f'alpha_plus_{p:.4f}',r,omega,amp*(1+p),lam,theta,0.),(f'alpha_minus_{p:.4f}',r,omega,amp*(1-p),lam,theta,0.)]
    for p in (.01,.05,.1):vars += [(f'lambda_plus_{p:.3f}',r,omega,amp,lam*(1+p),theta,0.)]
    for p in (.001,.01):vars += [(f'theta_plus_{p:.3f}',r,omega,amp,lam,theta+p,0.),(f'phase_plus_{p:.3f}',r,omega,amp,lam,theta,p)]
    out=[];periods=8000;steps=96
    for tag,rr,w,a,l,th,phase in [('nominal',r,omega,amp,lam,theta,0.),*vars]:
        tr,mx,ex=integrate(rr,th,0.,w,phase,modes,coeff,a,l,periods,steps,True)
        row=dict(variant=tag,r0=rr,omega0=w,alpha=a*lead,lambda_=l,theta0=th,phase0=phase,
                 max_radius=float(mx),**classify(tr,periods,ex))
        out.append(row);print(tag,row['state'],round(row['turns_total'],2),round(row['min_radius'],5),round(mx,5),row['turns_windows'],flush=True)
    (RESULTS/'positive_candidate_robustness.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
