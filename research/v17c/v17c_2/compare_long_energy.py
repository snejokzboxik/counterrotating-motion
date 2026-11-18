"""Long and energetic checks of best short single-particle branches."""
import json
import numpy as np
from full_rf_balances import integrate_balance
from single_particle import RESULTS,load_geometry,integrate

def main():
    outer=json.loads((RESULTS/'outer_phase_search.json').read_text())
    cases=[]
    for geom in ['6_ABC_moderate','6_ABC_thin','9_ABC_moderate']:
        good=[x for x in outer if x['name']==geom and x['state']=='negative']
        if good: cases.append(max(good,key=lambda x:abs(x['turns_total'])))
    cases.append(dict(name='6_ABC_thin',r0=.34,alpha=.1044*.7,omega0=.12470596599090433,
                      theta0=.371,lambda_=.0005,state='positive'))
    rows=[];periods=2000;steps=96
    for i,c in enumerate(cases):
        name=c['name'];d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
        alpha=c.get('alpha',.0);amp=alpha/lead;lam=c.get('lambda_',.0005)
        check,checkmax,escaped=integrate(c['r0'],c['theta0'],0.,c['omega0'],0.,modes,coeff,amp,lam,periods,steps,True)
        if escaped:
            row=dict(name=name,branch=c.get('state','negative'),r0=c['r0'],alpha=alpha,omega0=c['omega0'],
                     theta0=c['theta0'],lambda_=lam,periods=periods,steps=steps,status='escape',
                     first_escape_RF_period=len(check)-1,max_radius=float(checkmax))
            rows.append(row);print(row,flush=True);continue
        tr,E,D,T,TD,th=integrate_balance(c['r0'],c['theta0'],c['omega0'],0.,modes,coeff,amp,lam,periods,steps)
        rr=np.hypot(tr[:,0],tr[:,1]);K=.5*(tr[:,2]**2+tr[:,3]**2);L=tr[:,0]*tr[:,3]-tr[:,1]*tr[:,2]
        er=K-K[0]-E+D;lr=L-L[0]-T-TD
        crossing=np.flatnonzero(rr<.05)
        row=dict(name=name,branch=c.get('state','negative'),r0=c['r0'],alpha=alpha,omega0=c['omega0'],theta0=c['theta0'],
            lambda_=lam,periods=periods,steps=steps,turns=float((th[-1]-th[0])/(2*np.pi)),
            min_radius=float(rr.min()),max_radius=float(rr.max()),radius_at=[float(rr[q]) for q in [0,100,500,1000,1500,2000]],
            first_r_below_0p05=int(crossing[0]) if len(crossing) else None,
            initial_K=float(K[0]),final_K=float(K[-1]),rf_work=float(E[-1]),drag_loss=float(D[-1]),
            energy_residual=float(er[-1]),rf_angular_impulse=float(T[-1]),drag_angular_impulse=float(TD[-1]),angular_residual=float(lr[-1]),status='finite')
        rows.append(row)
        np.savez_compressed(RESULTS/f'long_energy_{i}_{name}.npz',rf_period=np.arange(periods+1),state=tr,theta=th,
                            K=K,RF_work=E,drag_loss=D,RF_angular_impulse=T,drag_angular_impulse=TD)
        print(row,flush=True)
    (RESULTS/'long_energy_comparison.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__':main()
