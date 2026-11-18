"""Validate the strongest negative starts for five drag e-folding times."""
import json,math
import numpy as np
from single_particle import RESULTS,load_geometry,integrate,classify
from full_rf_balances import integrate_balance

def main():
    scan=json.loads((RESULTS/'negative_damping_scan.json').read_text());chosen=[]
    for geom in ['6_ABC_moderate','6_ABC_thin']:
      for lam in [.0001,.0005,.002]:
        q=[x for x in scan if x['name']==geom and abs(x['lambda_']-lam)<1e-12 and x['state']=='negative']
        if q:chosen.append(max(q,key=lambda x:abs(x['turns_total'])))
    rows=[]
    for i,c in enumerate(chosen):
        name=c['name'];lam=c['lambda_'];periods=math.ceil(5/(2*np.pi*lam));steps=96
        d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]]);amp=c['alpha']/lead
        tr,mx,ex=integrate(c['r0'],c['theta0'],0.,c['omega0'],0.,modes,coeff,amp,lam,periods,steps,True,8)
        diag=classify(tr,periods*8,ex)
        eb,Wr,Drag,Tr,Td,angle=integrate_balance(c['r0'],c['theta0'],c['omega0'],0.,modes,coeff,amp,lam,periods,steps)
        rr=np.hypot(eb[:,0],eb[:,1]);K=.5*(eb[:,2]**2+eb[:,3]**2);L=eb[:,0]*eb[:,3]-eb[:,1]*eb[:,2]
        er=K-K[0]-Wr+Drag;lr=L-L[0]-Tr-Td
        row=dict(name=name,r0=c['r0'],alpha=c['alpha'],omega0=c['omega0'],theta0=c['theta0'],lambda_=lam,
          periods=periods,nominal_damping_times=5.,state=diag['state'],turns=diag['turns_total'],
          turns_windows=diag['turns_windows'],min_radius=diag['min_radius'],max_radius=mx,
          rf_work=float(Wr[-1]),drag_loss=float(Drag[-1]),net_rf_minus_drag=float(Wr[-1]-Drag[-1]),
          initial_K=float(K[0]),final_K=float(K[-1]),energy_residual=float(er[-1]),
          rf_angular_impulse=float(Tr[-1]),drag_angular_impulse=float(Td[-1]),angular_residual=float(lr[-1]))
        rows.append(row)
        np.savez_compressed(RESULTS/f'negative_long_{i}_{name}_lambda{lam:g}.npz',rf_period=np.arange(periods+1),state=eb,
          theta=angle,K=K,RF_work=Wr,drag_loss=Drag,RF_angular_impulse=Tr,drag_angular_impulse=Td)
        print(row,flush=True)
    (RESULTS/'negative_damping_long.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
