"""Long-horizon and perturbation checks for opposite G1 single-particle orbits."""
import json
import numpy as np
from single_particle import RESULTS, load_geometry, integrate, classify

def window_radii(tr,periods):
    r=np.hypot(tr[:,0],tr[:,1]);ans=[]
    for lo,hi in ((.4,.6),(.6,.8),(.8,1.)):
        a=min(len(r)-1,int(periods*lo));b=min(len(r),int(periods*hi))
        ans.append([float(r[a:b].min()),float(r[a:b].max()),float(r[a:b].mean())])
    return ans

def main():
    specs=[dict(label='inner_positive',r=.45216,alpha=.1,omega=.425,theta=0.),
           dict(label='outer_negative',r=.6397,alpha=.05,omega=-.098075855,theta=np.pi/2)]
    d,modes,coeff=load_geometry('6_ABC_moderate'); lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
    rows=[];periods=8000;steps=96
    for s in specs:
      amp=s['alpha']/lead;lam=.0005
      variants=[('nominal',s['r'],s['omega'],amp,lam,s['theta'],0.),
        ('radius_plus_1pct',s['r']*1.01,s['omega'],amp,lam,s['theta'],0.),
        ('radius_minus_1pct',s['r']*.99,s['omega'],amp,lam,s['theta'],0.),
        ('speed_plus_1pct',s['r'],s['omega']*1.01,amp,lam,s['theta'],0.),
        ('amplitude_plus_5pct',s['r'],s['omega'],amp*1.05,lam,s['theta'],0.),
        ('damping_plus_10pct',s['r'],s['omega'],amp,lam*1.1,s['theta'],0.),
        ('angle_plus_0p01',s['r'],s['omega'],amp,lam,s['theta']+.01,0.),
        ('rf_phase_plus_0p01',s['r'],s['omega'],amp,lam,s['theta'],.01)]
      for tag,r,w,a,l,theta,phase in variants:
        tr,mx,ex=integrate(r,theta,0.,w,phase,modes,coeff,a,l,periods,steps,True)
        diag=classify(tr,periods,ex)
        row=dict(case=s['label'],variant=tag,r0=r,alpha=a*lead,omega0=w,lambda_=l,
                 rf_phase=phase,theta0=theta,max_radius=float(mx),late_radius_windows=window_radii(tr,periods),
                 **diag)
        rows.append(row)
        if tag=='nominal':
            ang=np.unwrap(np.arctan2(tr[:,1],tr[:,0]))
            np.savez_compressed(RESULTS/f"G1_{s['label']}_8000RF.npz",rf_periods=np.arange(len(tr)),
                                state=tr,theta=ang,parameters=json.dumps(row))
        print(s['label'],tag,diag['state'],round(diag['turns_total'],3),diag['turns_windows'],
              'r',round(diag['min_radius'],4),round(float(mx),4),flush=True)
    (RESULTS/'long_candidate_validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
