"""Directed full-RF scan of negative-spin starts beyond the torque root."""
import json
import numpy as np
from single_particle import RESULTS, load_geometry, integrate, classify
from guided_search import omega_circular

def main():
    configs=[('6_ABC_moderate',.6277),('6_ABC_thin',.5674),
             ('9_ABC_moderate',.7915),('12_ABC_moderate',.8715)]
    rows=[];periods=280;steps=64;lam=.0005
    for name,root in configs:
        d,modes,coeff=load_geometry(name); lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
        n=int(d['N'])-int(d['m'])
        for r0 in [root+.012,root+.04,root+.09,root+.16]:
          if r0>.96: continue
          for alpha in [.02,.05,.1,.2,.4,.8,1.2]:
            amp=alpha/lead;wc=omega_circular(r0,modes,coeff,amp)
            tq=sum(np.sign(k)*abs(int(k))**2*(abs(int(k))-1)*abs(a)**2*r0**(2*abs(int(k))-3) for k,a in zip(modes,coeff))*amp**2
            wt=tq/r0
            starts=[-.65*wc,-wc,-1.35*wc,wt,-.8/n,-1/n,-1.2/n]
            starts=sorted(set(round(float(w),10) for w in starts if w<0))
            for omega in starts:
              for theta in np.arange(8)*np.pi/4:
                tr,mx,ex=integrate(r0,float(theta),0.,omega,0.,modes,coeff,amp,lam,periods,steps,True)
                row=dict(name=name,root=root,r0=r0,alpha=alpha,amp=amp,lambda_=lam,omega0=omega,
                         theta0=float(theta),max_radius=float(mx),**classify(tr,periods,ex))
                rows.append(row)
        (RESULTS/'outer_phase_checkpoint.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        good=[x for x in rows if x['name']==name and x['state']=='negative' and abs(x['turns_total'])>=1]
        print(name,'runs',len([x for x in rows if x['name']==name]),'bounded negative',len(good),flush=True)
    (RESULTS/'outer_phase_search.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    good=[x for x in rows if x['state']=='negative' and abs(x['turns_total'])>=1]
    for x in sorted(good,key=lambda x:abs(x['turns_total']),reverse=True)[:25]:
        print(x['name'],'r',x['r0'],'alpha',x['alpha'],'omega',x['omega0'],'theta',x['theta0'],
              'turns',round(x['turns_total'],2),'wins',np.round(x['turns_windows'],2),'rminmax',round(x['min_radius'],3),round(x['max_radius'],3))

if __name__=='__main__':main()
