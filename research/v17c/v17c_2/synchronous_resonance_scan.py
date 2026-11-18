"""Grid scan of harmonic synchronous rotation in the full 2-D RF field."""
import json
import numpy as np
from single_particle import RESULTS, load_geometry, integrate, classify

def main():
    geometries=[('6_ABC_moderate',.628),('9_ABC_moderate',.792),('12_ABC_moderate',.871)]
    rows=[]; periods=320;steps=64;lam=.0005
    for name,root in geometries:
        d,modes,coeff=load_geometry(name)
        lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
        sync=[(1,1/int(d['m']),[max(.18,root*.45),root*.72,root-.04]),
              (-1,-1/(int(d['N'])-int(d['m'])),[root+.025,min(.9,root+.11),min(.95,root+.20)])]
        for sign,omega_sync,rs in sync:
          for alpha in (.05,.1,.2,.4,.7,1.0):
            amp=alpha/lead
            for r in rs:
              for wf in (.85,1.,1.15):
                for theta in np.arange(8)*np.pi/4:
                  omega0=wf*omega_sync
                  tr,mx,ex=integrate(r,float(theta),0.,omega0,0.,modes,coeff,amp,lam,periods,steps,True)
                  diag=classify(tr,periods,ex)
                  row=dict(name=name,sign=sign,root=root,r0=r,alpha=alpha,amp=amp,lambda_=lam,
                           omega0=omega0,
                           theta0=float(theta),max_radius=float(mx),**diag)
                  rows.append(row)
        (RESULTS/'synchronous_resonance_checkpoint.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(name,'done; candidates',sum(x['state'] in ('positive','negative') and abs(x['turns_total'])>=5 for x in rows),flush=True)
    (RESULTS/'synchronous_resonance_scan.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for sign in (-1,1):
        good=[x for x in rows if x['sign']==sign and x['state'] in ('positive','negative') and abs(x['turns_total'])>=5]
        print('sign',sign,'valid short',len(good))
        for x in sorted(good,key=lambda x:abs(x['turns_total']),reverse=True)[:12]:
            print(x['name'],x['r0'],x['alpha'],x['omega0'],x['theta0'],x['turns_total'],x['turns_windows'],x['min_radius'],x['max_radius'])

if __name__=='__main__':main()
