"""Outer negative-spin scan over RF strength, friction, radius and start speed."""
import json
import numpy as np
from single_particle import RESULTS,load_geometry,integrate,classify
from guided_search import omega_circular

def main():
    configs=[('6_ABC_moderate',.6277),('6_ABC_thin',.5674)]
    rows=[];periods=320;steps=64
    for name,root in configs:
        d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]]);n=int(d['N'])-int(d['m'])
        for r in [root+.012,root+.04,root+.09,root+.16]:
          if r>.96:continue
          for alpha in [.005,.01,.02,.05,.1,.2,.4,.8]:
            amp=alpha/lead;wc=omega_circular(r,modes,coeff,amp)
            tq=sum(np.sign(k)*abs(int(k))**2*(abs(int(k))-1)*abs(a)**2*r**(2*abs(int(k))-3) for k,a in zip(modes,coeff))*amp**2
            wtor=tq/r
            for lam in [.0001,.0005,.002,.00477464829]:
              starts=set(round(float(w),10) for w in [-.65*wc,-wc,-1.35*wc,wtor,-1/n] if w<0)
              for omega in starts:
               for theta in np.arange(8)*np.pi/4:
                tr,mx,ex=integrate(r,float(theta),0.,omega,0.,modes,coeff,amp,lam,periods,steps,True)
                row=dict(name=name,root=root,r0=r,alpha=alpha,omega0=omega,lambda_=lam,theta0=float(theta),
                         max_radius=float(mx),**classify(tr,periods,ex))
                rows.append(row)
        (RESULTS/'negative_damping_checkpoint.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(name,'runs',sum(x['name']==name for x in rows),'bounded negative',sum(x['name']==name and x['state']=='negative' for x in rows),flush=True)
    (RESULTS/'negative_damping_scan.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for geom in configs:
        name=geom[0]
        for lam in [.0001,.0005,.002,.00477464829]:
            q=[x for x in rows if x['name']==name and x['lambda_']==lam and x['state']=='negative']
            print('best',name,'lambda',lam,'count',len(q),max([abs(x['turns_total']) for x in q],default=0),flush=True)

if __name__=='__main__':main()
