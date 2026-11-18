"""Weak-micromotion torque theory and full-spectrum radial sign maps."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from single_particle import load_geometry,INPUT,RESULTS

HERE=Path(__file__).resolve().parent

def microscopic_average(k,A,r,lam,amp=1.,nangle=128):
    """Average J*delta force over center angle from complex RF phasors."""
    n=abs(int(k)); vals=[]
    for theta in np.arange(nangle)*2*np.pi/nangle:
        u=r*np.exp(1j*theta)
        if k>0:
            w=n*A*u**(n-1); h=n*(n-1)*A*u**(n-2)
            f=np.array([-w,-1j*w])*amp
            J=np.array([[-h,-1j*h],[-1j*h,h]],complex)*amp
        else:
            w=n*A*u.conjugate()**(n-1);h=n*(n-1)*A*u.conjugate()**(n-2)
            f=np.array([-w,1j*w])*amp
            J=np.array([[-h,1j*h],[1j*h,h]],complex)*amp
        delta=f/(-1-1j*lam)
        mean=.5*np.real(J@delta.conjugate())
        vals.append(-np.sin(theta)*mean[0]+np.cos(theta)*mean[1])
    return float(np.mean(vals))

def predicted(k,A,r,lam,amp=1.):
    n=abs(int(k))
    return np.sign(k)*lam/(1+lam*lam)*n*n*(n-1)*(amp*abs(A))**2*r**(2*n-3)

def main():
    RESULTS.mkdir(exist_ok=True)
    specs=['6_ABC_moderate','6_ABC_thin','9_ABC_moderate','9_ABC_thin','12_ABC_moderate']
    rows=[];checks=[]
    for name in specs:
        d,modes,coeff=load_geometry(name); amp=.1044/abs(coeff[np.where(modes==d['m'])[0][0]])
        lam=.00477464829
        lead=int(d['m']); opp=-(d['N']-d['m'])
        # Compare the full phasor perturbation calculation with the derived law.
        for k in (lead,opp):
            A=coeff[np.where(modes==k)[0][0]];r=.43
            exact=microscopic_average(k,A,r,lam,amp);theory=predicted(k,A,r,lam,amp)
            checks.append(dict(name=name,k=k,phasor_average=exact,formula=theory,
                               relative_error=abs(exact-theory)/(abs(theory)+1e-30)))
        rs=np.linspace(.08,.98,451)
        torque=np.zeros_like(rs)
        for k,A in zip(modes,coeff):
            n=abs(int(k));torque+=np.sign(k)*n*n*(n-1)*abs(amp*A)**2*rs**(2*n-2)
        # drag factor is common, so it changes magnitude but not root.
        rootvals=[]
        for x0,x1,y0,y1 in zip(rs[:-1],rs[1:],torque[:-1],torque[1:]):
            if y0*y1<0:rootvals.append(brentq(lambda x:sum(np.sign(k)*abs(k)**2*(abs(k)-1)*abs(amp*A)**2*x**(2*abs(k)-2) for k,A in zip(modes,coeff)),x0,x1))
        pair=(np.sqrt((d['m']-1)/(d['N']-d['m']-1))/( (d['N']-d['m'])/d['m']*abs(coeff[np.where(modes==opp)[0][0]]/coeff[np.where(modes==lead)[0][0]]) ))**(1/(d['N']-2*d['m'])) if d['N']-2*d['m'] else None
        for x,T in zip(rs,torque): rows.append(dict(name=name,r_over_R=x,torque_dimensionless=float(T),
              torque_accel=lam/(1+lam*lam)*float(T)/x if x else 0.))
        print(name,'full spectrum zero',rootvals,'pair estimate',pair,flush=True)
    with (RESULTS/'averaged_torque_map.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (RESULTS/'averaging_micro_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')

if __name__=='__main__':main()
