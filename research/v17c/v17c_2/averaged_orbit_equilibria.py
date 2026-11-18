"""Find and linearly test circular fixed points of the averaged 2-D dynamics."""
import json
import numpy as np
from single_particle import load_geometry, RESULTS

def raw(r, modes, coeff):
    # U=amp^2/4 <|E_RF|^2>, and the weak-micromotion guiding-center torque.
    up=0.; tq=0.
    for k,a in zip(modes,coeff):
        n=abs(int(k)); w=n*n*abs(a)**2*r**(2*n-3)*(n-1)
        up += .5*w
        tq += np.sign(k)*w
    return up,tq

def eigs(r,amp,lam,spin_sign):
    # state (r, vr, omega), including geometric Coriolis terms.
    h=max(1e-6,r*1e-5)
    def f(y):
        rr,vv,om=y
        up=amp*amp*(up1(rr-h)+0) if False else 0.
        # use centered numerical derivatives of the analytic radial pseudopotential
        # U=1/4 sum n^2 |A|^2 r^(2n-2).
        upr=0.; tq=0.
        for k,a in zip(modes0,coeff0):
            n=abs(int(k)); q=n*n*abs(a)**2
            upr += .25*q*(2*n-2)*rr**(2*n-3)
            tq += np.sign(k)*q*(n-1)*rr**(2*n-3)
        upr*=amp*amp; tq*=amp*amp
        ath=lam/(1+lam*lam)*tq
        return np.array([vv,rr*om*om-upr-lam*vv,(ath-lam*rr*om-2*vv*om)/rr])
    upr0=0.
    for k,a in zip(modes0,coeff0):
        n=abs(int(k)); upr0 += .25*n*n*abs(a)**2*(2*n-2)*r**(2*n-3)
    J=np.empty((3,3)); y=np.array([r,0.,spin_sign*np.sqrt(max(amp*amp*upr0/r,0))])
    for j in range(3):
        d=np.zeros(3);d[j]=max(1e-7,abs(y[j])*1e-6)
        J[:,j]=(f(y+d)-f(y-d))/(2*d[j])
    return np.linalg.eigvals(J),f(y)

def main():
    global modes0,coeff0
    rows=[]
    for name in ['6_ABC_moderate','6_ABC_thin','9_ABC_moderate','9_ABC_thin','12_ABC_moderate','12_ABC_thin','12_ABC_extreme']:
        try: d,modes,coeff=load_geometry(name)
        except StopIteration: continue
        modes0,coeff0=modes,coeff
        rts=np.linspace(.06,.97,1000)
        for lam in [1e-4,5e-4,.002,.00477464829]:
            for r in rts:
                u,t=raw(r,modes,coeff)
                if abs(t)<1e-15: continue
                amp2=(1+lam*lam)**2*r*u/t**2
                amp=np.sqrt(amp2)
                lead=abs(coeff[np.where(modes==d['m'])[0][0]])
                base=.1044/lead
                if amp/base > 10 or amp/base < 1e-3: continue
                # equilibrium omega follows torque, while radial balance imposes magnitude.
                om=np.sign(t)*np.sqrt(amp2*u/r)
                # Jacobian at (r,0,omega)
                ev,_=eigs(r,amp,lam,np.sign(t))
                rows.append(dict(name=name,r=r,lambda_=lam,amplitude=amp,relative_to_base=amp/base,
                                 torque_sign=int(np.sign(t)),omega=om,eigenvalues=[[float(z.real),float(z.imag)] for z in ev],
                                 stable=bool(np.max(ev.real)<0)))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'averaged_orbit_equilibria.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for name in sorted(set(x['name'] for x in rows)):
        aa=[x for x in rows if x['name']==name]
        print(name,'points',len(aa),'stable',sum(x['stable'] for x in aa),
              'signs stable',sorted(set(x['torque_sign'] for x in aa if x['stable'])),
              'ranges',[(sg,min(x['r'] for x in aa if x['stable'] and x['torque_sign']==sg),max(x['r'] for x in aa if x['stable'] and x['torque_sign']==sg)) for sg in [-1,1] if any(x['stable'] and x['torque_sign']==sg for x in aa)])

if __name__=='__main__': main()
