"""Differential-evolution search for long-lived outer negative-spin orbits."""
import json
import numpy as np
from scipy.optimize import differential_evolution
from single_particle import RESULTS, load_geometry, integrate

def optimize(name,root,seed):
    d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
    periods=1200;steps=64;sample=8;lam=.0005; best={'score':1e9}
    bounds=[(root+.005,min(.96,root+.24)),(-1.7,.15),(-3.5,-.0001),(0,2*np.pi),(0,2*np.pi)]
    def obj(v):
        r,loga,omega,theta,phase=v;alpha=10**loga;amp=alpha/lead
        tr,mx,escaped=integrate(r,theta,0.,omega,phase,modes,coeff,amp,lam,periods,steps,True,sample)
        surv=(len(tr)-1)/(periods*sample)
        rad=np.hypot(tr[:,0],tr[:,1]);angle=np.unwrap(np.arctan2(tr[:,1],tr[:,0]))
        if escaped:
            score=50+50*(1-surv)+10*max(0.,mx-1.)
        else:
            a=int(.4*len(rad));b=int(.7*len(rad));c=len(rad)-1
            r1=rad[a:b];r2=rad[b:c]
            t1=(angle[b]-angle[a])/(2*np.pi);t2=(angle[c]-angle[b])/(2*np.pi)
            late=rad[a:c]
            score=0.5*np.std(late)+1.8*abs(float(np.mean(r1)-np.mean(r2)))
            score+=12*max(0.,.05-float(late.min()))+8*max(0.,float(late.max())-.93)
            score+=4*max(0.,t1)+4*max(0.,t2)
            score+=.08*max(0.,15-min(abs(t1),abs(t2)))
            if t1*t2<=0:score+=2
        if score<best['score']:
            best.update(score=float(score),r=float(r),alpha=float(alpha),omega=float(omega),theta=float(theta),phase=float(phase),
                        survival=float(surv),escaped=bool(escaped),max_radius=float(mx),turns=float((angle[-1]-angle[0])/(2*np.pi)),
                        min_radius=float(rad[int(.4*len(rad)):].min()),windows=[float((angle[q]-angle[p])/(2*np.pi)) for p,q in ((int(.4*len(angle)),int(.6*len(angle))), (int(.6*len(angle)),int(.8*len(angle))), (int(.8*len(angle))-1,len(angle)-1))])
            (RESULTS/f'optimize_{name}_best.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
        return float(score)
    rng=np.random.default_rng(seed);init=np.array([[lo+(hi-lo)*q for (lo,hi),q in zip(bounds,row)]
                        for row in rng.random((40,len(bounds)))])
    if name=='6_ABC_moderate': init[0]=[.6397,np.log10(.05),-.098075855,np.pi/2,0.]
    else: init[0]=[.5794,np.log10(.05),-.0981346613,np.pi/2,0.]
    res=differential_evolution(obj,bounds,init=init,seed=seed,popsize=8,maxiter=14,tol=.02,
                               polish=False,updating='immediate',workers=1,disp=False)
    result=dict(name=name,success=bool(res.success),message=str(res.message),fun=float(res.fun),x=res.x.tolist(),best=best)
    (RESULTS/f'optimize_{name}_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(name,'result',res.fun,res.x,'best',best,flush=True)
    return result

def main():
    optimize('6_ABC_moderate',.6277,1729)
    optimize('6_ABC_thin',.5674,1730)

if __name__=='__main__':main()
