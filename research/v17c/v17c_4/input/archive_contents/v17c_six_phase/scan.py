"""12 circular electrodes, grounded circular shield. MFS 2D complex voltage phasors.
V=Re[sum A_k r^|k| exp(ik theta-i tau)], r normalized to R=D-a.
Requires numpy, scipy, threadpoolctl. Run python scan.py."""
import numpy as np
import itertools,csv
from scipy.linalg import solve
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from pathlib import Path
OUT=Path(__file__).resolve().parent
def transfer(N,a,B,p=32,K=48):
    th=np.arange(N)*2*np.pi/N; ct=np.exp(1j*th); ang=th[:,None]+np.arange(p)*2*np.pi/p
    z=(ct[:,None]+a*np.exp(1j*ang)).ravel(); s=(ct[:,None]+.65*a*np.exp(1j*ang)).ravel()
    def G(z):
        return np.log(np.abs(B*B-z[:,None]*s[None,:].conj())/(B*np.abs(z[:,None]-s[None,:])))
    with threadpool_limits(limits=1):
        V=np.repeat(np.eye(N),p,axis=0)
        q=solve(G(z),V)
        zz=(ct[:,None]+a*np.exp(1j*(ang+np.pi/p))).ravel()
        err=G(zz)@q-V
        k=np.r_[np.arange(-K,0),np.arange(1,K+1)]
        ff=np.array([(s**(-abs(i))-(s.conj()/B**2)**abs(i))*(1-a)**abs(i)/(2*abs(i)) for i in k])
        ff[k<0]=ff[k<0].conj()
        return k,ff@q,err
def roots(k,h):
    n=abs(k); w=np.sign(k)*n*n*(n-1)*abs(h)**2
    def f(r):return np.dot(w,r**(2*n-2))
    rs=np.linspace(.05,.98,250)
    return [brentq(f,a,b) for a,b in zip(rs[:-1],rs[1:]) if f(a)*f(b)<0]
def diag(k,h,r):
    n=abs(k); term=n*n*(n-1)*abs(h)**2*r**(2*n-2)
    ch=np.sum(np.sign(k)*term)/np.sum(term)
    th=np.arange(720)*2*np.pi/720; a=n*h*r**(n-1)
    plus=np.exp(1j*np.outer(th,k[k>0]))@a[k>0]
    minus=np.exp(1j*np.outer(th,k[k<0]))@a[k<0]
    w=abs(plus)**2+abs(minus)**2
    return ch,(w.max()-w.min())/w.mean(),w.min()/w.mean(),w.mean()
def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
    seqs=[(0,)+p for p in itertools.permutations(range(1,6))]
    names=[''.join("ABCDEF"[x] for x in p)*2 for p in seqs]
    V=np.exp(2j*np.pi*np.tile(np.array(seqs),(1,2)).T/6)
    k,T,E=transfer(12,.05,1.1)
    H=T@V; rr=np.linspace(.05,.95,181); n=abs(k)
    w=np.sign(k)*n*n*(n-1)
    M=(rr[:,None]**(2*n-2)*w)@abs(H)**2
    norm=(rr[:,None]**(2*n-2)*abs(w))@abs(H)**2
    ia=np.argmin(abs(rr-.3));ib=np.argmin(abs(rr-.75))
    rows=[];spec=[]
    for j,name in enumerate(names):
        rows.append(dict(sequence=name,chi_03=M[ia,j]/norm[ia,j],chi_075=M[ib,j]/norm[ib,j],roots=";".join(map(str,roots(k,H[:,j]))) if np.max(abs(M[:,j]/norm[:,j]))>1e-8 else "numerically_zero",boundary_error=max(abs(E@V[:,j]))))
        for n0,h0 in zip(k,H[:,j]): spec.append(dict(sequence=name,k=n0,re=h0.real,im=h0.imag,abs=abs(h0)))
    write("permutations.csv",rows);write("permutation_spectra.csv",spec)
    patterns={"standard6":[0,1,2,3,4,5],"mixed6":[0,3,2,5,4,1],"control3":[0,2,4,0,2,4]}
    geoms=[(.05,1.1),(.01,1.03),(.12,1.25),(.18,1.5),(.08,1.2),(.12,1.5),(.2,2),(.05,2)]
    rows=[];conv=[];spec=[];rad=[]
    for a,B in geoms:
        for p,K in [(32,48),(48,48),(64,72)] if (a,B)==(.05,1.1) else [(32,48)]:
            k,T,E=transfer(12,a,B,p,K)
            for label,seq in patterns.items():
                v=np.exp(2j*np.pi*np.array(seq*2)/6);h=T@v
                d=dict(geometry_a_D=a,geometry_B_D=B,p=p,K=K,pattern=label,roots=";".join(map(str,roots(k,h))),boundary_error=max(abs(E@v)))
                for n0 in [2,-4,4,-8,8,-10,14,-16]:d["A_"+str(n0)]=abs(h[k==n0][0])
                conv.append(d)
                if p==32:
                    rows.append(d)
                    for n0,h0 in zip(k,h):spec.append(dict(a_D=a,B_D=B,pattern=label,k=n0,re=h0.real,im=h0.imag,abs=abs(h0)))
                    for r in [.2,.3,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]:
                        chi,corr,min_mean,mean=diag(k,h,r)
                        rad.append(dict(a_D=a,B_D=B,pattern=label,r_R=r,chirality=chi,corrugation=corr,min_mean=min_mean,mean_half_gradient_squared=mean))
    write("geometries.csv",rows);write("convergence.csv",conv);write("geometry_spectra.csv",spec);write("radial_diagnostics.csv",rad)
if __name__=="__main__":run()
