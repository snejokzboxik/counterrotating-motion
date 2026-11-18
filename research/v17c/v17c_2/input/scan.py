import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.linalg import solve
import csv
from pathlib import Path
OUT=Path(__file__).resolve().parent

def model(N,m,a,B,p=24):
    # D=1, a=rod radius, B=ground shell radius, R=1-a clear aperture
    th=np.arange(N)*2*np.pi/N; ct=np.exp(1j*th); ang=th[:,None]+np.arange(p)*2*np.pi/p
    z=(ct[:,None]+a*np.exp(1j*ang)).ravel(); s=(ct[:,None]+.65*a*np.exp(1j*ang)).ravel()
    def G(z):
        return np.log(np.abs(B*B-z[:,None]*s[None,:].conj())/(B*np.abs(z[:,None]-s[None,:])))
    q=solve(G(z),np.repeat(np.exp(1j*m*th),p)); R=1-a; n=N-m
    def c(k):
        f=(s**(-abs(k))-(s.conj()/B**2)**abs(k))*R**abs(k)/(2*abs(k)); return np.dot(q,f if k>0 else f.conj())
    # Independent boundary validation at interleaved angles
    zz=(ct[:,None]+a*np.exp(1j*(ang+np.pi/p))).ravel()
    err=np.max(np.abs(G(zz)@q-np.repeat(np.exp(1j*m*th),p)))
    cp,cn=c(m),c(-n); ratio=abs(cn/cp); F=n/m*ratio; cross=(1/F)**(1/(n-m))
    return dict(N=N,m=m,n=n,a_D=a,B_D=B,gap_D=B-1-a,p=p,Aplus=abs(cp),Aminus=abs(cn),ratio=ratio,F_R=F,F_08=F*.8**(n-m),cross_R=cross,boundary_error=err,phase_diff_rad=np.angle(cn/cp),next_positive=abs(c(m+N)),next_negative=abs(c(-n-N)))

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__=='__main__':
    rows=[]
    for N,m in [(6,2),(9,3),(12,4),(15,5),(18,6),(9,4),(10,4),(11,4)]:
        for a in [.01,.02,.03,.05,.08,.12,.18,.22,.28,.35,.42]:
            if a>=.9*np.sin(np.pi/N):continue
            for gap in [.02,.05,.1,.2,.5,1.,3.]:
                rows.append(model(N,m,a,1+a+gap))
    write('scan.csv',rows)
    valid=[r for r in rows if r['boundary_error']<1e-5]
    chosen=[]
    for N,m in [(6,2),(9,3),(12,4),(15,5),(18,6),(9,4),(10,4),(11,4)]:
        sel=[r for r in valid if r['N']==N and r['m']==m]
        best=max(sel,key=lambda r:r['F_08']); practical=max([r for r in sel if r['a_D']>=.05 and r['gap_D']>=.05],key=lambda r:r['F_08'])
        print(N,m,'best',best,'less extreme',practical)
        for r in (best,practical):
            for p in [16,24,40,64]:chosen.append(model(N,m,r['a_D'],r['B_D'],p))
    write('convergence.csv',chosen)
    print('total',len(rows),'valid',len(valid))
