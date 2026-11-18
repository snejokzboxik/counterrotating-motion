import numpy as np
from model import batch
from scipy.optimize import minimize
from pathlib import Path
root=Path(__file__).parent;k=.1044
for n in [7,12]:
 ys=[];p=[];ms=[];qs=[];gs=[];groups=[];rng=np.random.default_rng(27181+n)
 for c in [3e-6,5e-5,2e-4]:
  for mr,qr,gr in [(0.2,1,1),(10,1,1),(1,.2,1),(1,5,1),(10,3,.1),(.2,.5,5),(1,1,.1),(1,1,10)]:
   group=np.arange(n)<max(2,n//3);m=np.where(group,mr,1.);q=np.where(group,qr,1.)
   for mode,eps in [(0,0),(1,0),(0,.32945),(0,1.2)]:
    a=(4 if mode==1 else 8)*k*k*q*q/m
    def fun(x):
     r=x.reshape(n,2);r2=(r*r).sum(1);d=r[:,None,:]-r[None,:,:];s=(d*d).sum(2)+np.eye(n);iv=s**(-.5);np.fill_diagonal(iv,0);qq=q[:,None]*q[None,:];grad=6*a[:,None]*r2[:,None]**2*r-c*(d*(qq*iv**3)[:,:,None]).sum(1)
     return (a*r2**3).sum()+c*(qq*iv).sum()/2,grad.ravel()
    x=minimize(fun,rng.normal(0,.2,(n,2)).ravel(),jac=True,method='L-BFGS-B',options={'gtol':1e-10,'ftol':1e-13,'maxiter':1500}).x.reshape(n,2);z=x[:,0]+1j*x[:,1];z*=np.exp(1j*rng.uniform(0,2*np.pi))
    for g0,spin in [(.003,0),(.001,.025)]:
     v=1j*spin*np.where(group,-1,1)*z
     if mode==0:v+=4*k*q/m*((z**3).imag+1j*(z**3).real)-8*k*eps*q/m*((z**7).imag+1j*(z**7).real)
     ys.append(np.column_stack((z.real,z.imag,v.real,v.imag)));p.append([c,g0,mode,.3,eps,0,spin,mr,qr,gr]);ms.append(m);qs.append(q);gs.append(np.where(group,g0*gr,g0));groups.append(group)
 ys,p,ms,qs,gs,groups=map(np.array,(ys,p,ms,qs,gs,groups));steps=48;periods=600
 for start in range(0,len(p),16):
  sl=slice(start,start+16);pp=p[sl];st,o,mi,rs,fin=batch(ys[sl],pp[:,0],gs[sl],pp[:,2].astype(int),pp[:,3],pp[:,4],pp[:,5],k,steps,periods,ms[sl],qs[sl]);np.savez(root/f'scout_n{n}_{start}.npz',ys=ys[sl],p=pp,m=ms[sl],q=qs[sl],g=gs[sl],groups=groups[sl],st=st,o=o,mi=mi,rs=rs,fin=fin,steps=steps,periods=periods)
  turns=o[:,:,:,0]/(2*np.pi);good=mi.min(axis=1)>.02;pos=(turns.min(axis=1)>.05)&good;neg=(turns.max(axis=1)<-.05)&good;ix=np.where((st==1)&pos.any(axis=1)&neg.any(axis=1))[0];print(n,start,int(sum(st)),ix.tolist(),flush=True)
