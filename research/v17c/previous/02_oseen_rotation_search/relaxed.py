import numpy as np
from model import batch
from scipy.optimize import minimize
from pathlib import Path
root=Path(__file__).parent;k=.1044
for n in [7,12,19]:
 ys=[];p=[];rng=np.random.default_rng(912+n)
 for c in [1e-7,3e-6,5e-5]:
  for mode,eps in [(0,0),(1,0),(0,.32945)]:
   a=8*k*k*(.5 if mode==1 else 1.)
   def fun(x):
    r=x.reshape(n,2);r2=(r*r).sum(1);d=r[:,None,:]-r[None,:,:];s=(d*d).sum(2)+np.eye(n);inv=s**(-.5);np.fill_diagonal(inv,0);grad=6*a*r2[:,None]**2*r-c*(d*inv[:,:,None]**3).sum(1)
    return a*(r2**3).sum()+c*inv.sum()/2,grad.ravel()
   x=rng.normal(0,.12,(n,2));x=minimize(fun,x.ravel(),jac=True,method='L-BFGS-B', options={'gtol':1e-12,'ftol':1e-15,'maxiter':2000}).x.reshape(n,2)
   r=np.linalg.norm(x,axis=1); ix=np.argsort(r); inner=ix[:max(2,n//3)];z=x[:,0]+1j*x[:,1]
   for g in [.0002,.001,.005]:
    for spin in [.0,.025]:
     w=np.full(n,spin);w[inner]=-spin;v=1j*w*z
     if mode==0:v+=4*k*(z**3).imag+1j*4*k*(z**3).real
     if eps:v+=-8*k*eps*(z**7).imag-1j*8*k*eps*(z**7).real
     y=np.column_stack((x,v.real,v.imag));ys.append(y);p.append([c,g,mode,.3,eps,0,spin])
 ys=np.array(ys);p=np.array(p);steps=48;periods=1000
 st,o,mi,rs,fin=batch(ys,p[:,0],p[:,1],p[:,2].astype(int),p[:,3],p[:,4],p[:,5],k,steps,periods)
 np.savez(root/f'relaxed{n}.npz',ys=ys,p=p,st=st,o=o,mi=mi,rs=rs,fin=fin,periods=periods,steps=steps)
 turns=o[:,:,:,0]/(2*np.pi);good=mi.min(axis=1)>.02;pos=(turns.min(axis=1)>.05)&good;neg=(turns.max(axis=1)<-.05)&good
 ix=np.where((st==1)&pos.any(axis=1)&neg.any(axis=1))[0];print(n,'retained',sum(st),'candidates',ix.tolist(),'maxre',np.max(rs[:,2]),flush=True)
 for i in ix:print(p[i],turns[i],flush=True)
