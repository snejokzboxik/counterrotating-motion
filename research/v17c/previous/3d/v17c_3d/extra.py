import os
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
from model import batch
from run import initial
n=12;ys=[];ps=[];ms=[];qs=[];gs=[];meta=[]
for az in [1,2]:
 for c in [2e-5,1e-4]:
  for mr,qr,gr in [(1,.3,1),(1,3,1),(.4,.5,5),(4,2,.2)]:
   for e in [0,.32945]:
    p=np.array([.1044,c,e,az,.045,.55,0,.3]);m=np.r_[np.full(n//2,mr),np.ones(n//2)];q=np.r_[np.full(n//2,qr),np.ones(n//2)];g=np.r_[np.full(n//2,.003*gr),np.full(n//2,.003)];seed=987+len(ps)//2;ys.append(initial(n,p,m,q,seed,.015));ps.append(p);ms.append(m);qs.append(q);gs.append(g);meta.append([mr,qr,gr,seed])
ys,ps,ms,qs,gs,meta=map(np.array,(ys,ps,ms,qs,gs,meta));np.savez('extra_inputs.npz',ys=ys,ps=ps,ms=ms,qs=qs,gs=gs,meta=meta)
for i in range(0,len(ys),8):
 sl=slice(i,i+8);st,s,mi,ct,rs,fin=batch(ys[sl],ps[sl],ms[sl],qs[sl],gs[sl],64,800);np.savez(f'extra_{i}.npz',st=st,s=s,mi=mi,ct=ct,rs=rs,fin=fin,steps=64,periods=800);turns=s[:,:,:,0]/(2*np.pi);pos=(turns.min(1)>.1)&(mi.min(1)>.015);neg=(turns.max(1)<-.1)&(mi.min(1)>.015);print(i,sum(st),(np.where((st==1)&pos.any(1)&neg.any(1))[0]+i).tolist(),flush=True)
