from pathlib import Path
import numpy as np
from model import batch
root=Path(__file__).parent
base=np.load(root/'confirm_n12_192_4000.npz');idx=0;y=base['fin'][idx].copy();r=np.linalg.norm(y[:,:2],axis=1);inner=np.argsort(r)[:2];group=np.zeros(len(y),bool);group[inner]=True
ys=[];ps=[];ms=[];qs=[];gs=[]
for eps in [0,.32945,1.2]:
 for ratio in [.1,1,10]:
  for re in [0,.3]:
   for spin in [0,-.015]:
    yy=y.copy(); yy[inner,2]+= -spin*yy[inner,1];yy[inner,3]+=spin*yy[inner,0]
    ys.append(yy);ps.append([5e-5,eps,ratio,re,spin]);ms.append(np.ones(12));qs.append(np.ones(12));gs.append(np.where(group,.003*ratio,.003))
ys,p,m,q,g=map(np.array,(ys,ps,ms,qs,gs));periods=2000;steps=96
st,o,mi,rs,fin=batch(ys,p[:,0],g,np.zeros(len(p),int),p[:,3],p[:,1],np.zeros(len(p)),.1044,steps,periods,m,q)
np.savez(root/'targeted.npz',ys=ys,p=p,m=m,q=q,g=g,group=group,st=st,o=o,mi=mi,rs=rs,fin=fin,steps=steps,periods=periods)
t=o[:,:,:,0]/(2*np.pi);g0=mi.min(1)>.02;pos=(t.min(1)>.05)&g0;neg=(t.max(1)<-.05)&g0;ix=np.where((st==1)&pos.any(1)&neg.any(1))[0];print(len(st),sum(st),ix.tolist(),flush=True)
for i in range(len(p)):
 if st[i]:print(i,p[i],np.round(t[i,-1]*1e5/(periods*.2),2),flush=True)
