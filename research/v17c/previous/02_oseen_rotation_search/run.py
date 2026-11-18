import numpy as np
from model import batch
from pathlib import Path
root=Path(__file__).parent
k=.1044
for n in [7,12,19]:
 ys=[]; params=[];ni={7:2,12:3,19:6}[n]
 for rad in [.32,.52,.68]:
  for g in [.001,.005,.025]:
   rng=np.random.default_rng(17000+n+int(rad*100))
   a=np.r_[np.arange(ni)*2*np.pi/ni+.5,np.arange(n-ni)*2*np.pi/(n-ni)]+rng.uniform(0,6.28)+rng.uniform(-.05,.05,n)
   r=rad*np.r_[np.full(ni,.39),np.ones(n-ni)]*(1+rng.uniform(-.06,.06,n)); w=np.r_[np.full(ni,-.04),np.full(n-ni,.04)]; y=np.column_stack((r*np.cos(a),r*np.sin(a),-w*r*np.sin(a),w*r*np.cos(a)))
   c=48*k*k*rad**7/(n-1)
   for mode,eps in [(0,0),(1,0),(2,0),(0,.32945)]:
    for re in [0.,.3]:
     ys.append(y.copy());params.append([c,g,mode,re,eps,0.])
 ys=np.array(ys);p=np.array(params);steps=48;periods=400
 st,o,mi,rs,fin=batch(ys,p[:,0],p[:,1],p[:,2].astype(int),p[:,3],p[:,4],p[:,5],k,steps,periods)
 np.savez(root/f'n{n}.npz',ys=ys,p=p,st=st,o=o,mi=mi,rs=rs,fin=fin,periods=periods,steps=steps)
 turns=o[:,:,:,0]/(2*np.pi); good=mi.min(axis=1)>.02; pos=(turns.min(axis=1)>.05)&good;neg=(turns.max(axis=1)<-.05)&good
 ix=np.where((st==1)&pos.any(axis=1)&neg.any(axis=1))[0];print(n,'retained',sum(st),'candidates',ix.tolist(), 'maxre',np.max(rs[:,2]),flush=True)
