import numpy as np
from hydro_model import batch
from pathlib import Path
root=Path(__file__).parent
for n in [7,12,19]:
 z=np.load(root/f'relaxed{n}.npz');p=z['p'];ix=np.where((p[:,0]>=3e-6)&(p[:,1]>=.001)&(p[:,2]==0))[0];p=p[ix];ys=z['ys'][ix];steps=96;periods=2000
 st,o,mi,rs,fin=batch(ys,p[:,0],p[:,1],p[:,2].astype(int),p[:,3],p[:,4],p[:,5],.1044,steps,periods)
 np.savez(root/f'hydro{n}.npz',ys=ys,p=p,st=st,o=o,mi=mi,rs=rs,fin=fin,periods=periods,steps=steps,source_ix=ix)
 turns=o[:,:,:,0]/(2*np.pi);good=mi.min(axis=1)>.02;pos=(turns.min(axis=1)>.05)&good;neg=(turns.max(axis=1)<-.05)&good;print(n,len(ix),sum(st),np.where((st==1)&pos.any(axis=1)&neg.any(axis=1))[0], 'minsep survivors',rs[st==1,1].min() if any(st) else None,flush=True)
