import numpy as np
from pathlib import Path
from model import batch
root=Path(__file__).parent
for n in [7,12]:
 records=[]
 for f in sorted(root.glob(f'scout_n{n}_*.npz')):
  d=np.load(f);t=d['o'][:,:,:,0]/(2*np.pi);good=d['mi'].min(1)>.02
  for i in range(len(t)):
   if d['st'][i] and np.any(good[i]&(t[i,-1]>.005)) and np.any(good[i]&(t[i,-1]<-.005)):
    records.append((f.name,i,*[d[key][i] for key in ['ys','p','m','q','g','groups']]))
 if not records:continue
 files,indices,ys,p,m,q,g,groups=map(np.array,zip(*records));
 for steps,periods in [(96,600),(96,4000),(192,4000)]:
  st,o,mi,rs,fin=batch(ys,p[:,0],g,p[:,2].astype(int),p[:,3],p[:,4],p[:,5],.1044,steps,periods,m,q)
  np.savez(root/f'confirm_n{n}_{steps}_{periods}.npz',files=files,indices=indices,ys=ys,p=p,m=m,q=q,g=g,groups=groups,st=st,o=o,mi=mi,rs=rs,fin=fin,steps=steps,periods=periods)
  turns=o[:,:,:,0]/(2*np.pi);good=mi.min(axis=1)>.02;pos=(turns.min(axis=1)>.05)&good;neg=(turns.max(axis=1)<-.05)&good;ix=np.where((st==1)&pos.any(axis=1)&neg.any(axis=1))[0];print(n,steps,periods,len(st),int(sum(st)),ix.tolist(),flush=True)
  for i in ix:print(p[i],turns[i],flush=True)
