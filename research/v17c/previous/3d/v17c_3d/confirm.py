import os
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
from model import batch,one
from pathlib import Path
D=np.load('inputs.npz');idx=np.array([35,39,43,47,55,59,63,67,71,83,87]);ys,ps,ms,qs,gs=[D[k][idx] for k in ['ys','ps','ms','qs','gs']]
for steps in [96,192]:
 st,s,mi,ct,rs,fin=batch(ys,ps,ms,qs,gs,steps,4000);np.savez(f'confirm_{steps}.npz',idx=idx,st=st,s=s,mi=mi,ct=ct,rs=rs,fin=fin,steps=steps,periods=4000)
 turns=s[:,:,:,0]/(2*np.pi); good=mi.min(1)>.015;pos=(turns.min(1)>1)&good;neg=(turns.max(1)<-1)&good;print(steps,'retained',sum(st),'counter',idx[(st==1)&pos.any(1)&neg.any(1)].tolist(),flush=True)
 for i in range(len(idx)):
  print(idx[i],st[i],np.round(turns[i],2).tolist(),flush=True)
# detailed stroboscopic trajectories for best electrostatic scenario, and matched same-field controls
for i in [59,83,80,81,82]:
 y,p,m,q,g=[D[k][i] for k in ['ys','ps','ms','qs','gs']];st,s,mi,ct,rs,fin,tr=one(y,p,m,q,g,192,4000);np.savez(f'trace_{i}.npz',st=st,s=s,mi=mi,ct=ct,rs=rs,fin=fin,tr=tr,periods=4000,steps=192,idx=i)
