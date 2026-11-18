import numpy as np
from model import batch
from pathlib import Path
root=Path(__file__).parent
z=np.load(root/'relaxed12.npz');y=z['ys'][15];p=z['p'][15,:6];ys=np.repeat(y[None],6,axis=0);ps=np.repeat(p[None],6,axis=0);ps[1,3]=0;ps[2,4]=0;ps[3,2]=1;ps[3,4]=0;ps[4,3]=0;ps[4,4]=0;ps[5,2]=2;ps[5,4]=0
for steps,periods in [(96,1000),(96,6000),(192,6000)]:
 st,o,mi,rs,fin=batch(ys,ps[:,0],ps[:,1],ps[:,2].astype(int),ps[:,3],ps[:,4],ps[:,5],.1044,steps,periods)
 np.savez(root/f'confirm_{steps}_{periods}.npz',ys=ys,p=ps,st=st,o=o,mi=mi,rs=rs,fin=fin,steps=steps,periods=periods)
 turns=o[:,:,:,0]/(2*np.pi);print(steps,periods,'status',st,flush=True);print(np.round(turns,3),flush=True)
