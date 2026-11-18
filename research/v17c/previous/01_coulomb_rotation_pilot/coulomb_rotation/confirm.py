from pilot import *
from scipy.integrate import solve_ivp
import json
root=Path(__file__).parent
for n in [2,3,4,6,8]:
 d=np.load(root/f'raw_n{n}.npz'); f=d['out'][:,:,:,0]/(80*2*np.pi)*1e5; h=d['out'][:,:,:,1]/(80*2*np.pi)*1e5
 op=lambda x: np.any(np.min(x,axis=1)>1,axis=1)&np.any(np.max(x,axis=1)<-1,axis=1)&(d['status']==1)
 idx=np.where(op(f)|op(h))[0];
 # focus on all higher-c cases and selected lower-c examples in each field
 sel=list(idx[d['cs'][idx]>1e-9]);sel+=list(idx[d['cs'][idx]<1e-9][::3]);idx=np.array(sorted(set(sel)),dtype=int)
 if not len(idx):continue
 status,out,ms,rs,fin=batch(d['ys'][idx],d['cs'][idx],d['gs'][idx],d['modes'][idx],d['phases'][idx],.1044,96,3000)
 np.savez(root/f'confirm_n{n}.npz',idx=idx,status=status,out=out,mins=ms,rs=rs,final=fin,periods=3000,steps=96)
 f=out[:,:,:,0]/(600*2*np.pi)*1e5;h=out[:,:,:,1]/(600*2*np.pi)*1e5
 op2=lambda x: np.any(np.min(x,axis=1)>1,axis=1)&np.any(np.max(x,axis=1)<-1,axis=1)&(status==1)
 print(n,'tested',len(idx),'ret',sum(status),'axis',idx[op2(f)],'COM',idx[op2(h)],flush=True)
 for j in np.where(op2(f)|op2(h))[0]:print('case',idx[j],d['ids'][idx[j]],'axis',np.round(f[j],2),'COM',np.round(h[j],2),'rmin',np.round(ms[j].min(axis=0),6),flush=True)
