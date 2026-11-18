import os
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
from model import batch
from pathlib import Path
from scipy.optimize import minimize
root=Path(__file__).parent

def initial(n,p,m,q,seed,spin):
 rng=np.random.default_rng(seed);h=n//2;k,c,e,az,w,d,mode,re=p
 th=np.r_[np.arange(h)*2*np.pi/h,np.arange(n-h)*2*np.pi/(n-h)+.3]+rng.uniform(0,2*np.pi);x=np.column_stack((.32*np.cos(th),.32*np.sin(th),np.r_[np.full(h,-d),np.full(n-h,d)]))
 def energy(x):
  x=x.reshape(n,3);r2=(x[:,:2]**2).sum(1);z=x[:,2];a=8*k*k*q*q/m*(.5 if mode==1 else 1);E=(a*r2**3).sum()
  if az==0:E+=(q*.5*w*w*(z*z-r2/2)).sum()
  elif az==1:E+=(m*w*w*(z*z-d*d)**2/(8*d*d)).sum()
  else:E+=(q*w*w/(8*d*d)*(z**4-3*z*z*r2+3/8*r2*r2-2*d*d*(z*z-r2/2))).sum()
  dd=x[:,None]-x[None,:];s=(dd**2).sum(2)+np.eye(n);ip=1/np.sqrt(s);np.fill_diagonal(ip,0);return E+c/2*(q[:,None]*q[None,:]*ip).sum()
 x=minimize(energy,x.ravel(),method='L-BFGS-B',bounds=[(-.8,.8),(-.8,.8),(-1.2,1.2)]*n,options={'maxiter':400,'ftol':1e-12,'gtol':1e-8}).x.reshape(n,3);u=x[:,0]+1j*x[:,1];v=1j*spin*np.where(np.arange(n)<h,-1,1)*u;chir=1 if mode==0 else (0 if mode==1 else x[:,2]/d)
 v+=4*k*q/m*chir*((u**3).imag+1j*(u**3).real)-8*k*e*q/m*((u**7).imag+1j*(u**7).real)
 return np.column_stack((x,v.real,v.imag,np.zeros(n)))
if __name__=='__main__':
 n=6;ys=[];ps=[];ms=[];qs=[];gs=[];meta=[]
 for az in [0,1,2]:
  for c in [3e-5,2e-4]:
   for w,d in [(.045,.45),(.12,.65)]:
    for mr,qr,gr in [(1,1,1),(4,1,.3)]:
     for mode,e in [(0,0),(0,.32945),(1,0),(2,0)]:
      p=np.array([.1044,c,e,az,w,d,mode,.3]);m=np.r_[np.full(n//2,mr),np.ones(n-n//2)];q=np.r_[np.full(n//2,qr),np.ones(n-n//2)];g=np.r_[np.full(n//2,.003*gr),np.full(n-n//2,.003)];seed=1234+len(ps)//4;ys.append(initial(n,p,m,q,seed,.015));ps.append(p);ms.append(m);qs.append(q);gs.append(g);meta.append([mr,qr,gr,seed])
 ys,ps,ms,qs,gs,meta=map(np.array,(ys,ps,ms,qs,gs,meta));steps=48;periods=600;np.savez(root/'inputs.npz',ys=ys,ps=ps,ms=ms,qs=qs,gs=gs,meta=meta)
 for i in range(0,len(ys),12):
  sl=slice(i,i+12);st,s,mi,ct,rs,fin=batch(ys[sl],ps[sl],ms[sl],qs[sl],gs[sl],steps,periods);np.savez(root/f'scout_{i}.npz',st=st,s=s,mi=mi,ct=ct,rs=rs,fin=fin,steps=steps,periods=periods)
  turns=s[:,:,:,0]/(2*np.pi);good=mi.min(1)>.015;pos=(turns.min(1)>.1)&good;neg=(turns.max(1)<-.1)&good;print(i,'retained',sum(st),'candidate',(np.where((st==1)&pos.any(1)&neg.any(1))[0]+i).tolist(),flush=True)
