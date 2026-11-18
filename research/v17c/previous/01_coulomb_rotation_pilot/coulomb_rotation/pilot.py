import numpy as np
from numba import njit,prange
from pathlib import Path
@njit
def rhs(y,t,k,c,g,mode):
 n=len(y);d=np.empty_like(y);ct=np.cos(t);st=np.sin(t) if mode==0 else 0.
 for i in range(n):
  x,b,vx,vy=y[i]; re=x*(x*x-3*b*b); im=b*(3*x*x-b*b)
  d[i,0]=vx;d[i,1]=vy;d[i,2]=-4*k*(re*ct+im*st)-g*vx;d[i,3]=4*k*(im*ct-re*st)-g*vy
  for j in range(n):
   if i!=j:
    dx=x-y[j,0];dy=b-y[j,1];r2=max(dx*dx+dy*dy,1e-24);fac=c/(r2*np.sqrt(r2));d[i,2]+=fac*dx;d[i,3]+=fac*dy
 return d
@njit
def one(y,k,c,g,mode,phase,steps,periods):
 dt=2*np.pi/steps;n=len(y);sums=np.zeros((3,n,3));mins=np.ones((3,n))*1e8;maxr=0.;minsep=1e8;t=phase
 for it in range(periods*steps):
  a=rhs(y,t,k,c,g,mode);b=rhs(y+dt/2*a,t+dt/2,k,c,g,mode);d=rhs(y+dt/2*b,t+dt/2,k,c,g,mode);e=rhs(y+dt*d,t+dt,k,c,g,mode)
  old=y;y=y+dt/6*(a+2*b+2*d+e);t+=dt
  cx=0.;cy=0.;ox=0.;oy=0.
  for i in range(n):
   cx+=y[i,0]/n;cy+=y[i,1]/n;ox+=old[i,0]/n;oy+=old[i,1]/n
  for i in range(n):
   x,b,vx,vy=y[i];r=np.sqrt(x*x+b*b);maxr=max(maxr,r)
   if not np.isfinite(r) or r>=1: return 0,sums,mins,maxr,minsep,y
   for j in range(i): minsep=min(minsep,np.sqrt((x-y[j,0])**2+(b-y[j,1])**2))
   if it>=periods*steps*0.4:
    w=min(2,int((it-periods*steps*0.4)/(periods*steps*0.2)));px,py=old[i,0],old[i,1];ux,uy=x-cx,b-cy;qx,qy=px-ox,py-oy
    sums[w,i,0]+=np.arctan2(px*b-py*x,px*x+py*b);sums[w,i,1]+=np.arctan2(qx*uy-qy*ux,qx*ux+qy*uy);sums[w,i,2]+=x*vy-b*vx;mins[w,i]=min(mins[w,i],r)
 return 1,sums,mins,maxr,minsep,y
@njit(parallel=True)
def batch(ys,cs,gs,modes,phases,k,steps,periods):
 m=len(ys);n=ys.shape[1];out=np.zeros((m,3,n,3));ms=np.zeros((m,3,n));status=np.zeros(m);rs=np.zeros((m,2));final=np.zeros_like(ys)
 for j in prange(m):
  ok,s,mi,mr,sep,y=one(ys[j],k,cs[j],gs[j],modes[j],phases[j],steps,periods);status[j]=ok;out[j]=s;ms[j]=mi;rs[j,0]=mr;rs[j,1]=sep;final[j]=y
 return status,out,ms,rs,final

def main():
 k=.1044;rng=np.random.default_rng(12456);root=Path(__file__).parent
 for n in [2,3,4,6,8]:
  ys=[];cs=[];gs=[];modes=[];phases=[];ids=[]
  for c in [2.14e-10,1e-7,1e-5,1e-3]:
   for gamma in [300,3000,10000]:
    for seed in range(4):
     r=(max(n-1,1)*c/(48*k*k))**(1/7);a=np.arange(n)*2*np.pi/n+rng.uniform(0,2*np.pi);y=np.zeros((n,4));y[:,0]=r*np.cos(a);y[:,1]=r*np.sin(a);y[:,:2]+=rng.normal(0,.13*r,(n,2));y[:,2:]=rng.normal(0,.3*4*k*r**3,(n,2));p=rng.uniform(0,2*np.pi)
     for mode in [0,1]: ys.append(y.copy());cs.append(c);gs.append(gamma/(2*np.pi*1e5));modes.append(mode);phases.append(p);ids.append((c,gamma,seed,mode))
  ys=np.array(ys);cs=np.array(cs);gs=np.array(gs);modes=np.array(modes);phases=np.array(phases);steps=48;periods=400
  status,out,ms,rs,fin=batch(ys,cs,gs,modes,phases,k,steps,periods)
  fs=out[:,:,:,0]/(periods*.2*2*np.pi)*1e5;fc=out[:,:,:,1]/(periods*.2*2*np.pi)*1e5
  def mixed(f): return (np.any(np.min(f,axis=1)>1,axis=1)&np.any(np.max(f,axis=1)<-1,axis=1)&(status==1))
  np.savez(root/f'raw_n{n}.npz',ys=ys,cs=cs,gs=gs,modes=modes,phases=phases,status=status,out=out,mins=ms,rs=rs,final=fin,ids=ids,periods=periods,steps=steps)
  print(n,'retained',int(sum(status)),'mixed',np.where(mixed(fs))[0],'relative',np.where(mixed(fc))[0],flush=True)
if __name__=='__main__':main()
