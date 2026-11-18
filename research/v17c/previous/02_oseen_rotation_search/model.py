import numpy as np
from numba import njit,prange
from pathlib import Path
@njit
def rhs(y,t,k,c,g,mode,re,eps):
 n=len(y);d=np.empty_like(y);ct=np.cos(t);st=np.sin(t) if mode==0 else 0.
 for i in range(n):
  x,b,vx,vy=y[i]; ar=x*(x*x-3*b*b); im=b*(3*x*x-b*b); drag=g*(1+3/16*re*np.sqrt(vx*vx+vy*vy))
  d[i,0]=vx;d[i,1]=vy;d[i,2]=-4*k*(ar*ct+im*st)-drag*vx;d[i,3]=4*k*(im*ct-ar*st)-drag*vy
  if mode==2:
   d[i,2]=-2*k*x*ct-drag*vx; d[i,3]=2*k*b*ct-drag*vy
  if eps!=0:
   a4=ar*x-im*b; b4=im*x+ar*b; a7=a4*ar-b4*im; b7=a4*im+b4*ar
   d[i,2]-=8*k*eps*(a7*ct-b7*st); d[i,3]+=8*k*eps*(b7*ct+a7*st)
  for j in range(n):
   if i!=j:
    dx=x-y[j,0];dy=b-y[j,1];r2=max(dx*dx+dy*dy,1e-24);fac=c/(r2*np.sqrt(r2));d[i,2]+=fac*dx;d[i,3]+=fac*dy
 return d
@njit
def one(y,k,c,g,mode,re,eps,phase,steps,periods):
 dt=2*np.pi/steps;n=len(y);sums=np.zeros((3,n,3));mins=np.ones((3,n))*1e8;maxre=0.;maxr=0.;minsep=1e8;t=phase
 for it in range(periods*steps):
  a=rhs(y,t,k,c,g,mode,re,eps);b=rhs(y+dt/2*a,t+dt/2,k,c,g,mode,re,eps);d=rhs(y+dt/2*b,t+dt/2,k,c,g,mode,re,eps);e=rhs(y+dt*d,t+dt,k,c,g,mode,re,eps)
  old=y;y=y+dt/6*(a+2*b+2*d+e);t+=dt
  cx=0.;cy=0.;ox=0.;oy=0.
  for i in range(n):
   cx+=y[i,0]/n;cy+=y[i,1]/n;ox+=old[i,0]/n;oy+=old[i,1]/n
  for i in range(n):
   x,b,vx,vy=y[i];r=np.sqrt(x*x+b*b);maxr=max(maxr,r);maxre=max(maxre,re*np.sqrt(vx*vx+vy*vy))
   if not np.isfinite(r) or r>=1: return 0,sums,mins,maxr,minsep,maxre,y
   for j in range(i): minsep=min(minsep,np.sqrt((x-y[j,0])**2+(b-y[j,1])**2))
   if it>=periods*steps*0.4:
    w=min(2,int((it-periods*steps*0.4)/(periods*steps*0.2)));px,py=old[i,0],old[i,1];ux,uy=x-cx,b-cy;qx,qy=px-ox,py-oy
    sums[w,i,0]+=np.arctan2(px*b-py*x,px*x+py*b);sums[w,i,1]+=np.sqrt(x*x+b*b);sums[w,i,2]+=x*vy-b*vx;mins[w,i]=min(mins[w,i],r)
 return 1,sums,mins,maxr,minsep,maxre,y
@njit(parallel=True)
def batch(ys,cs,gs,modes,res,epss,phases,k,steps,periods):
 m=len(ys);n=ys.shape[1];out=np.zeros((m,3,n,3));ms=np.zeros((m,3,n));status=np.zeros(m);rs=np.zeros((m,3));final=np.zeros_like(ys)
 for j in prange(m):
  ok,s,mi,mr,sep,mre,y=one(ys[j],k,cs[j],gs[j],modes[j],res[j],epss[j],phases[j],steps,periods);status[j]=ok;out[j]=s;ms[j]=mi;rs[j,0]=mr;rs[j,1]=sep;rs[j,2]=mre;final[j]=y
 return status,out,ms,rs,final

