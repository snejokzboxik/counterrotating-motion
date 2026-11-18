import numpy as np
from numba import njit,prange
@njit(cache=True)
def rhs(y,t,p,m,q,g):
 k,C,e,az,w,d,mode,re=p;n=len(y);out=np.empty_like(y);a=np.exp(-1j*t);b=a.conjugate()
 for i in range(n):
  x,h,z,vx,vy,vz=y[i];u=complex(x,h);A=a;Dz=0j
  if mode==1:A=complex(a.real,0)
  if mode==2:A=(1+z/d)*a/2+(1-z/d)*b/2;Dz=(a-b)/(2*d)
  D=k*(4*u**3*A+8*e*u**7*b);fx=-D.real;fy=D.imag;fz=-(k*u**4*Dz).real
  if az==0:fx+=w*w*x/2;fy+=w*w*h/2;fz-=w*w*z
  elif az==1:fz-=w*w*z*(z*z-d*d)/(2*d*d)*m[i]/q[i]
  else:
   r2=x*x+h*h;v=w*w/(8*d*d);fx-=v*x*(-6*z*z+1.5*r2+2*d*d);fy-=v*h*(-6*z*z+1.5*r2+2*d*d);fz-=v*(4*z**3-6*z*r2-4*d*d*z)
  drag=g[i]*(1+3/16*re*np.sqrt(vx*vx+vy*vy+vz*vz));out[i,0]=vx;out[i,1]=vy;out[i,2]=vz;out[i,3]=q[i]/m[i]*fx-drag*vx;out[i,4]=q[i]/m[i]*fy-drag*vy;out[i,5]=q[i]/m[i]*fz-drag*vz
 for i in range(n):
  for j in range(i):
   dx=y[i,0]-y[j,0];dy=y[i,1]-y[j,1];dz=y[i,2]-y[j,2];r2=max(dx*dx+dy*dy+dz*dz,1e-24);f=C*q[i]*q[j]/(r2*np.sqrt(r2))
   for h in range(3):
    dd=y[i,h]-y[j,h];out[i,h+3]+=f*dd/m[i];out[j,h+3]-=f*dd/m[j]
 return out
@njit(cache=True)
def one(y,p,m,q,g,steps,periods):
 y=y.copy();n=len(y);dt=2*np.pi/steps;s=np.zeros((3,n,4));mi=np.full((3,n),1e9);ct=np.zeros(3,np.int64);rs=np.array([0.,0.,1e9,0.]);tr=np.zeros((periods//5+1,n,6));tr[0]=y;nt=1
 for it in range(steps*periods):
  t=2*np.pi*(it%steps)/steps;a=rhs(y,t,p,m,q,g);b=rhs(y+dt/2*a,t+dt/2,p,m,q,g);c=rhs(y+dt/2*b,t+dt/2,p,m,q,g);d=rhs(y+dt*c,t+dt,p,m,q,g);old=y;y=y+dt/6*(a+2*b+2*c+d)
  wi=-1
  if it>=steps*periods*2//5:wi=min(2,(it-steps*periods*2//5)*5//(steps*periods));ct[wi]+=1
  for i in range(n):
   x,h,z,vx,vy,vz=y[i];r=np.sqrt(x*x+h*h);v=np.sqrt(vx*vx+vy*vy+vz*vz);rs[0]=max(rs[0],r);rs[1]=max(rs[1],abs(z));rs[3]=max(rs[3],p[7]*v)
   if not np.isfinite(r+z+v) or r>=1 or abs(z)>=1.5:return 0,s,mi,ct,rs,y,tr[:nt]
   if wi>=0:
    px,py=old[i,0],old[i,1];s[wi,i,0]+=np.arctan2(px*h-py*x,px*x+py*h);s[wi,i,1]+=r;s[wi,i,2]+=z;s[wi,i,3]+=m[i]*(x*vy-h*vx);mi[wi,i]=min(mi[wi,i],r)
   for j in range(i):
    dx=x-y[j,0];dy=h-y[j,1];dz=z-y[j,2];rs[2]=min(rs[2],np.sqrt(dx*dx+dy*dy+dz*dz))
  if (it+1)%(5*steps)==0:tr[nt]=y;nt+=1
 return 1,s,mi,ct,rs,y,tr[:nt]
@njit(cache=True,parallel=True)
def batch(ys,ps,ms,qs,gs,steps,periods):
 N=len(ys);n=ys.shape[1];st=np.zeros(N,np.int64);s=np.zeros((N,3,n,4));mi=np.zeros((N,3,n));ct=np.zeros((N,3),np.int64);rs=np.zeros((N,4));fin=np.zeros_like(ys)
 for i in prange(N):
  a,b,c,d,e,f,tr=one(ys[i],ps[i],ms[i],qs[i],gs[i],steps,periods);st[i]=a;s[i]=b;mi[i]=c;ct[i]=d;rs[i]=e;fin[i]=f
 return st,s,mi,ct,rs,fin
